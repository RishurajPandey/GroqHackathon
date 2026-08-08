from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable,
    AgeRestricted,
    RequestBlocked,
    IpBlocked,
    InvalidVideoId,
)
import validators
import yt_dlp
import json
import tempfile
import glob

from flask_cors import CORS
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

# Hard ceiling on transcript size — this is the "truly out of bounds" cutoff, not a
# quality trade-off. ~300k characters is roughly a 5-6 hour continuous-speech video,
# comfortably covering long lectures/podcasts. Beyond this we fail fast with a clear
# error instead of grinding through hours of sequential/parallel LLM calls.
MAX_TRANSCRIPT_CHARS = 300000

# Once the running summary shrinks below this, it's small enough to safely hand to
# the final HTML-formatting LLM call in one shot.
REDUCE_TARGET_CHARS = 6000
MAX_REDUCE_ROUNDS = 5

# Gemma2-9b-It (previously used throughout this file) has been decommissioned by
# Groq and now fails every request with a 400 model_decommissioned error — this
# was silently breaking every LLM call in the video/QA/fake-news pipelines.
GROQ_MODEL = "llama-3.3-70b-versatile"

class Summary(BaseModel):
    summary: str = Field(description="The generated summary")

class QuestionAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")

class FakeNewsAnalysis(BaseModel):
    is_fake: bool = Field(description="Whether the news is likely fake")
    confidence: float = Field(description="Confidence score of the analysis")
    reasons: list[str] = Field(description="List of reasons supporting the analysis")
    suggestions: list[str] = Field(description="Suggestions for fact-checking")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:8080",
    "http://localhost:8081",
    "https://brieflensnews.onrender.com"
]}})


def _split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents([Document(page_content=text)])


def _translate_to_english(text, llm, source_language):
    """Translates arbitrary-language text to English, chunk-by-chunk and in
    parallel. Naming the detected source language in the prompt (rather than
    leaving the model to infer it) measurably improves translation quality.
    Chunks that fail to translate fall back to their original text rather than
    being dropped, so a single bad chunk can't sink the whole transcript."""
    translation_prompt = PromptTemplate(
        template="""
        Translate the following {language} text to English. Maintain the
        original meaning and context. Return only the English translation
        with no extra commentary.

        {language} text:
        {text}
        """,
        input_variables=["text", "language"]
    )
    translation_chain = translation_prompt | llm

    def translate_chunk(doc):
        try:
            return translation_chain.invoke({"text": doc.page_content, "language": source_language}).content
        except Exception:
            return doc.page_content

    docs = _split_text(text, chunk_size=3000, chunk_overlap=0)
    with ThreadPoolExecutor(max_workers=min(6, len(docs))) as executor:
        translated_parts = list(executor.map(translate_chunk, docs))
    return " ".join(translated_parts)


def _map_summarize(text, chain):
    """One map pass: split text into chunks and summarize each chunk in
    parallel. Used both for the initial transcript and, recursively, for
    reducing an over-long combined summary — this is what lets arbitrarily
    long transcripts be summarized without truncating any content."""
    documents = _split_text(text)
    if not documents:
        raise ValueError("No documents created after splitting")

    def summarize_chunk(chunk):
        try:
            result = chain.invoke({"input_text": chunk.page_content})
            return result
        except Exception:
            return {"summary": chunk.page_content[:500] + "..."}

    with ThreadPoolExecutor(max_workers=min(6, len(documents))) as executor:
        partial_summaries = list(executor.map(summarize_chunk, documents))
    return " ".join(item['summary'] for item in partial_summaries)


def summarize_video_pipeline(original_text):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)

    if not original_text or not original_text.strip():
        raise ValueError("Empty input text provided")

    # Detect language from a sample only — the whole transcript can be huge and
    # language doesn't need the full text to identify.
    language_detection_prompt = PromptTemplate(
        template="""
        Identify the primary language of the following text. Respond with only
        the language's English name (e.g. "English", "Hindi", "Spanish",
        "French", "Japanese"). Nothing else.

        Text:
        {text}
        """,
        input_variables=["text"]
    )
    language_chain = language_detection_prompt | llm
    detected_language = language_chain.invoke({"text": original_text[:3000]}).content.strip()

    was_translated = False
    if "english" not in detected_language.lower():
        original_text = _translate_to_english(original_text, llm, detected_language)
        was_translated = True

    parser = JsonOutputParser(pydantic_object=Summary)
    prompt = PromptTemplate(
        template="""
        You are a professional summarization assistant.
        Generate a JSON-formatted summary of the following text chunk.
        The output MUST contain only JSON with a 'summary' field.

        Text chunk:
        {input_text}

        {format_instructions}
        """,
        input_variables=["input_text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )
    chain = prompt | llm | parser

    # Map, then recursively reduce: repeatedly re-summarize the combined
    # summary until it's short enough to format in one final pass. This lets
    # transcripts of any length converge to a fixed-size final input instead
    # of being truncated. MAX_REDUCE_ROUNDS is just a safety valve against a
    # pathological case where summaries stop shrinking.
    combined_summary = _map_summarize(original_text, chain)
    rounds = 0
    while len(combined_summary) > REDUCE_TARGET_CHARS and rounds < MAX_REDUCE_ROUNDS:
        combined_summary = _map_summarize(combined_summary, chain)
        rounds += 1
    final_combined_summary = combined_summary

    # Final formatting
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a formatting assistant. Format this summary with proper HTML tags:
         - Use <h2> for main headings
         - Use <h3> for subheadings
         - Use <p> for paragraphs
         - Use <ul> and <li> for bullet points
         - Use <strong> for important text
         - Use <em> for emphasis
         - Use <br> for line breaks
         - Ensure all HTML tags are properly closed
         - Do not use markdown syntax (** or *)
         - Make sure the output is valid HTML"""),
        ("user", "{input}")
    ])

    final_chain = final_prompt | llm
    final_summary = final_chain.invoke({"input": final_combined_summary})

    return {
        "summary": final_summary.content,
        "detectedLanguage": detected_language,
        "wasTranslated": was_translated,
    }

@app.route("/")
def home():
    return "Flask app is running!"

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        original_text = data.get("originalText")
        if not original_text:
            return jsonify({"error": "Missing 'originalText' field"}), 400

        pipeline_result = summarize_video_pipeline(original_text)
        return jsonify({
            "summarizedText": pipeline_result["summary"],
            "detectedLanguage": pipeline_result["detectedLanguage"],
            "wasTranslated": pipeline_result["wasTranslated"],
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}


def extract_video_id(url):
    """Extracts an 11-char YouTube video ID from any common URL shape:
    watch?v=, youtu.be/, /shorts/, /embed/, /live/, on youtube.com,
    m.youtube.com, music.youtube.com, or the nocookie embed domain."""
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            for prefix in ("/shorts/", "/embed/", "/live/"):
                if parsed.path.startswith(prefix):
                    candidate = parsed.path[len(prefix):].split("/")[0]
                    break

    if candidate and _VIDEO_ID_RE.match(candidate):
        return candidate
    return None


def fetch_youtube_transcript(video_id):
    """Fetches a transcript in any available language, returning
    (transcript_text, language_code, was_translated_by_youtube). Preference
    order: manually-created English > auto-generated English > any transcript
    YouTube can translate to English > the first available transcript as-is
    (the summarization pipeline will detect its language and translate it).
    Raises the underlying youtube_transcript_api exception on total failure so
    the caller can map it to a specific, honest error message."""
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    available = list(transcript_list)

    if not available:
        raise NoTranscriptFound(video_id, [], transcript_list)

    def fetch_text(transcript_obj):
        fetched = transcript_obj.fetch()
        return " ".join(snippet.text for snippet in fetched)

    manual_en = next((t for t in available if t.language_code.startswith("en") and not t.is_generated), None)
    if manual_en:
        return fetch_text(manual_en), manual_en.language_code, False

    auto_en = next((t for t in available if t.language_code.startswith("en")), None)
    if auto_en:
        return fetch_text(auto_en), auto_en.language_code, False

    for t in available:
        if t.is_translatable:
            try:
                translated = t.translate("en")
                return fetch_text(translated), t.language_code, True
            except Exception:
                continue

    # No English transcript and no translatable one — take whatever exists in
    # its native language; summarize_video_pipeline() will translate it.
    fallback = available[0]
    return fetch_text(fallback), fallback.language_code, False


def fetch_youtube_transcript_ytdlp(video_id):
    """Second-line transcript source, tried only when youtube_transcript_api
    fails. yt-dlp is a separately-maintained extractor (much larger contributor
    base, typically patched within hours of a YouTube-side change) with its own
    request/client patterns, so its failure modes don't fully correlate with
    youtube_transcript_api's — this exists purely to raise availability, not to
    replace the primary source. Returns (transcript_text, language_code)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get("subtitles") or {}
        autosubs = info.get("automatic_captions") or {}

        def pick_english(tracks):
            if "en" in tracks:
                return "en", tracks["en"]
            for lang, fmts in tracks.items():
                if lang.startswith("en"):
                    return lang, fmts
            return None, None

        lang, track = pick_english(subs)
        if track is None:
            lang, track = pick_english(autosubs)
        if track is None and subs:
            lang, track = next(iter(subs.items()))
        if track is None and autosubs:
            lang, track = next(iter(autosubs.items()))
        if track is None:
            raise RuntimeError("yt-dlp found no caption tracks for this video")

        fmt = next((f for f in track if f.get("ext") == "json3"), track[0])
        raw = ydl.urlopen(fmt["url"]).read()
        data = json.loads(raw)
        text_parts = [
            seg["utf8"]
            for event in data.get("events", [])
            for seg in (event.get("segs") or [])
            if seg.get("utf8")
        ]
        text = "".join(text_parts).replace("\n", " ").strip()
        if not text:
            raise RuntimeError("yt-dlp returned an empty transcript")
        return text, lang


GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# Groq's Whisper endpoint rejects files above this size — checked after download
# so we fail with a clear message instead of a doomed multi-minute upload.
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


def fetch_youtube_transcript_via_audio(video_id):
    """Last-resort transcript source: downloads just the audio track — a
    heavily-used, first-class yt-dlp feature, far sturdier than its caption-
    scraping path — and transcribes it with Groq's Whisper API (the same model
    already used for the /audio upload endpoint). This works on videos with no
    captions at all and doesn't touch YouTube's fragile, rate-limited caption
    endpoint, so it's tried only after both caption-based sources fail — it
    costs real time and API usage per call. Returns (transcript_text, language).
    """
    if not os.environ.get('GROQ_API_KEY'):
        raise RuntimeError("GROQ_API_KEY is not configured")

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = [f for f in glob.glob(os.path.join(tmpdir, "*")) if os.path.isfile(f)]
        if not files:
            raise RuntimeError("yt-dlp did not produce an audio file")
        audio_path = max(files, key=os.path.getsize)

        size = os.path.getsize(audio_path)
        if size > MAX_AUDIO_BYTES:
            raise RuntimeError(
                f"Audio track is too large to transcribe ({size / 1024 / 1024:.1f} MB, "
                f"limit {MAX_AUDIO_BYTES / 1024 / 1024:.0f} MB)"
            )

        with open(audio_path, "rb") as f:
            response = requests.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"},
                files={"file": (os.path.basename(audio_path), f)},
                data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
                timeout=180,
            )
        response.raise_for_status()
        result = response.json()
        text = (result.get("text") or "").strip()
        if not text:
            raise RuntimeError("Whisper returned an empty transcript")
        return text, result.get("language")


@app.route('/summarize-video', methods=['POST'])
def summarize_video():
    data = request.get_json(silent=True)
    if not data or 'videoUrl' not in data:
        return jsonify({"error": "missing_field", "message": "Missing 'videoUrl' field"}), 400

    url = data['videoUrl']
    if not isinstance(url, str) or not validators.url(url):
        return jsonify({"error": "invalid_url", "message": "Please provide a valid video URL"}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "invalid_url", "message": "Could not recognize a YouTube video ID in that URL. Only YouTube links are currently supported."}), 400

    try:
        app.logger.info(f"Fetching transcript for video ID: {video_id}")
        try:
            transcript_text, source_lang, _ = fetch_youtube_transcript(video_id)
            app.logger.info(f"Transcript fetched via youtube_transcript_api: {len(transcript_text)} chars, source language '{source_lang}'")
        except Exception as primary_error:
            app.logger.warning(f"youtube_transcript_api failed ({type(primary_error).__name__}: {primary_error}); trying yt-dlp caption fallback")
            try:
                transcript_text, source_lang = fetch_youtube_transcript_ytdlp(video_id)
                app.logger.info(f"Transcript fetched via yt-dlp captions: {len(transcript_text)} chars, source language '{source_lang}'")
            except Exception as caption_fallback_error:
                app.logger.warning(f"yt-dlp caption fallback failed ({caption_fallback_error}); trying audio+Whisper fallback")
                try:
                    transcript_text, source_lang = fetch_youtube_transcript_via_audio(video_id)
                    app.logger.info(f"Transcript fetched via audio+Whisper: {len(transcript_text)} chars, detected language '{source_lang}'")
                except Exception as audio_fallback_error:
                    app.logger.warning(f"Audio+Whisper fallback also failed: {audio_fallback_error}")
                    # If the video genuinely has no captions/is unavailable, that's
                    # still the clearest thing to tell the user even though audio
                    # also failed (its failure is usually a symptom of the same
                    # underlying issue). Otherwise, the caption failure reason
                    # (e.g. "rate limited") is no longer the real story once audio
                    # has failed for its own, different reason — say so honestly.
                    if isinstance(primary_error, (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, VideoUnplayable, AgeRestricted, InvalidVideoId)):
                        raise primary_error
                    raise RuntimeError(
                        f"All transcript sources failed. Captions: {primary_error}. Audio transcription: {audio_fallback_error}"
                    )
    except TranscriptsDisabled:
        return jsonify({"error": "no_captions", "message": "Captions are disabled for this video, so it can't be summarized."}), 404
    except NoTranscriptFound:
        return jsonify({"error": "no_captions", "message": "No captions are available for this video in any language."}), 404
    except (VideoUnavailable, VideoUnplayable):
        return jsonify({"error": "video_unavailable", "message": "This video is unavailable (private, deleted, or region-locked)."}), 400
    except AgeRestricted:
        return jsonify({"error": "age_restricted", "message": "This video is age-restricted, and its captions can't be accessed."}), 400
    except InvalidVideoId:
        return jsonify({"error": "invalid_url", "message": "That doesn't look like a valid YouTube video ID."}), 400
    except (RequestBlocked, IpBlocked):
        app.logger.warning(f"YouTube blocked transcript request for {video_id}")
        return jsonify({"error": "rate_limited", "message": "YouTube is temporarily blocking transcript requests from our server. Please try again in a few minutes."}), 503
    except Exception as e:
        app.logger.error(f"Transcript fetch failed: {str(e)}", exc_info=True)
        return jsonify({"error": "transcript_fetch_failed", "message": f"Transcript fetch failed: {str(e)}"}), 500

    if len(transcript_text.strip()) < 50:
        return jsonify({"error": "transcript_too_short", "message": "Transcript too short to summarize"}), 400

    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        return jsonify({
            "error": "transcript_too_long",
            "message": f"This video's transcript is too long to summarize in one request ({len(transcript_text):,} characters, limit {MAX_TRANSCRIPT_CHARS:,}). Try a shorter video or an excerpt."
        }), 413

    try:
        app.logger.info("Starting summarization pipeline")
        pipeline_result = summarize_video_pipeline(transcript_text)
        app.logger.info("Summarization completed successfully")
        return jsonify({
            "summarizedText": pipeline_result["summary"],
            "transcription": transcript_text,
            "detectedLanguage": pipeline_result["detectedLanguage"],
            "wasTranslated": pipeline_result["wasTranslated"],
            "captionLanguage": source_lang,
            "status": "success"
        })
    except Exception as e:
        app.logger.error(f"Summarization pipeline failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "summarization_failed",
            "message": f"Summarization failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/answer-question', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        summary = data.get("summary")
        question = data.get("question")
        
        if not summary or not question:
            return jsonify({"error": "Missing 'summary' or 'question' field"}), 400

        # Using a more capable model with higher temperature for more creative responses
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.9)
        parser = JsonOutputParser(pydantic_object=QuestionAnswer)
        
        prompt = PromptTemplate(
            template="""
            You are a comprehensive AI assistant with access to current information and the ability to perform web searches.
            Your goal is to provide the most accurate and up-to-date answers possible.
            
            CRITICAL INSTRUCTION: NEVER say "the summary doesn't contain this information" or similar phrases.
            Instead, actively search for and provide the information using your knowledge base and web search capabilities.
            
            Guidelines:
            1. Knowledge Utilization:
               - First, check if the answer can be found in the provided summary
               - If not in summary, IMMEDIATELY search your knowledge base and perform a web search
               - Always provide the most relevant information, regardless of source
               - Combine information from multiple sources when relevant
               - Never indicate that information is missing from the summary
            
            2. Answer Structure:
               - Start with the most relevant information (whether from summary or other sources)
               - Add context and additional details
               - Include specific facts, dates, and statistics
               - Provide examples and real-world applications
               - Use proper formatting for clarity
            
            3. Information Gathering:
               - Actively search for information when not in summary
               - Use your knowledge base extensively
               - Perform web searches for recent or specific information
               - Cross-reference multiple sources
               - Provide the most up-to-date information available
            
            4. Special Cases:
               - For sports/events: Provide current statistics, results, and player information
               - For recent events: Include the latest developments
               - For technical topics: Provide detailed explanations
               - For opinion-based questions: Offer balanced perspectives
            
            5. Always:
               - Be thorough and detailed in responses
               - Provide accurate and up-to-date information
               - Never say information is missing or unavailable
               - Use clear and professional language
               - Maintain a helpful and informative tone
            
            Summary:
            {summary}
            
            Question:
            {question}
            
            {format_instructions}
            """,
            input_variables=["summary", "question"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        
        chain = prompt | llm | parser
        result = chain.invoke({"summary": summary, "question": question})
        
        return jsonify({
            "answer": result["answer"],
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/detect-fake-news', methods=['POST'])
def detect_fake_news():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400

        llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
        parser = JsonOutputParser(pydantic_object=FakeNewsAnalysis)
        
        prompt = PromptTemplate(
            template="""
            You are an expert fact-checker and fake news detector with access to current information up to April 2025. 
            Analyze the following news content and determine if it's likely fake or not.
            
            IMPORTANT: Be extremely conservative in labeling content as fake. Only label as fake if there is overwhelming evidence.
            
            Guidelines:
            1. Source Evaluation (Most Important):
               - Content from established news organizations is presumed credible unless proven otherwise
               - YouTube channels with large followings and good reputation are generally credible
               - Consider the channel's history of accuracy and reliability
               - Look for official partnerships or affiliations with credible organizations
               - Check if the content creator has relevant expertise or credentials
            
            2. Content Analysis Framework:
               a) Fact Verification:
                  - Separate facts from opinions
                  - Verify only factual claims, not opinions or analysis
                  - Look for specific, verifiable information
                  - Check dates, numbers, and specific claims
               
               b) Context Understanding:
                  - Consider the content's purpose (news, analysis, opinion)
                  - Understand the target audience
                  - Consider cultural and regional context
                  - Account for different reporting styles
            
            3. Verification Standards:
               - Require multiple independent sources for fake news claims
               - Official statements or documents are strong evidence
               - Expert consensus is important for technical claims
               - Historical context matters for current events
               - Consider the possibility of new information
            
            4. Red Flags (Require Multiple to Consider Fake):
               - Clear contradictions with established facts
               - Proven manipulation of images or videos
               - Demonstrated history of spreading misinformation
               - Lack of any credible sources
               - Clear evidence of fabrication
            
            5. Confidence Levels:
               - Very High (0.9-1.0): Multiple independent verifications, official documentation
               - High (0.7-0.8): Strong evidence from credible sources
               - Medium (0.5-0.6): Some verification possible
               - Low (0.3-0.4): Limited verification, mostly opinions
               - Very Low (0.0-0.2): Speculative content
            
            6. Special Considerations:
               - Breaking news may have incomplete information
               - Different perspectives don't necessarily mean fake news
               - Opinions and analysis are not fake news
               - Consider the possibility of new developments
               - Account for different reporting styles
            
            7. Output Requirements:
               a) Determination:
                  - Only label as fake with overwhelming evidence
                  - Consider "unverified" instead of "fake" when uncertain
                  - Acknowledge limitations in verification
               
               b) Evidence:
                  - Provide specific examples of false claims
                  - Reference credible sources for verification
                  - Explain the verification process
                  - Note any uncertainties
            
            8. Final Checks:
               - Is there overwhelming evidence of fabrication?
               - Are multiple independent sources confirming the falsehood?
               - Is this a matter of opinion rather than fact?
               - Could this be new information not yet widely known?
               - Is the source generally credible?
            
            News content:
            {text}
            
            {format_instructions}
            """,
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        
        chain = prompt | llm | parser
        result = chain.invoke({"text": text})
        
        # Additional validation of the result
        if result.get("is_fake", False):
            # Require higher confidence for fake news claims
            if result.get("confidence", 0) < 0.8:
                result["is_fake"] = False
                result["confidence"] = 1 - result["confidence"]
                result["reasons"] = ["Content could not be verified as fake with sufficient confidence"]
        
        return jsonify({
            "analysis": result,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
