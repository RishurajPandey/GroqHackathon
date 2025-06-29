from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
import validators

from flask_cors import CORS
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re

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
    "https://brieflensnews.onrender.com"
]}})


def summarize_video_pipeline(original_text):
    llm = ChatGroq(model="Gemma2-9b-It", temperature=0.3)
    
    if not original_text or not original_text.strip():
        raise ValueError("Empty input text provided")

    # First, detect if the text is in Hindi
    language_detection_prompt = PromptTemplate(
        template="""
        Analyze the following text and determine if it's primarily in Hindi or English.
        Return only 'hindi' or 'english' as your response.
        
        Text:
        {text}
        """,
        input_variables=["text"]
    )
    
    language_chain = language_detection_prompt | llm
    detected_language = language_chain.invoke({"text": original_text}).content.lower()
    
    # If the text is in Hindi, translate it to English first
    if "hindi" in detected_language:
        translation_prompt = PromptTemplate(
            template="""
            Translate the following Hindi text to English. Maintain the original meaning and context.
            Return only the English translation.
            
            Hindi text:
            {text}
            """,
            input_variables=["text"]
        )
        
        translation_chain = translation_prompt | llm
        original_text = translation_chain.invoke({"text": original_text}).content
    
    text_documents = [Document(page_content=original_text)]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    documents = text_splitter.split_documents(text_documents)
    
    if not documents:
        raise ValueError("No documents created after splitting")

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

    def summarize_chunk(chunk):
        try:
            result = chain.invoke({"input_text": chunk.page_content})
            return result
        except Exception as e:
            return {"summary": chunk.page_content[:500] + "..."}

    partial_summaries = [summarize_chunk(c) for c in documents]
    final_combined_summary = " ".join(item['summary'] for item in partial_summaries)

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
    
    return final_summary.content

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

        summarized_text = summarize_video_pipeline(original_text)
        return jsonify({
            "summarizedText": summarized_text,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    
def extract_video_id(url):
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return None
    
@app.route('/summarize-video', methods=['POST'])
def summarize_video():
    try:
        data = request.get_json()
        if not data or 'videoUrl' not in data:
            return jsonify({"error": "Missing 'videoUrl' field"}), 400

        url = data['videoUrl']
        if not validators.url(url):
            return jsonify({"error": "Invalid URL"}), 400

        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Could not extract video ID"}), 400

        try:
            app.logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
            
            # First try to get English transcript
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                app.logger.info("Successfully retrieved English transcript")
            except Exception as e:
                app.logger.warning(f"English transcript unavailable: {str(e)}")
                # If English transcript not available, try Hindi
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
                    app.logger.info("Successfully retrieved Hindi transcript")
                except Exception as e2:
                    app.logger.error(f"Hindi transcript also unavailable: {str(e2)}")
                    return jsonify({"error": f"No transcripts available in supported languages: {str(e2)}"}), 404
            
            transcript_text = " ".join([t['text'] for t in transcript_list])
            app.logger.info(f"Transcript length: {len(transcript_text)} characters")
            
            if len(transcript_text) < 50:  # Arbitrary minimum length
                return jsonify({"error": "Transcript too short to summarize"}), 400
                
        except Exception as e:
            app.logger.error(f"Transcript fetch failed: {str(e)}")
            return jsonify({"error": f"Transcript fetch failed: {str(e)}"}), 500

        try:
            app.logger.info("Starting summarization pipeline")
            summarized_text = summarize_video_pipeline(transcript_text)
            app.logger.info("Summarization completed successfully")
            return jsonify({
                "summarizedText": summarized_text,
                "status": "success"
            })
        except Exception as e:
            app.logger.error(f"Summarization pipeline failed: {str(e)}", exc_info=True)
            return jsonify({
                "error": f"Summarization failed: {str(e)}",
                "status": "error"
            }), 500

    except Exception as e:
        app.logger.error(f"General request processing error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
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
        llm = ChatGroq(model="Gemma2-9b-It", temperature=0.9)
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

        llm = ChatGroq(model="Gemma2-9b-It", temperature=0.3)
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
