import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import { Video, Sparkles, Copy, ExternalLink, Play } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { analyzeVideo } from "@/utils/api";
import axios from "axios";
const VideoEditor = ({ videoUrl, setVideoUrl }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [summary, setSummary] = useState("");
  const { toast } = useToast();

  // Use useEffect to apply white color to the heading
  // useEffect(() => {
  //   const heading = document.querySelector(".video-analyzer-heading");
  //   if (heading) {
  //     heading.style.color = "#FFFFFF";
  //   }
  // }, []);

  const handleAnalyze = async () => {
    if (!videoUrl.trim()) {
      toast({
        title: "Empty URL",
        description: "Please enter a video URL to analyze",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);

    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast({
          title: "Login first",
          description: "Please ",
          variant: "destructive",
        });
      }
      const response = await axios.post(
        "http://localhost:5000/news/video",
        {
          inputType: "video",
          videoUrl: videoUrl,
          status: "completed",
        },
        {
          headers: {
            token: token,
          },
        }
      );

      const summaryText = response.data?.data?.summary;

      if (summaryText) {
        setSummary(summaryText);
        toast({
          title: "Analysis complete",
          description: "The video was successfully analyzed",
        });
      } else {
        throw new Error("No summary returned from server");
      }
    } catch (error) {
      console.error("Error analyzing video:", error);
      toast({
        title: "Analysis failed",
        description: error.response?.data?.message || "Failed to analyze video",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopy = () => {
    if (summary) {
      navigator.clipboard.writeText(summary);
      toast({
        title: "Copied!",
        description: "Summary copied to clipboard",
      });
    }
  };

  const isValidUrl = (url) => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-6">
        <div className="flex items-center space-x-2 mb-4">
          <Video className="h-5 w-5 text-news-primary" />
          <h2 className="text-xl font-semibold text-foreground dark:text-white">
            Video Analyzer
          </h2>
        </div>
        <div className="content-card">
          <div className="mb-4">
            <label
              htmlFor="videoUrl"
              className="block text-sm font-medium text-foreground mb-1"
            >
              Enter video URL
            </label>
            <div className="flex space-x-2">
              <Input
                id="videoUrl"
                type="url"
                placeholder="https://example.com/video"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                className="flex-1"
              />
              <Button
                onClick={handleAnalyze}
                className="bg-news-primary hover:bg-news-dark whitespace-nowrap flex items-center gap-2"
                disabled={isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    <span>Analyze</span>
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* {videoUrl && isValidUrl(videoUrl) && (
            <div className="mt-6 border rounded-lg overflow-hidden bg-gray-100">
              ...
            </div>
          )} */}
          {/* {videoUrl && isValidUrl(videoUrl) && (
            <div className="mt-6 border rounded-lg overflow-hidden bg-gray-100">
              <div className="aspect-video relative flex items-center justify-center bg-gray-900">
                <div className="text-white text-center">
                  <Play className="h-16 w-16 mx-auto mb-2 opacity-70" />
                  <p className="text-sm opacity-80">Video Player</p>
                  <p className="text-xs mt-1 opacity-50 max-w-sm truncate">{videoUrl}</p>
                </div>
                <div className="absolute bottom-4 right-4">
                  <a 
                    href={videoUrl} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="bg-white/20 backdrop-blur-sm text-white text-xs rounded-full px-3 py-1 inline-flex items-center gap-1 hover:bg-white/30 transition-colors"
                  >
                    <ExternalLink className="h-3 w-3" />
                    <span>Open Video</span>
                  </a>
                </div>
              </div>
            </div>
          )} */}
        </div>
      </div>

      {summary && (
        <div className="animate-fade-in">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold text-foreground">
                Generated Summary
              </h2>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopy}
              className="flex items-center gap-1"
            >
              <Copy className="h-4 w-4" />
              <span>Copy</span>
            </Button>
          </div>

          <div className="bg-muted p-6 rounded-xl shadow-md border border-border">
            <div
              className="prose text-foreground dark:prose-invert max-w-none"
              dangerouslySetInnerHTML={{ __html: summary }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoEditor;
