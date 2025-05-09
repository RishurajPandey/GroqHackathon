
import { FileText, Mic, Video, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import ThemeToggle from "./ThemeToggle";

interface HeaderProps {
  selectedMedia: 'text' | 'video' | 'audio';
  setSelectedMedia: (media: 'text' | 'video' | 'audio') => void;
}

const Header = ({ selectedMedia, setSelectedMedia }: HeaderProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  return (
    <header className="relative z-10 bg-card border-b border-border shadow-sm">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 hover-scale">
            <div className="bg-news-primary p-2 rounded-lg green-gradient">
              <FileText className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-xl font-bold text-foreground hidden sm:block">
            BriefLens
            </h1>
            <h1 className="text-xl font-bold text-foreground sm:hidden">
            BriefLens
            </h1>
          </div>
          
          <div className="hidden md:flex items-center space-x-1 border border-border rounded-xl px-3 py-2  backdrop-blur-sm shadow-sm">
            <Button
              variant={selectedMedia === 'text' ? 'default' : 'outline'}
              className={`flex items-center gap-2 hover-lift ${selectedMedia === 'text' ? 'green-gradient hover:shadow-md hover:shadow-primary/30' : ''}`}
              onClick={() => setSelectedMedia('text')}
            >
              <FileText className="h-4 w-4" />
              <span>Text</span>
            </Button>
            <Button
              variant={selectedMedia === 'video' ? 'default' : 'outline'}
              className={`flex items-center gap-2 hover-lift ${selectedMedia === 'video' ? 'green-gradient hover:shadow-md hover:shadow-primary/30' : ''}`}
              onClick={() => setSelectedMedia('video')}
            >
              <Video className="h-4 w-4" />
              <span>Video</span>
            </Button>
            <Button
              variant={selectedMedia === 'audio' ? 'default' : 'outline'}
              className={`flex items-center gap-2 hover-lift ${selectedMedia === 'audio' ? 'green-gradient hover:shadow-md hover:shadow-primary/30' : ''}`}
              onClick={() => setSelectedMedia('audio')}
            >
              <Mic className="h-4 w-4" />
              <span>Audio</span>
            </Button>
            <ThemeToggle />
          </div>
          
          <div className="md:hidden">
            <Button variant="ghost" size="icon" onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="hover-scale">
              <Menu className="h-6 w-6" />
            </Button>
          </div>
        </div>
        
        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden mt-4 bg-card rounded-md shadow-lg border border-border animate-fade-in absolute right-4 left-4 z-50">
            <div className="py-2 space-y-1">
              <button 
                className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 transition-colors duration-200 ${selectedMedia === 'text' ? 'bg-muted text-primary font-medium' : 'text-foreground hover:bg-muted/50'}`}
                onClick={() => {
                  setSelectedMedia('text');
                  setMobileMenuOpen(false);
                }}
              >
                <FileText className="h-4 w-4" />
                <span>Text</span>
              </button>
              <button 
                className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 transition-colors duration-200 ${selectedMedia === 'video' ? 'bg-muted text-primary font-medium' : 'text-foreground hover:bg-muted/50'}`}
                onClick={() => {
                  setSelectedMedia('video');
                  setMobileMenuOpen(false);
                }}
              >
                <Video className="h-4 w-4" />
                <span>Video</span>
              </button>
              <button 
                className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 transition-colors duration-200 ${selectedMedia === 'audio' ? 'bg-muted text-primary font-medium' : 'text-foreground hover:bg-muted/50'}`}
                onClick={() => {
                  setSelectedMedia('audio');
                  setMobileMenuOpen(false);
                }}
              >
                <Mic className="h-4 w-4" />
                <span>Audio</span>
              </button>
              <div className="px-4 py-2 flex justify-center">
                <ThemeToggle />
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;