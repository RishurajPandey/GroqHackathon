import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import axios from 'axios';
import { useTheme } from "@/context/ThemeContext";

const History = () => {
  const [userhistory, setUserHistory] = useState([]);
  const { isDark } = useTheme();

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const history = await axios.get('http://localhost:5000/user/history', {
          headers: {
            token: localStorage.getItem('token'),
          },
        });
        console.log(history.data);
        setUserHistory(history.data);
      } catch (error) {
        console.error('Error fetching history:', error);
      }
    };

    fetchHistory();
  }, []);
  const processHtmlContent = (htmlContent) => {

    if (typeof window === 'undefined') return htmlContent;

    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');

    const allElements = doc.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, span, div, ul, ol');
    allElements.forEach(el => {
      el.style.color = '';
    });
    return doc.body.innerHTML;
  };

  return (
    <div className={`min-h-screen ${isDark ? 'dark-theme' : ''}`}>
      <Header />
      <div className="max-w-4xl mx-auto px-4 py-6">
        <h1 className="text-4xl font-bold mb-6 text-center text-black dark:text-white">User History</h1>

        {userhistory.length === 0 ? (
          <p className="text-center text-lg text-black dark:text-white">No history found.</p>
        ) : (
          userhistory
            .filter(uh => uh.videoUrl || uh.originalText)
            .map((uh, index) => (
              <div
                key={index}
                className={`rounded-2xl shadow p-6 mb-8 ${isDark ? 'bg-gray-800' : 'bg-white'}`}
              >
                <h2 className="text-2xl font-bold mb-4 text-black dark:text-white">
                  {uh.originalText
                    ? uh.originalText.substring(0, 100) + (uh.originalText.length > 100 ? '...' : '')
                    : uh.videoUrl
                    ? uh.videoUrl
                    : 'Failure'}
                </h2>

                {/* Apply a wrapper with text styles that will override child elements */}
                <div className={`custom-html-content text-lg ${isDark ? 'dark-content' : 'light-content'}`}>
                  <div
                    dangerouslySetInnerHTML={{ 
                      __html: processHtmlContent(uh.summarizedText) 
                    }}
                  />
                </div>
              </div>
            ))
        )}
      </div>
      
      {/* Add a style tag to enforce text colors and sizes on all elements */}
      <style jsx global>{`
        .dark-theme .dark-content * {
          color: white !important;
          font-size: 1.125rem !important; /* text-lg equivalent */
        }
        .light-content * {
          color: black !important;
          font-size: 1.125rem !important; /* text-lg equivalent */
        }
        .dark-content p, .light-content p {
          margin-bottom: 1rem !important;
          line-height: 1.7 !important;
        }
        .dark-content h1, .dark-content h2, .dark-content h3, 
        .dark-content h4, .dark-content h5, .dark-content h6 {
          color: white !important;
          font-weight: bold !important;
          margin-top: 1.5rem !important;
          margin-bottom: 1rem !important;
        }
        .light-content h1, .light-content h2, .light-content h3, 
        .light-content h4, .light-content h5, .light-content h6 {
          color: black !important;
          font-weight: bold !important;
          margin-top: 1.5rem !important;
          margin-bottom: 1rem !important;
        }
        .dark-content h1, .light-content h1 {
          font-size: 1.875rem !important; /* text-3xl */
        }
        .dark-content h2, .light-content h2 {
          font-size: 1.5rem !important; /* text-2xl */
        }
        .dark-content h3, .light-content h3 {
          font-size: 1.25rem !important; /* text-xl */
        }
        .dark-theme .dark-content a {
          color: #60a5fa !important;
          font-weight: bold !important;
          text-decoration: underline !important;
        }
        .light-content a {
          color: #2563eb !important;
          font-weight: bold !important;
          text-decoration: underline !important;
        }
        .dark-content ul, .light-content ul,
        .dark-content ol, .light-content ol {
          padding-left: 2rem !important;
          margin-bottom: 1rem !important;
        }
        .dark-content li, .light-content li {
          margin-bottom: 0.5rem !important;
        }
      `}</style>
    </div>
  );
};

export default History;