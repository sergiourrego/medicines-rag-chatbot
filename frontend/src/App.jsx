import { useState } from 'react'
import './App.css'
import Markdown from 'marked-react';

function App() {
  const [chatHistory, setChatHistory] = useState({"messages": []});
  const [chatCurrent, setChatCurrent] = useState('');

  const handleSendMessage = async() => {
    if (chatCurrent.trim()) { // Check for empty or whitespace-only input
      const updatedChatHistory = {"messages": [
        ...chatHistory.messages,
        { content: chatCurrent, role: 'user' }
      ]
    };
      setChatCurrent(''); // Clear input after sending message
      setChatHistory(updatedChatHistory);
      await sendChatToAI(updatedChatHistory);
    }
  };
  // dev environment: this request is proxied to backend - see vite.config
  const sendChatToAI = async (chathistory) => {
    try {
      const response = await fetch('/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chathistory),
      });
      console.log(response)
      if (!response.ok) {
        throw new Error(`Error sending message: ${response.statusText}`);
      } else {
        let newChat = await response.json()
        setChatHistory(newChat);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-200">
      <div className="chat-area overflow-auto p-4 flex-grow">
        {chatHistory.messages.map((message, index) => (
          <div
            key={index}
            className={`chat ${message.role === 'user' ? 'chat-end' : 'chat-start'}`}
          >
            <div className={`chat-bubble ${message.role === 'user' ? 'bg-slate-500' : 'bg-slate-800'}`}><Markdown>{message.content}</Markdown></div>
          </div>
        ))}
      </div>
      <div className="chat-input flex p-4">
        <textarea
          className="textarea textarea-bordered w-full mr-2"
          placeholder="Ask a question about medication..."
          value={chatCurrent}
          onChange={(e) => setChatCurrent(e.target.value)}
        />
        <button className="btn bg-blue-300 text-gray" onClick={handleSendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}


export default App
