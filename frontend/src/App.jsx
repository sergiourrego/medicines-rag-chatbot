import { useState, useRef, useEffect } from 'react'
import './App.css'
import Markdown from 'marked-react';
import nhslogo from './assets/nhs_attribution.png'

function App() {
  const [chatHistory, setChatHistory] = useState({"messages": []});
  const [chatCurrent, setChatCurrent] = useState('');
  const [urlHistory, setURLHistory] = useState([]);

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
        body: JSON.stringify({input: chathistory, urls: urlHistory}),
      });
      console.log(response)
      if (!response.ok) {
        throw new Error(`Error sending message: ${response.statusText}`);
      } else {
        let result = await response.json();
        let newChat = result.input;
        let urlHistory = result.urls.filter(url => url != null); //remove null objects when no url returned
        setURLHistory(urlHistory);
        setChatHistory(newChat);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };
  // Parse url into title
  const parseTitle = (url) => {
    let title = url.match(/medicines\/(.+)\/#/) 
    let formatTitle = title[1]
    .replace(/-/g, " ")
    .replace(/\//g," | ")
    .split(" ")
    .map(word => {
      if (!/^and$|^to$|^for$/.test(word)){
        return word[0].toUpperCase() + word.substring(1)
    } else {
        return word
    }})
    .join(" ")
    return formatTitle
  }

  // Enable scroll to bottom of chat when new message
  const chatRef = useRef(null)
  useEffect(() => {chatRef.current?.lastElementChild?.scrollIntoView()}, [chatHistory])
  // also for url display
  const urlsRef = useRef(null)
  useEffect(() => {urlsRef.current?.lastElementChild?.scrollIntoView()}, [urlHistory])

  return (
    <div className="flex flex-row h-screen bg-slate-200">
      <div className='flex-col w-2/3'>
        <div className="chat-area overflow-y-auto snap-y p-4 h-5/6 content-end" ref={chatRef}>
          {chatHistory.messages.map((message, index) => (
            <div
              key={index}
              className={`snap-end chat ${message.role === 'user' ? 'chat-end' : 'chat-start'}`}
            >
              <div className={`chat-bubble ${message.role === 'user' ? 'bg-slate-500' : 'bg-slate-800'}`}><Markdown>{message.content}</Markdown></div>
            </div>
          ))}
        </div>
        <div className="chat-input flex-col p-2 h-1/6 content-end">
          <textarea
            className="textarea textarea-bordered w-full h-2/3 mb-2"
            placeholder="Ask a question about medication..."
            value={chatCurrent}
            onChange={(e) => setChatCurrent(e.target.value)}
          />
          <button className="btn bg-blue-300 text-gray w-full mb-2" onClick={handleSendMessage}>
            Send
          </button>
        </div>
      </div>
      <div className="flex-col content-end w-1/3 bg-slate-400 border-x-2 border-slate-500">
        <div className="info-urls h-5/6 content-end overflow-y-auto snap-y" ref={urlsRef}>
          {urlHistory.map((urls, index) => (
            <ul key={index} className="menu menu-dropdown bg-blue-100 rounded-lg m-1">
              {urls.map((url, index)=>(
                <li key={index} className= {`bg-blue-300 rounded-md border border-blue-800 ${index != 0 && 'mt-2'}`}>
                  <a href={url} target='_blank' className="text-blue-900 font-bold hover:text-blue-600">{parseTitle(url)}</a>
                </li>
              ))}
            </ul>
          ))}
        </div>
        <div  className='h-1/6 content-center bg-white border-2 rounded-md border-slate-500 hover:bg-blue-100'>
          <a href="https://www.nhs.uk/medicines/" target="_blank"><img src={nhslogo} alt="Content supplied by the NHS website: nhs.uk" className='p-4'></img></a>
        </div>
      </div>
    </div>
  );
}


export default App
