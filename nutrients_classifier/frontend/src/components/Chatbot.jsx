import React, { useState, useRef, useEffect, useContext } from 'react';
import { MessageCircle, X, Send, Loader2 } from 'lucide-react';
import { AuthContext } from '../context/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { text: "Hi! I'm NutriVision AI. Based on your logged meals today, how can I help you?", isBot: true }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  const { user } = useContext(AuthContext);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = input.trim();
    setMessages(prev => [...prev, { text: userMsg, isBot: false }]);
    setInput("");
    
    if (!user) {
      setMessages(prev => [...prev, { text: "Please log in to chat with the AI Nutritionist!", isBot: true }]);
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: user.id, message: userMsg })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { text: data.reply, isBot: true }]);
    } catch (err) {
      setMessages(prev => [...prev, { text: "Network error. Make sure the FastAPI server is running.", isBot: true }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button 
        onClick={() => setIsOpen(true)}
        style={{
          position: 'fixed',
          bottom: '30px',
          right: '30px',
          width: '60px',
          height: '60px',
          borderRadius: '30px',
          background: 'var(--accent)',
          color: 'white',
          border: 'none',
          boxShadow: '0 10px 25px rgba(16, 185, 129, 0.4)',
          cursor: 'pointer',
          display: isOpen ? 'none' : 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          transition: 'transform 0.2s',
          ':hover': { transform: 'scale(1.1)' }
        }}
      >
        <MessageCircle size={28} />
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div style={{
          position: 'fixed',
          bottom: '30px',
          right: '30px',
          width: '350px',
          height: '500px',
          background: 'var(--surface)',
          borderRadius: '20px',
          boxShadow: 'var(--shadow)',
          border: '1px solid var(--border)',
          backdropFilter: 'blur(20px)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 1000,
          overflow: 'hidden'
        }}>
          {/* Header */}
          <div style={{ padding: '15px 20px', background: 'var(--bg)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="pulse-dot"></span>
              <span className="syne-font" style={{ fontWeight: 700 }}>AI Nutritionist</span>
            </div>
            <button onClick={() => setIsOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
              <X size={20} />
            </button>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {messages.map((m, idx) => (
              <div key={idx} style={{ 
                alignSelf: m.isBot ? 'flex-start' : 'flex-end', 
                maxWidth: '85%',
                padding: '12px 16px',
                borderRadius: m.isBot ? '16px 16px 16px 0' : '16px 16px 0 16px',
                background: m.isBot ? 'var(--bg)' : 'var(--accent)',
                color: m.isBot ? 'var(--text)' : 'white',
                fontSize: '0.95rem',
                border: m.isBot ? '1px solid var(--border)' : 'none'
              }}>
                {m.text}
              </div>
            ))}
            {loading && (
              <div style={{ alignSelf: 'flex-start', padding: '10px 16px', borderRadius: '16px 16px 16px 0', background: 'var(--bg)', border: '1px solid var(--border)' }}>
                <Loader2 size={16} className="spinner" style={{ animation: 'spin 1s linear infinite' }} />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '15px', borderTop: '1px solid var(--border)', display: 'flex', gap: '10px', background: 'var(--bg)' }}>
            <input 
              type="text" 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              placeholder="Ask about your diet..."
              style={{ flex: 1, padding: '10px 15px', borderRadius: '20px', border: '1px solid var(--border)', outline: 'none', fontSize: '0.9rem' }}
            />
            <button 
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ width: '40px', height: '40px', borderRadius: '20px', background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Send size={16} style={{ marginLeft: '-2px' }} />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;
