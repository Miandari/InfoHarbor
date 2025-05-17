# Client Integration Guide for Dastyar API

This guide provides instructions for integrating your Node.js/React application with the Dastyar Assistant API hosted on Render.

## Connection Configuration

When connecting to the Dastyar API on Render, optimize for speed with these configurations:

### Basic Setup

```javascript
// config.js
export const API_BASE_URL = "https://dastyar-api.onrender.com"; // Replace with your actual Render URL
export const API_KEY = "your-api-key";

// Default fetch options with proper headers
export const defaultOptions = {
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  },
};
```

## Optimizing for Low Latency

### 1. Client-Side Caching with SWR

SWR is a React Hooks library for data fetching that includes built-in cache and revalidation.

```bash
npm install swr
```

```javascript
// hooks/useConversation.js
import useSWR from 'swr';
import { API_BASE_URL, defaultOptions } from '../config';

const fetcher = (url) => fetch(url, defaultOptions).then(res => res.json());

export function useConversation(conversationId) {
  const { data, error, mutate } = useSWR(
    conversationId ? `${API_BASE_URL}/conversation/${conversationId}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 10000, // Don't refetch within 10 seconds
    }
  );

  return {
    conversation: data,
    isLoading: !error && !data,
    isError: error,
    refreshConversation: mutate
  };
}
```

### 2. Connection Pooling with Persistent Connections

For Node.js backend that communicates with the Dastyar API:

```javascript
// api/dastyarClient.js
const fetch = require('node-fetch');
const https = require('https');
const { API_BASE_URL, API_KEY } = require('../config');

// Create a persistent HTTP agent
const agent = new https.Agent({
  keepAlive: true,
  maxSockets: 25, // Adjust based on expected load
  timeout: 60000, // Longer timeout for LLM responses
});

// Create client with connection pooling
const dastyarClient = {
  async sendChat(message, conversationId = null, userId = null) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        user_id: userId,
      }),
      agent, // Use the persistent agent
    });
    
    return response.json();
  },
  
  // Stream chat for real-time responses
  streamChat(message, conversationId = null, onChunk, onComplete, onError) {
    const eventSource = new EventSource(
      `${API_BASE_URL}/chat/stream?api_key=${API_KEY}`, 
      {
        withCredentials: true,
        headers: { 'X-API-Key': API_KEY }
      }
    );
    
    fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
      }),
      agent, // Use the persistent agent
    }).then(response => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      function processStream() {
        reader.read().then(({ done, value }) => {
          if (done) {
            onComplete && onComplete();
            return;
          }
          
          const text = decoder.decode(value);
          const lines = text.split('\n\n');
          
          lines.forEach(line => {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'chunk') {
                onChunk && onChunk(data.chunk);
              } else if (data.type === 'error') {
                onError && onError(data.error);
              }
            }
          });
          
          processStream();
        }).catch(err => {
          onError && onError(err);
        });
      }
      
      processStream();
    });
  },
  
  // Other API methods
  async getTools() {
    const response = await fetch(`${API_BASE_URL}/tools`, {
      headers: {
        'X-API-Key': API_KEY,
      },
      agent,
    });
    
    return response.json();
  },
  
  async toggleTool(toolName, enabled) {
    const response = await fetch(`${API_BASE_URL}/tools/toggle`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
      body: JSON.stringify({
        tool_name: toolName,
        enabled,
      }),
      agent,
    });
    
    return response.json();
  },
};

module.exports = dastyarClient;
```

### 3. React Component Example with Streaming

```jsx
// components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import { dastyarClient } from '../api/dastyarClient';

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [streamingResponse, setStreamingResponse] = useState('');
  
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    // Add user message
    const userMessage = input;
    setMessages([...messages, { role: 'user', content: userMessage }]);
    setInput('');
    setIsLoading(true);
    
    try {
      // Use streaming for faster perceived response time
      setStreamingResponse('');
      let fullResponse = '';
      
      dastyarClient.streamChat(
        userMessage,
        conversationId,
        (chunk) => {
          fullResponse += chunk;
          setStreamingResponse(fullResponse);
        },
        () => {
          // On complete
          setMessages(prev => [...prev, { role: 'assistant', content: fullResponse }]);
          setStreamingResponse('');
          setIsLoading(false);
        },
        (error) => {
          console.error('Stream error:', error);
          setMessages(prev => [...prev, { 
            role: 'assistant', 
            content: 'Sorry, there was an error processing your request.' 
          }]);
          setStreamingResponse('');
          setIsLoading(false);
        }
      );
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request.' 
      }]);
      setIsLoading(false);
    }
  };
  
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.content}
          </div>
        ))}
        
        {streamingResponse && (
          <div className="message assistant streaming">
            {streamingResponse}
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
}
```

## Performance Optimization Tips

1. **Implement Browser Caching**: Add appropriate cache headers to your React application to cache static assets.

2. **Connection Persistence**: Keep connections alive with `keepAlive` settings to avoid the overhead of establishing new connections.

3. **Preconnect to API**: Use resource hints to establish early connections:
   ```html
   <link rel="preconnect" href="https://dastyar-api.onrender.com">
   ```

4. **Request Compression**: The API supports compression. Ensure your requests include:
   ```javascript
   headers: {
     'Accept-Encoding': 'gzip, deflate, br'
   }
   ```

5. **Batching Requests**: Where possible, batch multiple requests together to reduce round trips.

6. **Progressive Loading**: Implement skeleton screens while content loads to improve perceived performance.

## Example .env File for Node.js Backend

```
# API Configuration
DASTYAR_API_URL=https://dastyar-api.onrender.com
DASTYAR_API_KEY=your-api-key-here

# Connection Pool Settings
MAX_CONNECTIONS=25
CONNECTION_TIMEOUT=60000
KEEP_ALIVE_TIMEOUT=30000
```