/**
 * Dastyar API Integration Example
 * 
 * This file demonstrates how to integrate the Dastyar API into a Node.js/Express application.
 */

const express = require('express');
const axios = require('axios');
const app = express();
const path = require('path');

// For parsing application/json
app.use(express.json());
// For parsing application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Configuration
const DASTYAR_API_URL = process.env.DASTYAR_API_URL || 'https://your-dastyar-api.onrender.com';
const DASTYAR_API_KEY = process.env.DASTYAR_API_KEY || 'your-api-key';

// In-memory conversation mapping (use a database in production)
const conversationMapping = new Map();

// Cleanup old conversations periodically
setInterval(() => {
  const now = Date.now();
  for (const [sessionId, data] of conversationMapping.entries()) {
    // Remove conversations older than 24 hours
    if (now - data.lastActivity > 24 * 60 * 60 * 1000) {
      conversationMapping.delete(sessionId);
    }
  }
}, 60 * 60 * 1000); // Check every hour

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString()
  });
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, sessionId } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    // Generate a session ID if not provided
    const currentSessionId = sessionId || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Get the Dastyar conversation ID from our mapping
    const sessionData = conversationMapping.get(currentSessionId) || {};
    const dastyarConversationId = sessionData.dastyarConversationId;
    
    // Prepare the request to Dastyar API
    const dastyarRequest = {
      message,
      conversation_id: dastyarConversationId,
      user_id: currentSessionId,
      metadata: {
        source: 'web_client',
        client_timestamp: new Date().toISOString(),
        ...req.body.metadata
      }
    };
    
    // Call the Dastyar API
    const dastyarResponse = await axios.post(
      `${DASTYAR_API_URL}/chat`,
      dastyarRequest,
      {
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': DASTYAR_API_KEY
        }
      }
    );
    
    // Update our conversation mapping
    conversationMapping.set(currentSessionId, {
      dastyarConversationId: dastyarResponse.data.conversation_id,
      lastActivity: Date.now()
    });
    
    // Return the response
    res.json({
      message: dastyarResponse.data.response,
      sessionId: currentSessionId,
      timestamp: dastyarResponse.data.created_at,
      metadata: dastyarResponse.data.metadata
    });
    
  } catch (error) {
    console.error('Error communicating with Dastyar API:', error);
    res.status(500).json({ 
      error: 'Failed to communicate with assistant',
      details: error.response ? error.response.data : error.message
    });
  }
});

// Stream chat endpoint
app.post('/api/chat/stream', (req, res) => {
  const { message, sessionId } = req.body;
  
  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }
  
  // Generate a session ID if not provided
  const currentSessionId = sessionId || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  // Get the Dastyar conversation ID from our mapping
  const sessionData = conversationMapping.get(currentSessionId) || {};
  const dastyarConversationId = sessionData.dastyarConversationId;
  
  // Set up SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  // Prepare the request to Dastyar API
  const dastyarRequest = {
    message,
    conversation_id: dastyarConversationId,
    user_id: currentSessionId
  };
  
  // Use axios to make a request to the Dastyar streaming endpoint
  axios({
    method: 'post',
    url: `${DASTYAR_API_URL}/chat/stream`,
    data: dastyarRequest,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': DASTYAR_API_KEY
    },
    responseType: 'stream'
  }).then(response => {
    // Pipe the Dastyar API response stream directly to our response
    response.data.on('data', (chunk) => {
      const chunkStr = chunk.toString();
      
      // Parse the data to extract conversation_id if available
      if (chunkStr.includes('"type":"start"')) {
        try {
          const dataMatch = chunkStr.match(/data: (.*?)(?:\n|$)/);
          if (dataMatch) {
            const data = JSON.parse(dataMatch[1]);
            if (data.conversation_id) {
              // Update our mapping with the conversation ID
              conversationMapping.set(currentSessionId, {
                dastyarConversationId: data.conversation_id,
                lastActivity: Date.now()
              });
            }
          }
        } catch (err) {
          console.error('Error parsing stream start data:', err);
        }
      }
      
      // Forward the chunk to the client
      res.write(chunk);
    });
    
    response.data.on('end', () => {
      res.end();
    });
    
  }).catch(error => {
    console.error('Stream error:', error);
    res.write(`data: ${JSON.stringify({type: 'error', error: error.message})}\n\n`);
    res.end();
  });
});

// Get conversation history
app.get('/api/conversation/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const sessionData = conversationMapping.get(sessionId);
    
    if (!sessionData || !sessionData.dastyarConversationId) {
      return res.status(404).json({ error: 'Conversation not found' });
    }
    
    // Call the Dastyar API to get conversation history
    const dastyarResponse = await axios.get(
      `${DASTYAR_API_URL}/conversation/${sessionData.dastyarConversationId}`,
      {
        headers: {
          'X-API-Key': DASTYAR_API_KEY
        }
      }
    );
    
    // Return the response
    res.json({
      sessionId: sessionId,
      messages: dastyarResponse.data.messages,
      created_at: dastyarResponse.data.created_at,
      updated_at: dastyarResponse.data.updated_at
    });
    
  } catch (error) {
    console.error('Error retrieving conversation:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve conversation history',
      details: error.response ? error.response.data : error.message
    });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Serve a simple HTML client for testing
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});