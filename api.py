"""
FastAPI implementation for the Dastyar assistant API.

This module exposes the LangGraph-based assistant as an API service with endpoints for
chat interactions, streaming responses, conversation history, and tool management.
"""
import os
import uuid
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging
from functools import lru_cache

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field
import redis.asyncio as redis

# Import your LangGraph assistant
from main import run_info_assistant
from graph.state import InfoAssistantState
from config import ALLOWED_ORIGINS, API_RATE_LIMIT, MAX_CONVERSATION_HISTORY
from utils.middleware import RateLimitMiddleware, CacheMiddleware, ConnectionPoolMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dastyar-api")

# Create FastAPI instance
app = FastAPI(
    title="Dastyar Assistant API",
    description="API for interacting with the Dastyar LangGraph-based assistant",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, rate_limit_per_minute=API_RATE_LIMIT)

# Add connection pooling middleware
app.add_middleware(ConnectionPoolMiddleware)

# Redis pool for shared access
redis_pool = None

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global redis_pool
    
    # For Render deployment, get Redis URL from environment variable
    redis_url = os.getenv("REDIS_URL") or os.getenv("RENDER_REDIS_URL")
    
    if not redis_url:
        # Fallback to localhost for development
        redis_url = "redis://localhost:6379/0"
        logger.info(f"No Redis URL found in environment, using default: {redis_url}")
    else:
        logger.info("Redis URL found in environment variables")
    
    try:
        # Configure Redis connection with appropriate pool settings for cloud environment
        redis_pool = await redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=10,
            socket_timeout=5.0,
            socket_keepalive=True,
            retry_on_timeout=True,
            health_check_interval=30
        )
        await redis_pool.ping()
        logger.info("Redis connection pool initialized successfully")
        
        # Add cache middleware with the Redis pool
        app.add_middleware(CacheMiddleware, redis_pool=redis_pool, ttl=300)
        
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Running without persistent storage.")
        redis_pool = None

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global redis_pool
    if redis_pool:
        await redis_pool.close()
        logger.info("Redis connection pool closed")
    
    # Close all connection pools
    await ConnectionPoolMiddleware.close_pools()

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Read API keys from environment variable or from a config file
@lru_cache()
def get_api_keys() -> List[str]:
    """Get list of valid API keys."""
    api_keys_str = os.getenv("API_KEYS", "")
    if api_keys_str:
        return [k.strip() for k in api_keys_str.split(",")]
    return []

async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify the provided API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing",
        )
    
    valid_keys = get_api_keys()
    if not valid_keys or api_key in valid_keys:
        return api_key
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Request and Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user message")
    conversation_id: Optional[str] = Field(None, description="ID of existing conversation")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional request metadata")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    created_at: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field({}, description="Additional response metadata")

class ToolConfig(BaseModel):
    enabled: bool = Field(..., description="Whether the tool is enabled")
    description: Optional[str] = Field(None, description="Tool description")

class ToggleToolRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to toggle")
    enabled: bool = Field(..., description="Whether to enable or disable the tool")

class ConversationHistory(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[Dict[str, Any]] = Field(..., description="Messages in the conversation")
    created_at: str = Field(..., description="Conversation creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field({}, description="Conversation metadata")

class ToolsList(BaseModel):
    tools: Dict[str, ToolConfig] = Field(..., description="Available tools and their configurations")

# Store for tool configurations (in-memory fallback)
tool_configs = {
    "podcast_tools": {"enabled": True, "description": "Podcast recommendation tools"},
    "news_tools": {"enabled": True, "description": "News search and summarization tools"}
}

# Helper functions
def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())

async def save_conversation(conversation_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
    """Save conversation state to Redis or in-memory store."""
    global redis_pool
    
    # Create a serializable copy of the state
    serializable_state = {}
    
    # Handle special object types that need custom serialization
    for key, value in state.items():
        if key == "messages":
            # Convert message objects to serializable dictionaries
            serializable_messages = []
            for msg in value:
                if hasattr(msg, "to_dict"):
                    # Use built-in LangChain serialization if available
                    msg_dict = msg.to_dict()
                    serializable_messages.append(msg_dict)
                elif hasattr(msg, "content") and hasattr(msg, "type"):
                    # Handle messages with content and type attributes
                    msg_dict = {
                        "type": getattr(msg, "type", "unknown"),
                        "content": getattr(msg, "content", ""),
                        "additional_kwargs": getattr(msg, "additional_kwargs", {})
                    }
                    serializable_messages.append(msg_dict)
                else:
                    # Fallback for other objects
                    msg_dict = {
                        "type": msg.__class__.__name__,
                        "content": str(msg)
                    }
                    serializable_messages.append(msg_dict)
            serializable_state["messages"] = serializable_messages
        else:
            # Handle other potentially non-serializable objects
            try:
                # Test if the value is serializable
                json.dumps(value)
                serializable_state[key] = value
            except (TypeError, OverflowError):
                # If not serializable, convert to string or simple representation
                if hasattr(value, "to_dict"):
                    serializable_state[key] = value.to_dict()
                elif isinstance(value, list):
                    serializable_state[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None), dict, list)) else item for item in value]
                elif isinstance(value, dict):
                    serializable_state[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None), dict, list)) else v for k, v in value.items()}
                else:
                    serializable_state[key] = str(value)
    
    # Prepare conversation data
    conversation_data = {
        "state": json.dumps(serializable_state),
        "updated_at": datetime.now().isoformat(),
        "metadata": json.dumps(metadata or {})
    }
    
    # Extract messages for easier retrieval
    messages = []
    for msg in state.get("messages", []):
        msg_dict = None
        
        if hasattr(msg, "to_dict"):
            msg_dict = msg.to_dict()
        elif hasattr(msg, "type") and hasattr(msg, "content"):
            msg_dict = {
                "type": getattr(msg, "type", "unknown"),
                "content": getattr(msg, "content", "")
            }
        else:
            # Handle different message formats
            msg_dict = {
                "type": "human" if "human" in str(type(msg)).lower() else "ai",
                "content": getattr(msg, "content", str(msg))
            }
        
        messages.append({
            "role": "user" if msg_dict.get("type", "").lower() == "human" else "assistant",
            "content": msg_dict.get("content", ""),
            "timestamp": datetime.now().isoformat()
        })
    
    conversation_data["messages"] = json.dumps(messages)
    
    # Save to Redis if available
    if redis_pool:
        try:
            # Check if this is a new conversation
            is_new = not await redis_pool.exists(f"conversation:{conversation_id}")
            
            if is_new:
                conversation_data["created_at"] = datetime.now().isoformat()
                
            # Use a pipeline for atomic operations
            async with redis_pool.pipeline(transaction=True) as pipe:
                # Save conversation data
                await pipe.hset(f"conversation:{conversation_id}", mapping=conversation_data)
                
                # Add to index
                await pipe.zadd(
                    "conversations", 
                    {conversation_id: time.time()}
                )
                
                # Limit total number of conversations if needed
                if is_new:
                    await pipe.zremrangebyrank(
                        "conversations", 
                        0, 
                        -MAX_CONVERSATION_HISTORY-1
                    )
                
                await pipe.execute()
                
            logger.debug(f"Saved conversation {conversation_id} to Redis")
            return
            
        except Exception as e:
            logger.error(f"Error saving to Redis: {e}, falling back to in-memory storage")
    
    # Fallback to in-memory store if Redis is not available
    if not hasattr(save_conversation, "conversation_store"):
        save_conversation.conversation_store = {}
        
    if conversation_id not in save_conversation.conversation_store:
        save_conversation.conversation_store[conversation_id] = {
            "created_at": datetime.now().isoformat(),
        }
        
    # Update the conversation
    save_conversation.conversation_store[conversation_id].update({
        "state": serializable_state,  # Use the serializable version here
        "messages": messages,
        "updated_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    })
    
    logger.debug(f"Saved conversation {conversation_id} to in-memory store")

async def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve conversation state from Redis or in-memory store."""
    global redis_pool
    
    # Try Redis first
    if redis_pool:
        try:
            conversation_data = await redis_pool.hgetall(f"conversation:{conversation_id}")
            if conversation_data:
                # Parse JSON fields
                if "state" in conversation_data:
                    conversation_data["state"] = json.loads(conversation_data["state"])
                if "messages" in conversation_data:
                    conversation_data["messages"] = json.loads(conversation_data["messages"])
                if "metadata" in conversation_data:
                    conversation_data["metadata"] = json.loads(conversation_data["metadata"])
                    
                return conversation_data
        except Exception as e:
            logger.error(f"Error retrieving from Redis: {e}, falling back to in-memory storage")
    
    # Fallback to in-memory store
    if hasattr(save_conversation, "conversation_store"):
        return save_conversation.conversation_store.get(conversation_id)
    
    return None

async def get_tool_configs() -> Dict[str, Dict[str, Any]]:
    """Get tool configurations from Redis or in-memory store."""
    global redis_pool
    
    # Try Redis first
    if redis_pool:
        try:
            tool_data = await redis_pool.get("tool_configs")
            if tool_data:
                return json.loads(tool_data)
        except Exception as e:
            logger.error(f"Error retrieving tool configs from Redis: {e}")
    
    # Return in-memory configs
    return tool_configs

async def save_tool_configs(configs: Dict[str, Dict[str, Any]]) -> None:
    """Save tool configurations to Redis and in-memory store."""
    global redis_pool, tool_configs
    
    # Update in-memory store
    tool_configs = configs
    
    # Save to Redis if available
    if redis_pool:
        try:
            await redis_pool.set("tool_configs", json.dumps(configs))
        except Exception as e:
            logger.error(f"Error saving tool configs to Redis: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Dastyar Assistant API",
        "version": "1.0.0",
        "status": "online",
    }

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process a chat message and return a response."""
    conversation_id = request.conversation_id or generate_conversation_id()
    user_message = request.message
    
    # Get existing state if continuing a conversation
    conversation_data = await get_conversation(conversation_id)
    state = conversation_data.get("state") if conversation_data else None
    
    try:
        # Process the message with the LangGraph assistant
        start_time = time.time()
        response, updated_state = run_info_assistant(user_message, state, conversation_id)
        processing_time = time.time() - start_time
        
        # Add latency information to response metadata
        metadata = {
            "latency_ms": int(processing_time * 1000),
            "user_id": request.user_id,
            **request.metadata
        }
        
        # Save conversation state in the background
        background_tasks.add_task(
            save_conversation, 
            conversation_id, 
            updated_state, 
            metadata
        )
        
        # Log the interaction
        logger.info(f"Processed chat request: id={conversation_id}, latency={processing_time:.2f}s")
        
        # Return the response
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            created_at=datetime.now().isoformat(),
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )

async def generate_streaming_response(request: ChatRequest):
    """Generate streaming response for the chat request."""
    conversation_id = request.conversation_id or generate_conversation_id()
    user_message = request.message
    
    # Get existing state if continuing a conversation
    conversation_data = await get_conversation(conversation_id)
    state = conversation_data.get("state") if conversation_data else None
    
    try:
        # Begin streaming the response
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id})}\n\n"
        
        # Process the message with the LangGraph assistant
        start_time = time.time()
        response, updated_state = run_info_assistant(user_message, state, conversation_id)
        processing_time = time.time() - start_time
        
        # Save conversation state
        metadata = {
            "latency_ms": int(processing_time * 1000),
            "user_id": request.user_id,
            "streaming": True,
            **request.metadata
        }
        await save_conversation(conversation_id, updated_state, metadata)
        
        # Simulate streaming by chunking the response
        words = response.split()
        chunks = [' '.join(words[i:i+3]) for i in range(0, len(words), 3)]
        
        for i, chunk in enumerate(chunks):
            data = {
                "type": "chunk",
                "chunk": chunk + (" " if i < len(chunks) - 1 else ""),
                "done": False
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'end', 'done': True, 'latency_ms': int(processing_time * 1000)})}\n\n"
        
        # Log the interaction
        logger.info(f"Processed streaming chat request: id={conversation_id}, latency={processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        error_data = {
            "type": "error",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def stream_chat(request: Request):
    """Stream chat responses for real-time UI updates."""
    data = await request.json()
    chat_request = ChatRequest(**data)
    
    return StreamingResponse(
        generate_streaming_response(chat_request),
        media_type="text/event-stream"
    )

@app.get("/conversation/{conversation_id}", response_model=ConversationHistory, dependencies=[Depends(verify_api_key)])
async def get_conversation_history(conversation_id: str):
    """Get conversation history by ID."""
    conversation_data = await get_conversation(conversation_id)
    if not conversation_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=conversation_data.get("messages", []),
        created_at=conversation_data.get("created_at", datetime.now().isoformat()),
        updated_at=conversation_data.get("updated_at", datetime.now().isoformat()),
        metadata=conversation_data.get("metadata", {})
    )

@app.get("/tools", response_model=ToolsList, dependencies=[Depends(verify_api_key)])
async def list_tools():
    """List available tools and their configurations."""
    configs = await get_tool_configs()
    return ToolsList(
        tools={
            name: ToolConfig(**config) 
            for name, config in configs.items()
        }
    )

@app.post("/tools/toggle", dependencies=[Depends(verify_api_key)])
async def toggle_tool(request: ToggleToolRequest):
    """Enable or disable specific tools."""
    configs = await get_tool_configs()
    
    if request.tool_name not in configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request.tool_name}' not found"
        )
    
    # Update the tool configuration
    configs[request.tool_name]["enabled"] = request.enabled
    
    # Save the updated configurations
    await save_tool_configs(configs)
    
    return {"status": "success", "tool": request.tool_name, "enabled": request.enabled}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    global redis_pool
    
    # Check Redis connection
    redis_status = "connected"
    if redis_pool:
        try:
            await redis_pool.ping()
        except Exception as e:
            redis_status = f"error: {str(e)}"
    else:
        redis_status = "not configured"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "redis": redis_status
        }
    }

# Run the app if executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)