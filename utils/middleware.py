"""
Middleware for the Dastyar Assistant API.

This module provides middleware functionality for rate limiting, caching,
and other performance optimizations for the FastAPI application.

It also includes LangGraph middleware functions for state management.
"""
import time
from typing import Callable, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import logging
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from config import API_RATE_LIMIT

logger = logging.getLogger("dastyar-middleware")

# === LangGraph Middleware Functions ===

def add_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add metadata to messages for tracking purposes.
    
    Args:
        state: Current state
        
    Returns:
        State with added metadata
    """
    if "messages" not in state:
        return state
        
    updated_messages = []
    
    for msg in state["messages"]:
        # Skip messages that already have metadata
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("metadata"):
            updated_messages.append(msg)
            continue
            
        # Add metadata based on message type
        if isinstance(msg, HumanMessage):
            if not hasattr(msg, "additional_kwargs") or not msg.additional_kwargs.get("metadata"):
                # Create a copy with metadata
                updated_msg = HumanMessage(
                    content=msg.content,
                    additional_kwargs={
                        **(msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}),
                        "metadata": {"source": "user", "message_id": compute_message_hash(msg)}
                    }
                )
                updated_messages.append(updated_msg)
            else:
                updated_messages.append(msg)
        elif isinstance(msg, AIMessage):
            if not hasattr(msg, "additional_kwargs") or not msg.additional_kwargs.get("metadata"):
                # Create a copy with metadata
                updated_msg = AIMessage(
                    content=msg.content,
                    additional_kwargs={
                        **(msg.additional_kwargs if hasattr(msg, "additional_kwargs") else {}),
                        "metadata": {"source": "assistant", "message_id": compute_message_hash(msg)}
                    }
                )
                updated_messages.append(updated_msg)
            else:
                updated_messages.append(msg)
        else:
            # For other message types, pass through unchanged
            updated_messages.append(msg)
            
    return {**state, "messages": updated_messages}

def compute_message_hash(message: BaseMessage) -> str:
    """Compute a unique hash for a message based on its content."""
    content = getattr(message, "content", str(message))
    msg_type = message.__class__.__name__
    hash_input = f"{msg_type}:{content}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]

def track_message_fingerprints(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add fingerprints to messages for deduplication.
    This creates a stateful registry of messages to prevent duplication.
    
    Args:
        state: Current state
        
    Returns:
        State with message fingerprints
    """
    if "messages" not in state:
        return {**state, "message_fingerprints": {}}
    
    # Initialize or retrieve fingerprint registry
    fingerprints = state.get("message_fingerprints", {})
    unique_messages = []
    
    for msg in state["messages"]:
        # Extract content and create a fingerprint
        if isinstance(msg, dict):
            content = msg.get("content", "")
            msg_type = msg.get("type", "unknown")
        else:
            content = getattr(msg, "content", str(msg))
            msg_type = msg.__class__.__name__
        
        # Skip empty messages
        if not content.strip():
            continue
            
        # Create a fingerprint for this message
        msg_hash = compute_message_hash(msg) if isinstance(msg, BaseMessage) else hashlib.md5(f"{msg_type}:{content}".encode()).hexdigest()[:8]
        
        # Only include unique messages
        if msg_hash not in fingerprints:
            fingerprints[msg_hash] = True
            unique_messages.append(msg)
    
    # Update state with unique messages and fingerprint registry
    return {
        **state,
        "messages": unique_messages,
        "message_fingerprints": fingerprints
    }

def limit_conversation_history(state: Dict[str, Any], max_messages: int = 20) -> Dict[str, Any]:
    """
    Limit conversation history to prevent state from growing too large.
    
    Args:
        state: Current state
        max_messages: Maximum number of messages to keep
        
    Returns:
        State with limited conversation history
    """
    if "messages" not in state or len(state["messages"]) <= max_messages:
        return state
        
    # Apply fingerprinting before truncating
    state_with_fingerprints = track_message_fingerprints(state)
    
    # Keep the latest messages, prioritizing the most recent
    messages = state_with_fingerprints["messages"]
    truncated_messages = messages[-max_messages:]
    
    return {**state_with_fingerprints, "messages": truncated_messages}

# === API Middleware Classes ===

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting using token bucket algorithm.
    
    This middleware tracks request rates and limits them based on API keys
    or IP addresses to prevent abuse.
    """
    
    def __init__(self, app, rate_limit_per_minute: int = API_RATE_LIMIT):
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        # Dict to store rate limit data: {key: {"tokens": float, "last_refill": float}}
        self.limiters = {}
        self.refill_rate = rate_limit_per_minute / 60.0  # tokens per second
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for certain paths
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get the client identifier (API key or IP address)
        client_id = request.headers.get("X-API-Key", request.client.host)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Process the request
        return await call_next(request)
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if the client has exceeded their rate limit.
        
        Uses a token bucket algorithm where tokens refill at a constant rate.
        """
        now = time.time()
        
        # Initialize bucket for new clients
        if client_id not in self.limiters:
            self.limiters[client_id] = {
                "tokens": self.rate_limit,
                "last_refill": now
            }
            return True
        
        # Refill tokens based on time elapsed
        limiter = self.limiters[client_id]
        time_elapsed = now - limiter["last_refill"]
        new_tokens = time_elapsed * self.refill_rate
        limiter["tokens"] = min(self.rate_limit, limiter["tokens"] + new_tokens)
        limiter["last_refill"] = now
        
        # Check if any tokens are available
        if limiter["tokens"] < 1.0:
            return False
        
        # Consume a token
        limiter["tokens"] -= 1.0
        return True


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Response caching middleware for improved performance.
    
    Caches responses for GET requests to reduce latency and database load.
    """
    
    def __init__(self, app, redis_pool: Optional[redis.Redis] = None, ttl: int = 300):
        super().__init__(app)
        self.redis_pool = redis_pool
        self.ttl = ttl  # Cache TTL in seconds
        self.cacheable_paths = ["/tools", "/conversation/"]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests to certain paths
        if request.method != "GET" or not any(path in request.url.path for path in self.cacheable_paths):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache if Redis is available
        if self.redis_pool:
            try:
                cached_response = await self.redis_pool.get(cache_key)
                if cached_response:
                    response_data = json.loads(cached_response)
                    return Response(
                        content=response_data["content"],
                        status_code=response_data["status_code"],
                        headers=response_data["headers"],
                        media_type=response_data["media_type"]
                    )
            except Exception as e:
                logger.error(f"Cache error: {str(e)}")
        
        # If not in cache or error occurred, process the request
        response = await call_next(request)
        
        # Cache the response if successful
        if response.status_code == 200 and self.redis_pool:
            try:
                # Get response content
                response_body = [section async for section in response.body_iterator]
                response.body_iterator = iter(response_body)
                response_content = b"".join(response_body)
                
                # Store in cache
                response_data = {
                    "content": response_content.decode(),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "media_type": response.media_type
                }
                await self.redis_pool.setex(
                    cache_key,
                    self.ttl,
                    json.dumps(response_data)
                )
            except Exception as e:
                logger.error(f"Error caching response: {str(e)}")
                
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate a unique cache key based on the request."""
        key_parts = [
            request.method,
            str(request.url),
            request.headers.get("X-API-Key", "")
        ]
        key_string = ":".join(key_parts)
        return f"dastyar:cache:{hashlib.md5(key_string.encode()).hexdigest()}"


class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    """
    Middleware to manage database connection pooling.
    
    Ensures efficient reuse of database connections to minimize overhead.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.pools = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Attach connection pools to request state
        request.state.pools = self.pools
        
        # Process the request
        response = await call_next(request)
        
        return response

    @staticmethod
    async def get_pool(pool_name: str, creator_func: Callable, **kwargs) -> Any:
        """
        Get or create a connection pool.
        
        Args:
            pool_name: Name of the pool to get or create
            creator_func: Function to create the pool if it doesn't exist
            **kwargs: Arguments to pass to the creator function
        
        Returns:
            The connection pool
        """
        if pool_name not in ConnectionPoolMiddleware.pools:
            ConnectionPoolMiddleware.pools[pool_name] = await creator_func(**kwargs)
        return ConnectionPoolMiddleware.pools[pool_name]
    
    @staticmethod
    async def close_pools():
        """Close all connection pools."""
        for pool_name, pool in ConnectionPoolMiddleware.pools.items():
            if hasattr(pool, "close"):
                await pool.close()
        ConnectionPoolMiddleware.pools.clear()
