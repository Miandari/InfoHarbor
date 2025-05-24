"""
API routes for the Elderly Assistant Agent
"""
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.models import (
    ChatRequest, ChatResponse, HealthStatus,
    MemoryUpdateRequest, UserProfileResponse
)
from src.agent.graph import ElderlyAssistantAgent
from src.memory.manager import MemoryManager
from src.settings import settings

# Initialize components
agent = ElderlyAssistantAgent()
memory_manager = MemoryManager()
logger = logging.getLogger("api.routes")

# Create router
router = APIRouter(prefix=settings.api_prefix)


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint"""
    return HealthStatus(
        status="healthy",
        version="1.0.0"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return response
    """
    try:
        logger.info(f"Processing chat request for user {request.user_id}")
        
        # Process the message
        response = await agent.chat(
            message=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            platform=request.platform,
            elder_mode=request.elder_mode
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a chat response using Server-Sent Events
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in agent.chat_stream(
                message=request.message,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                platform=request.platform,
                elder_mode=request.elder_mode
            ):
                yield f"data: {event}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {{'type': 'error', 'error': '{str(e)}'}}\n\n"
    
    return EventSourceResponse(event_generator())


@router.get("/users/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """
    Get user profile and memory information
    """
    try:
        profile = await memory_manager.get_user_profile(user_id)
        
        # Convert to response format
        response_data = {
            "user_id": user_id,
            "memory_count": len(profile.preferences) + len(profile.health_info)
        }
        
        if profile.personal_info:
            response_data["personal_info"] = {
                "name": profile.personal_info.name,
                "preferred_name": profile.personal_info.preferred_name,
                "age": profile.personal_info.age,
                "location": profile.personal_info.location
            }
        
        if profile.health_info:
            response_data["health_info"] = [
                {
                    "condition": health.condition,
                    "medications": health.medications,
                    "allergies": health.allergies
                }
                for health in profile.health_info
            ]
        
        if profile.preferences:
            response_data["preferences"] = [
                {
                    "category": pref.category,
                    "likes": pref.likes,
                    "dislikes": pref.dislikes
                }
                for pref in profile.preferences
            ]
        
        return UserProfileResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/memory")
async def update_user_memory(user_id: str, request: MemoryUpdateRequest):
    """
    Update user memory with new information
    """
    try:
        # This would be implemented to manually add memories
        # For now, return success
        return {"status": "success", "message": "Memory update queued"}
        
    except Exception as e:
        logger.error(f"Memory update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}/memory")
async def clear_user_memory(user_id: str, background_tasks: BackgroundTasks):
    """
    Clear user memory (for privacy/GDPR compliance)
    """
    try:
        # Add background task to clear memory
        background_tasks.add_task(_clear_user_data, user_id)
        return {"status": "success", "message": "Memory clearing initiated"}
        
    except Exception as e:
        logger.error(f"Memory clearing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _clear_user_data(user_id: str):
    """Background task to clear user data"""
    try:
        # Implementation would clear all user memories from database
        logger.info(f"Clearing data for user {user_id}")
        # await memory_manager.clear_user_data(user_id)
        
    except Exception as e:
        logger.error(f"Background task error: {e}")


@router.get("/tools")
async def list_available_tools():
    """
    List all available tools and their descriptions
    """
    from src.tools.base import tool_registry
    
    tools = []
    for tool in tool_registry.get_all_tools():
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        })
    
    return {"tools": tools}