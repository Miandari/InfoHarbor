"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message")
    user_id: str = Field(..., description="Unique user identifier")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    platform: str = Field("api", description="Platform type")
    elder_mode: bool = Field(True, description="Enable elder-friendly features")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Agent's response")
    conversation_id: str = Field(..., description="Conversation identifier")
    created_at: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field("healthy", description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = Field("1.0.0", description="API version")


class MemoryUpdateRequest(BaseModel):
    """Request to update user memory"""
    user_id: str = Field(..., description="User identifier")
    memory_type: str = Field(..., description="Type of memory to update")
    content: Dict[str, Any] = Field(..., description="Memory content")


class UserProfileResponse(BaseModel):
    """User profile response"""
    user_id: str
    personal_info: Optional[Dict[str, Any]] = None
    health_info: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: List[Dict[str, Any]] = Field(default_factory=list)
    memory_count: int = 0