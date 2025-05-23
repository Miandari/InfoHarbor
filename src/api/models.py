"""
Pydantic models for Elder Care Assistant API
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="Unique user identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Assistant response")
    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    processed_input: Optional[Dict[str, Any]] = Field(None, description="Processed input details")
    tool_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Tool execution results")
    output_formats: Optional[Dict[str, str]] = Field(None, description="Different output formats")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class UserProfileResponse(BaseModel):
    """Response model for user profile"""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="User name")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    health_info: Dict[str, Any] = Field(default_factory=dict, description="Health information")
    family_info: Dict[str, Any] = Field(default_factory=dict, description="Family information")
    created_at: datetime = Field(..., description="Profile creation timestamp")
    updated_at: datetime = Field(..., description="Profile last update timestamp")


class MemoryResponse(BaseModel):
    """Response model for memory entries"""
    id: Optional[str] = Field(None, description="Memory identifier")
    user_id: str = Field(..., description="User identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Type of memory")
    timestamp: datetime = Field(..., description="Memory timestamp")
    importance: float = Field(1.0, description="Memory importance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional memory metadata")


class ToolResponse(BaseModel):
    """Response model for tool execution"""
    tool_name: str = Field(..., description="Name of the tool")
    result: Dict[str, Any] = Field(..., description="Tool execution result")
    success: bool = Field(..., description="Whether tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(None, description="Tool execution time in seconds")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field("1.0.0", description="API version")


class PreprocessedInput(BaseModel):
    """Model for preprocessed user input"""
    original_input: str = Field(..., description="Original user input")
    cleaned_input: str = Field(..., description="Cleaned input")
    processed_input: str = Field(..., description="Fully processed input")
    intent: str = Field(..., description="Detected intent")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted entities")
    context_flags: List[str] = Field(default_factory=list, description="Context flags")
    urgency: str = Field(..., description="Urgency level")
    timestamp: str = Field(..., description="Processing timestamp")


class PostprocessedOutput(BaseModel):
    """Model for postprocessed output"""
    original_response: str = Field(..., description="Original response")
    processed_response: str = Field(..., description="Processed response")
    output_formats: Dict[str, str] = Field(..., description="Different output formats")
    accessibility_applied: bool = Field(..., description="Whether accessibility features were applied")
    timestamp: str = Field(..., description="Processing timestamp")