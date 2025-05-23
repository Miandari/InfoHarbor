"""
Memory data models for Elder Care Assistant
"""
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class UserProfile:
    """User profile data model"""
    user_id: str
    name: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    health_info: Dict[str, Any] = field(default_factory=dict)
    family_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Memory:
    """Memory data model"""
    user_id: str
    content: str
    memory_type: str = "conversation"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class UserProfileRequest(BaseModel):
    """Pydantic model for user profile API requests"""
    name: str
    preferences: Dict[str, Any] = {}
    health_info: Dict[str, Any] = {}
    family_info: Dict[str, Any] = {}


class MemoryRequest(BaseModel):
    """Pydantic model for memory API requests"""
    content: str
    memory_type: str = "conversation"
    importance: float = 1.0
    metadata: Dict[str, Any] = {}