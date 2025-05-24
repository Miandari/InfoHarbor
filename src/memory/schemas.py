"""
Memory schemas for storing user information
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    """Types of memories that can be stored"""
    PERSONAL = "personal"
    HEALTH = "health"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    ROUTINE = "routine"
    IMPORTANT_EVENT = "important_event"
    CONVERSATION_SUMMARY = "conversation_summary"


class PersonalInfo(BaseModel):
    """Personal information about the user"""
    name: Optional[str] = None
    preferred_name: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    family_members: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    occupation: Optional[str] = None
    background: Optional[str] = None


class HealthInfo(BaseModel):
    """Health-related information"""
    condition: str
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    doctor_name: Optional[str] = None
    last_checkup: Optional[datetime] = None
    notes: Optional[str] = None


class UserPreference(BaseModel):
    """User preferences in different categories"""
    category: str  # e.g., "food", "entertainment", "communication"
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class DailyRoutine(BaseModel):
    """Daily routine information"""
    time: str  # e.g., "08:00"
    activity: str
    frequency: str  # daily, weekly, etc.
    notes: Optional[str] = None


class Memory(BaseModel):
    """Individual memory item"""
    id: Optional[str] = None
    user_id: str
    conversation_id: Optional[str] = None
    memory_type: MemoryType
    content: str
    importance: float = Field(default=1.0, ge=0.0, le=10.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = Field(default=0)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    """Summary of a conversation"""
    conversation_id: str
    user_id: str
    summary: str
    key_topics: List[str] = Field(default_factory=list)
    mood: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    message_count: int = 0


class UserProfile(BaseModel):
    """Complete user profile with all information"""
    user_id: str
    personal_info: Optional[PersonalInfo] = None
    health_info: List[HealthInfo] = Field(default_factory=list)
    preferences: List[UserPreference] = Field(default_factory=list)
    routines: List[DailyRoutine] = Field(default_factory=list)
    recent_summaries: List[ConversationSummary] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def to_context_string(self) -> str:
        """Convert profile to context string for prompts"""
        parts = []
        
        if self.personal_info:
            parts.append(f"Personal: {self.personal_info.model_dump_json()}")
        
        if self.health_info:
            parts.append(f"Health: {[h.model_dump() for h in self.health_info]}")
        
        if self.preferences:
            parts.append(f"Preferences: {[p.model_dump() for p in self.preferences]}")
        
        return "\n".join(parts)