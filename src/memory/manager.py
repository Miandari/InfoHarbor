"""
Memory management logic for Elder Care Assistant
"""
from typing import Dict, Any, List, Optional
import sqlite3
import json
from datetime import datetime
from src.memory.schemas import UserProfile, Memory
from src.memory.store import MemoryStore


class MemoryManager:
    """Manages user memories and profiles for personalized interactions"""
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.store = MemoryStore(db_path)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        profile_data = await self.store.get_user_profile(user_id)
        if profile_data:
            return UserProfile(**profile_data)
        return None
    
    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """Create a new user profile"""
        profile = UserProfile(
            user_id=user_id,
            name=profile_data.get("name", ""),
            preferences=profile_data.get("preferences", {}),
            health_info=profile_data.get("health_info", {}),
            family_info=profile_data.get("family_info", {}),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        await self.store.save_user_profile(profile)
        return profile
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserProfile]:
        """Update user profile"""
        profile = await self.get_user_profile(user_id)
        if not profile:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.utcnow()
        await self.store.save_user_profile(profile)
        return profile
    
    async def add_memory(self, user_id: str, content: str, memory_type: str = "conversation") -> Memory:
        """Add a new memory"""
        memory = Memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.utcnow(),
            importance=1.0
        )
        
        await self.store.save_memory(memory)
        return memory
    
    async def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content similarity"""
        return await self.store.search_memories(user_id, query, limit)
    
    async def get_recent_memories(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get most recent memories for a user"""
        return await self.store.get_recent_memories(user_id, limit)