"""
Memory management system for the Elderly Assistant Agent
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.memory.store import MemoryStore
from src.memory.schemas import (
    Memory, UserProfile, ConversationSummary, MemoryType,
    PersonalInfo, HealthInfo, UserPreference
)
from src.settings import settings


class MemoryManager:
    """Main memory management interface"""
    
    def __init__(self):
        self.store = MemoryStore()
        self.logger = logging.getLogger("memory.manager")
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure database is initialized"""
        if not self._initialized:
            await self.store.initialize()
            self._initialized = True
    
    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile with all information"""
        await self._ensure_initialized()
        profile = await self.store.get_user_profile(user_id)
        # The store always returns a UserProfile (creates empty one if none exists)
        return profile or UserProfile(user_id=user_id)
    
    async def update_user_profile(self, profile: UserProfile):
        """Update user profile"""
        await self._ensure_initialized()
        await self.store.store_user_profile(profile)
    
    async def store_memory(self, memory: Memory) -> str:
        """Store a memory"""
        await self._ensure_initialized()
        return await self.store.store_memory(memory)
    
    async def search_memories(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5
    ) -> List[Memory]:
        """Search memories relevant to a query"""
        await self._ensure_initialized()
        memories = await self.store.search_memories(user_id, query, limit)
        
        # Update access times for memories that have IDs
        for memory in memories:
            if memory.id:
                await self.store.update_memory_access(memory.id)
        
        return memories
    
    async def get_memories_by_type(
        self, 
        user_id: str, 
        memory_type: MemoryType,
        limit: int = 10
    ) -> List[Memory]:
        """Get memories of a specific type"""
        await self._ensure_initialized()
        return await self.store.get_memories(user_id, memory_type, limit)


class MemoryExtractor:
    """Extracts memories from conversations using LLM"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        # Initialize LLM without the API key first, let it use environment variable
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        )
        self.logger = logging.getLogger("memory.extractor")
    
    async def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        conversation_id: str
    ) -> List[Memory]:
        """Extract memories from a conversation"""
        
        # Create extraction prompt
        prompt = ChatPromptTemplate.from_template("""
You are a memory extraction system for an elderly care assistant. 
Analyze this conversation and extract important information to remember about the user.

Focus on:
1. Personal information (name, family, hobbies, background)
2. Health information (conditions, medications, doctor visits)
3. Preferences (likes, dislikes, routines)
4. Important events or concerns
5. Relationships and social connections

Conversation:
{conversation}

Extract memories in this JSON format:
{{
  "memories": [
    {{
      "type": "personal|health|preference|relationship|routine|important_event",
      "content": "Clear, specific memory content",
      "importance": 1-10,
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Only extract memories that are:
- Factual and specific
- Important for future conversations
- Relevant for elderly care
- Not already obvious or generic

Return valid JSON only.
""")
        
        # Format conversation
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        try:
            # Get extraction
            response = await self.llm.ainvoke(
                prompt.format(conversation=conversation_text)
            )
            
            # Parse response - handle both string and dict responses
            import json
            
            response_content = response.content
            if isinstance(response_content, str):
                extraction = json.loads(response_content)
            else:
                # If it's already a dict, use it directly
                extraction = response_content
            
            # Ensure extraction is a dict with memories key
            if not isinstance(extraction, dict):
                self.logger.warning("Extraction result is not a dict, skipping")
                return []
            
            # Create Memory objects
            memories = []
            for mem_data in extraction.get("memories", []):
                if not isinstance(mem_data, dict):
                    continue
                    
                memory = Memory(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_type=MemoryType(mem_data["type"]),
                    content=mem_data["content"],
                    importance=mem_data.get("importance", 5.0),
                    tags=mem_data.get("tags", [])
                )
                
                # Store the memory
                memory_id = await self.memory_manager.store_memory(memory)
                memory.id = memory_id
                memories.append(memory)
            
            self.logger.info(f"Extracted {len(memories)} memories from conversation")
            return memories
            
        except Exception as e:
            self.logger.error(f"Memory extraction failed: {e}")
            return []
    
    async def update_user_profile_from_memories(self, user_id: str):
        """Update user profile based on recent memories"""
        try:
            # Get current profile
            profile = await self.memory_manager.get_user_profile(user_id)
            
            # Get recent memories by type
            personal_memories = await self.memory_manager.get_memories_by_type(
                user_id, MemoryType.PERSONAL, 10
            )
            health_memories = await self.memory_manager.get_memories_by_type(
                user_id, MemoryType.HEALTH, 10
            )
            preference_memories = await self.memory_manager.get_memories_by_type(
                user_id, MemoryType.PREFERENCE, 10
            )
            
            # Update personal info
            if personal_memories and not profile.personal_info:
                profile.personal_info = PersonalInfo()
            
            # Update health info from memories
            for memory in health_memories:
                # Simple heuristic to avoid duplicates
                content_lower = memory.content.lower()
                existing = any(
                    content_lower in h.condition.lower() 
                    for h in profile.health_info
                )
                
                if not existing:
                    health_info = HealthInfo(condition=memory.content)
                    profile.health_info.append(health_info)
            
            # Update preferences from memories
            for memory in preference_memories:
                # Simple categorization
                category = "general"
                if "food" in memory.content.lower():
                    category = "food"
                elif "music" in memory.content.lower() or "podcast" in memory.content.lower():
                    category = "entertainment"
                
                existing_pref = next(
                    (p for p in profile.preferences if p.category == category),
                    None
                )
                
                if not existing_pref:
                    pref = UserPreference(
                        category=category,
                        likes=[memory.content] if "like" in memory.content.lower() else [],
                        dislikes=[memory.content] if "dislike" in memory.content.lower() else []
                    )
                    profile.preferences.append(pref)
            
            # Save updated profile
            await self.memory_manager.update_user_profile(profile)
            
        except Exception as e:
            self.logger.error(f"Profile update failed: {e}")
    
    async def create_conversation_summary(
        self,
        messages: List[Dict[str, str]],
        user_id: str,
        conversation_id: str
    ) -> ConversationSummary:
        """Create a summary of the conversation"""
        
        prompt = ChatPromptTemplate.from_template("""
Create a brief summary of this conversation with an elderly user.

Conversation:
{conversation}

Provide a JSON response with:
{{
  "summary": "Brief 1-2 sentence summary",
  "key_topics": ["topic1", "topic2", "topic3"],
  "mood": "positive|neutral|negative|concerned"
}}

Focus on:
- What the user needed help with
- Any important information shared
- The user's emotional state
- Key topics discussed

Return valid JSON only.
""")
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        try:
            response = await self.llm.ainvoke(
                prompt.format(conversation=conversation_text)
            )
            
            import json
            
            response_content = response.content
            if isinstance(response_content, str):
                summary_data = json.loads(response_content)
            else:
                summary_data = response_content
            
            # Ensure summary_data is a dict
            if not isinstance(summary_data, dict):
                raise ValueError("Summary data is not a dictionary")
            
            summary = ConversationSummary(
                conversation_id=conversation_id,
                user_id=user_id,
                summary=summary_data.get("summary", "Conversation completed"),
                key_topics=summary_data.get("key_topics", []),
                mood=summary_data.get("mood", "neutral"),
                message_count=len(messages)
            )
            
            # Store summary
            await self.memory_manager.store.store_conversation_summary(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary creation failed: {e}")
            return ConversationSummary(
                conversation_id=conversation_id,
                user_id=user_id,
                summary="Conversation completed",
                message_count=len(messages)
            )