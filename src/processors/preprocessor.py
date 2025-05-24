"""
Preprocessor node for input enhancement and memory retrieval
"""
import asyncio
from datetime import datetime
from typing import Dict, Any
import logging
import yaml

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.memory.manager import MemoryManager
from src.settings import settings

# Try to import aiofiles, fall back to sync if not available
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


class PreprocessorNode:
    """Handles input preprocessing and memory context injection"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.logger = logging.getLogger("preprocessor")
        self.prompts = None  # Will be loaded async
        
    async def _load_prompts(self):
        """Load personality prompts from YAML asynchronously"""
        if self.prompts is None:
            try:
                if HAS_AIOFILES:
                    # Use async file reading if aiofiles is available
                    async with aiofiles.open(settings.personality_prompt_path, 'r') as f:
                        content = await f.read()
                        self.prompts = yaml.safe_load(content)
                else:
                    # Fall back to sync reading in thread if aiofiles not available
                    def _sync_load():
                        with open(settings.personality_prompt_path, 'r') as f:
                            return yaml.safe_load(f)
                    
                    self.prompts = await asyncio.to_thread(_sync_load)
                    
            except Exception as e:
                self.logger.error(f"Error loading prompts: {e}")
                # Fallback prompts
                self.prompts = {
                    "system_prompt": "You are {agent_name}, a helpful AI assistant for elderly users. Today is {current_date}. Hello {user_name}!",
                    "memory_context_prompt": "Personal Information:\n{personal_info}\n\nHealth Information:\n{health_info}\n\nPreferences:\n{preferences}\n\nImportant Memories:\n{important_memories}\n\nRecent Context:\n{recent_context}",
                    "elder_mode_additions": "Please be patient, clear, and supportive in your responses."
                }
    
    async def process(self, state: AgentState) -> AgentState:
        """Process input and enhance with memory context"""
        self.logger.info(f"Preprocessing for user {state['user_id']}")
        
        # Ensure prompts are loaded
        await self._load_prompts()
        
        # Get the latest user message
        user_message = state["messages"][-1]
        if user_message.type != "human":
            return state
        
        # Convert content to string if it's not already
        user_input = user_message.content
        if isinstance(user_input, list):
            # Extract text content from list format
            user_input = self._extract_text_from_content(user_input)
        
        # Run preprocessing tasks in parallel
        tasks = [
            self._retrieve_memory_context(state["user_id"], user_input),
            self._enhance_input(user_input),
            self._check_health_indicators(user_input)
        ]
        
        memory_context, enhanced_input, health_check = await asyncio.gather(*tasks)
        
        # Build enhanced system prompt with memory context
        system_prompt = self._build_system_prompt(
            state["user_id"],
            memory_context,
            state.get("elder_mode", True)
        )
        
        # Update state
        state["memory_context"] = memory_context
        state["enhanced_context"] = {
            "original_input": user_input,
            "enhanced_input": enhanced_input,
            "timestamp": datetime.now(),
            "platform": state.get("platform", "api"),
            "has_memory": bool(memory_context.get("profile"))
        }
        state["health_check_needed"] = health_check
        state["preprocessing_done"] = True
        
        # Add system message with context
        if system_prompt:
            # Insert system message at the beginning if not present
            if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
                state["messages"].insert(0, SystemMessage(content=system_prompt))
            else:
                # Update existing system message
                state["messages"][0] = SystemMessage(content=system_prompt)
        
        return state
    
    async def _retrieve_memory_context(
        self, 
        user_id: str, 
        query: str
    ) -> Dict[str, Any]:
        """Retrieve relevant memories for the user"""
        try:
            # Get user profile
            profile = await self.memory_manager.get_user_profile(user_id)
            
            # Search for relevant memories based on current query
            relevant_memories = await self.memory_manager.search_memories(
                user_id, 
                query, 
                limit=3
            )
            
            # Get recent conversation summaries
            recent_summaries = profile.recent_summaries[:3] if profile.recent_summaries else []
            
            return {
                "profile": profile,
                "relevant_memories": relevant_memories,
                "recent_summaries": recent_summaries,
                "profile_string": profile.to_context_string() if profile else ""
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory context: {e}")
            return {}
    
    async def _enhance_input(self, input_text: str) -> str:
        """Enhance user input with clarifications"""
        # For MVP, just clean up the input
        enhanced = input_text.strip()
        
        # Add temporal context for ambiguous queries
        time_indicators = ["today", "now", "current", "latest"]
        if any(word in enhanced.lower() for word in time_indicators):
            enhanced += f" (Current date: {datetime.now().strftime('%Y-%m-%d')})"
        
        return enhanced
    
    async def _check_health_indicators(self, input_text: str) -> bool:
        """Check if the input contains health-related concerns"""
        health_keywords = [
            "pain", "hurt", "medication", "doctor", "sick", 
            "dizzy", "fall", "emergency", "help", "confused"
        ]
        
        input_lower = input_text.lower()
        return any(keyword in input_lower for keyword in health_keywords)
    
    def _build_system_prompt(
        self, 
        user_id: str, 
        memory_context: Dict[str, Any],
        elder_mode: bool
    ) -> str:
        """Build system prompt with memory context"""
        # Ensure prompts are available, use fallback if not
        if not self.prompts:
            self.prompts = {
                "system_prompt": "You are {agent_name}, a helpful AI assistant for elderly users. Today is {current_date}. Hello {user_name}!",
                "memory_context_prompt": "Personal Information:\n{personal_info}\n\nHealth Information:\n{health_info}\n\nPreferences:\n{preferences}\n\nImportant Memories:\n{important_memories}\n\nRecent Context:\n{recent_context}",
                "elder_mode_additions": "Please be patient, clear, and supportive in your responses."
            }
        
        # Start with base system prompt
        base_prompt = self.prompts["system_prompt"].format(
            agent_name=settings.agent_name,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            user_name=memory_context.get("profile", {}).personal_info.name 
                     if memory_context.get("profile") and memory_context["profile"].personal_info 
                     else "there"
        )
        
        # Add memory context if available
        if memory_context.get("profile_string"):
            memory_prompt = self.prompts["memory_context_prompt"].format(
                personal_info=self._format_personal_info(memory_context["profile"]),
                health_info=self._format_health_info(memory_context["profile"]),
                preferences=self._format_preferences(memory_context["profile"]),
                important_memories=self._format_important_memories(memory_context["relevant_memories"]),
                recent_context=self._format_recent_context(memory_context["recent_summaries"])
            )
            base_prompt += "\n\n" + memory_prompt
        
        # Add elder mode additions if enabled
        if elder_mode:
            base_prompt += "\n\n" + self.prompts["elder_mode_additions"]
        
        return base_prompt
    
    def _format_personal_info(self, profile) -> str:
        """Format personal info for prompt"""
        if not profile or not profile.personal_info:
            return "No personal information available yet."
        
        info = profile.personal_info
        parts = []
        
        if info.name:
            parts.append(f"- Name: {info.name}")
        if info.preferred_name:
            parts.append(f"- Preferred name: {info.preferred_name}")
        if info.age:
            parts.append(f"- Age: {info.age}")
        if info.location:
            parts.append(f"- Location: {info.location}")
        if info.family_members:
            parts.append(f"- Family: {', '.join(info.family_members)}")
            
        return "\n".join(parts) if parts else "No personal information available."
    
    def _format_health_info(self, profile) -> str:
        """Format health info for prompt"""
        if not profile or not profile.health_info:
            return "No health information recorded."
        
        parts = []
        for health in profile.health_info:
            parts.append(f"- Condition: {health.condition}")
            if health.medications:
                parts.append(f"  Medications: {', '.join(health.medications)}")
            if health.allergies:
                parts.append(f"  Allergies: {', '.join(health.allergies)}")
                
        return "\n".join(parts)
    
    def _format_preferences(self, profile) -> str:
        """Format preferences for prompt"""
        if not profile or not profile.preferences:
            return "No preferences recorded yet."
        
        parts = []
        for pref in profile.preferences:
            if pref.likes:
                parts.append(f"- Likes ({pref.category}): {', '.join(pref.likes)}")
            if pref.dislikes:
                parts.append(f"- Dislikes ({pref.category}): {', '.join(pref.dislikes)}")
                
        return "\n".join(parts)
    
    def _format_important_memories(self, memories) -> str:
        """Format important memories for prompt"""
        if not memories:
            return "No specific relevant memories for this conversation."
        
        parts = []
        for memory in memories[:3]:  # Limit to top 3
            parts.append(f"- {memory.content}")
            
        return "\n".join(parts)
    
    def _format_recent_context(self, summaries) -> str:
        """Format recent conversation summaries"""
        if not summaries:
            return "No recent conversations."
        
        parts = []
        for summary in summaries:
            date_str = summary.created_at.strftime("%Y-%m-%d")
            parts.append(f"- {date_str}: {summary.summary}")
            
        return "\n".join(parts)
    
    def _extract_text_from_content(self, content_list) -> str:
        """Extract text content from list format"""
        text_parts = []
        for item in content_list:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                text_parts.append(str(item['text']))
            elif isinstance(item, dict) and 'content' in item:
                text_parts.append(str(item['content']))
        return ' '.join(text_parts) if text_parts else str(content_list)