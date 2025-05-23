"""
Postprocessor node for response formatting and quality checks
"""
import re
import time
from datetime import datetime
from typing import Dict, Any, List
import logging

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.memory.manager import MemoryManager, MemoryExtractor
from src.memory.schemas import ConversationSummary
from src.config.settings import settings


class PostprocessorNode:
    """Handles response formatting and post-processing"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.memory_extractor = MemoryExtractor(self.memory_manager)
        self.logger = logging.getLogger("postprocessor")
        
    async def process(self, state: AgentState) -> AgentState:
        """Process and format the agent's response"""
        self.logger.info(f"Postprocessing response for user {state['user_id']}")
        
        # Get the latest AI message
        ai_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                ai_message = msg
                break
                
        if not ai_message:
            state["error"] = "No AI response found"
            return state
        
        # Extract and store memories from the conversation
        await self._extract_memories(state)
        
        # Format the response
        formatted_response = self._format_response(
            ai_message.content,
            state.get("platform", "api"),
            state.get("elder_mode", True)
        )
        
        # Perform quality checks
        quality_issues = self._quality_check(formatted_response)
        if quality_issues and state.get("retry_count", 0) < 2:
            state["needs_revision"] = True
            state["revision_feedback"] = quality_issues
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state
        
        # Calculate metadata
        metadata = self._build_metadata(state)
        
        # Build final response
        state["final_response"] = {
            "response": formatted_response,
            "conversation_id": state["conversation_id"],
            "created_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        state["postprocessing_done"] = True
        
        # Save conversation summary
        await self._save_conversation_summary(state)
        
        return state
    
    async def _extract_memories(self, state: AgentState):
        """Extract and store memories from the conversation"""
        try:
            # Convert messages to simple format for extraction
            messages = []
            for msg in state["messages"]:
                if hasattr(msg, 'content'):
                    messages.append({
                        "role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
                        "content": msg.content
                    })
            
            # Extract memories
            extracted = await self.memory_extractor.extract_from_conversation(
                messages=messages,
                user_id=state["user_id"],
                conversation_id=state["conversation_id"]
            )
            
            # Track what was extracted
            state["memory_updates"] = [
                {
                    "type": memory.memory_type,
                    "content": memory.content[:100],  # Preview
                    "importance": memory.importance
                }
                for memory in extracted
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to extract memories: {e}")
    
    def _format_response(
        self, 
        response: str, 
        platform: str, 
        elder_mode: bool
    ) -> str:
        """Format response based on platform and user needs"""
        
        # Basic formatting
        formatted = response.strip()
        
        # Elder mode formatting
        if elder_mode:
            # Break long sentences for clarity
            formatted = self._break_long_sentences(formatted)
            
            # Add clarifying phrases
            formatted = self._add_clarifications(formatted)
            
            # Ensure friendly tone
            formatted = self._ensure_friendly_tone(formatted)
        
        # Platform-specific formatting
        if platform == "web":
            # Convert markdown for web display
            formatted = self._enhance_markdown(formatted)
        elif platform == "voice":
            # Remove markdown for voice
            formatted = self._strip_markdown(formatted)
            
        return formatted
    
    def _quality_check(self, response: str) -> List[str]:
        """Check response quality and return issues"""
        issues = []
        
        # Check for incomplete sentences
        if response and not response.rstrip().endswith(('.', '!', '?', '...')):
            issues.append("Response appears incomplete")
            
        # Check for very short responses to complex questions
        if len(response.split()) < 10:
            issues.append("Response may be too brief")
            
        # Check for repetition
        sentences = response.split('.')
        if len(sentences) > 2:
            for i in range(len(sentences) - 1):
                if sentences[i].strip() and sentences[i].strip() == sentences[i + 1].strip():
                    issues.append("Response contains repetition")
                    break
                    
        # Check for error messages
        error_indicators = ["error", "sorry", "unable", "can't", "cannot"]
        if any(indicator in response.lower() for indicator in error_indicators):
            # Make sure it's not just being polite
            if "please" not in response.lower() and "help" not in response.lower():
                issues.append("Response may contain unhandled errors")
                
        return issues
    
    def _build_metadata(self, state: AgentState) -> Dict[str, Any]:
        """Build metadata for the response"""
        end_time = time.time()
        start_time = state.get("start_time", end_time)
        
        metadata = {
            "latency_ms": int((end_time - start_time) * 1000),
            "user_id": state["user_id"],
            "tools_used": state.get("tools_used", []),
            "memory_updates": {
                "count": len(state.get("memory_updates", [])),
                "types": list(set(u["type"] for u in state.get("memory_updates", [])))
            },
            "platform": state.get("platform", "api"),
            "elder_mode": state.get("elder_mode", True),
            "health_check_triggered": state.get("health_check_needed", False),
            "retry_count": state.get("retry_count", 0)
        }
        
        # Add token usage if available
        # In production, you'd track actual token usage
        metadata["tokens_used"] = {
            "prompt": 0,  # Would be calculated
            "completion": 0,  # Would be calculated
            "total": 0
        }
        
        return metadata
    
    async def _save_conversation_summary(self, state: AgentState):
        """Save a summary of the conversation"""
        try:
            # For MVP, create a simple summary
            # In production, use an LLM to generate a better summary
            messages = state["messages"]
            
            # Get topics discussed
            topics = []
            if state.get("tools_used"):
                topics.extend(state["tools_used"])
            if state.get("health_check_needed"):
                topics.append("health")
                
            # Create summary
            summary = ConversationSummary(
                user_id=state["user_id"],
                source_conversation_id=state["conversation_id"],
                summary=f"Discussed {', '.join(topics) if topics else 'general topics'}",
                key_topics=topics,
                sentiment="positive",  # Would be analyzed in production
                content=f"Conversation on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            await self.memory_manager.store_memory(summary)
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation summary: {e}")
    
    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences for easier reading"""
        sentences = text.split('. ')
        formatted_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 100:  # Long sentence
                # Try to break at commas
                parts = sentence.split(', ')
                if len(parts) > 2:
                    # Add pauses
                    sentence = ', '.join(parts[:2]) + '.\n' + ', '.join(parts[2:])
            formatted_sentences.append(sentence)
            
        return '. '.join(formatted_sentences)
    
    def _add_clarifications(self, text: str) -> str:
        """Add clarifying phrases for elderly users"""
        # Add "Did you get that?" for complex instructions
        if any(word in text.lower() for word in ['step', 'first', 'then', 'next']):
            if not text.endswith('?'):
                text += "\n\nDoes this make sense so far?"
                
        return text
    
    def _ensure_friendly_tone(self, text: str) -> str:
        """Ensure the response has a friendly tone"""
        # Add a friendly closing if missing
        friendly_closings = [
            "Is there anything else I can help you with?",
            "Please let me know if you need any clarification!",
            "I'm here if you have any other questions!",
            "Feel free to ask if you need more help!"
        ]
        
        if not any(closing in text for closing in friendly_closings):
            if not text.endswith('?') and len(text) < 200:
                text += f"\n\n{friendly_closings[0]}"
                
        return text
    
    def _enhance_markdown(self, text: str) -> str:
        """Enhance markdown for web display"""
        # Bold important words
        important_words = ['important', 'note', 'warning', 'remember']
        for word in important_words:
            text = re.sub(
                f'\\b({word})\\b', 
                f'**{word}**', 
                text, 
                flags=re.IGNORECASE
            )
            
        return text
    
    def _strip_markdown(self, text: str) -> str:
        """Remove markdown for voice output"""
        # Remove bold
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove italic
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Remove headers
        text = re.sub(r'#+\s*(.*?)\n', r'\1. ', text)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        return text