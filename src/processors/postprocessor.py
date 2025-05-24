"""
Postprocessor node for response formatting and memory updates
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import logging

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.memory.manager import MemoryManager, MemoryExtractor
from src.settings import settings


class PostprocessorNode:
    """Handles response postprocessing and memory updates"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.memory_extractor = MemoryExtractor(self.memory_manager)
        self.logger = logging.getLogger("postprocessor")
    
    async def process(self, state: AgentState) -> AgentState:
        """Post-process the agent's response"""
        
        ai_message = state["messages"][-1]
        if ai_message.type != "ai":
            return state
        
        # Convert content to string if it's not already
        ai_response = ai_message.content
        if isinstance(ai_response, list):
            # Extract text content from list format
            ai_response = self._extract_text_from_content(ai_response)
        
        # Process the response
        processed_response = await asyncio.gather(
            self._format_response_for_platform(ai_response, state),
            self._quality_check_response(ai_response, state)
        )
        
        formatted_response, quality_feedback = processed_response
        
        # Create final response
        final_response = {
            "response": formatted_response,
            "conversation_id": state["conversation_id"],
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "latency_ms": int((datetime.now().timestamp() - state["start_time"]) * 1000),
                "user_id": state["user_id"],
                "tools_used": state.get("tools_used", []),
                "memory_updates_count": len(state.get("memory_updates", [])),
                "platform": state.get("platform", "api"),
                "elder_mode": state.get("elder_mode", True)
            }
        }
        
        # Update state
        state["final_response"] = final_response
        state["postprocessing_done"] = True
        
        # Check if revision is needed based on quality feedback
        if quality_feedback.get("needs_revision") and state.get("retry_count", 0) < 2:
            state["needs_revision"] = True
            state["revision_feedback"] = quality_feedback.get("issues", [])
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    async def _format_response_for_platform(
        self, 
        response: str, 
        state: AgentState
    ) -> str:
        """Format response based on platform and user preferences"""
        
        platform = state.get("platform", "api")
        elder_mode = state.get("elder_mode", True)
        
        # Base formatting
        formatted = response.strip()
        
        if elder_mode:
            # Elder-friendly formatting
            formatted = self._apply_elder_friendly_formatting(formatted)
        
        if platform == "terminal":
            # Terminal-specific formatting
            formatted = self._apply_terminal_formatting(formatted)
        elif platform == "voice":
            # Voice-specific formatting (remove markdown, etc.)
            formatted = self._apply_voice_formatting(formatted)
        
        return formatted
    
    def _apply_elder_friendly_formatting(self, text: str) -> str:
        """Apply elder-friendly text formatting"""
        # Break up long paragraphs
        sentences = text.split('. ')
        
        # Group sentences into shorter paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Start new paragraph after 2-3 sentences
            if len(current_paragraph) >= 3:
                paragraphs.append('. '.join(current_paragraph) + '.')
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph))
        
        # Join with double line breaks for readability
        return '\n\n'.join(paragraphs)
    
    def _apply_terminal_formatting(self, text: str) -> str:
        """Apply terminal-specific formatting"""
        # Keep markdown for terminal display
        return text
    
    def _apply_voice_formatting(self, text: str) -> str:
        """Apply voice-specific formatting"""
        # Remove markdown formatting
        import re
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Remove bullet points and replace with "First, Second, etc."
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip().startswith('•') or line.strip().startswith('-'):
                # Convert bullet to numbered format
                content = line.strip()[1:].strip()
                formatted_lines.append(content)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    async def _extract_and_store_memories(self, state: AgentState) -> List[Dict[str, Any]]:
        """Extract and store memories from the conversation"""
        try:
            # Get conversation messages
            messages = []
            for msg in state["messages"]:
                if hasattr(msg, 'content'):
                    messages.append({
                        "role": msg.__class__.__name__.lower().replace("message", ""),
                        "content": msg.content
                    })
            
            # Extract memories
            extracted_memories = await self.memory_extractor.extract_from_conversation(
                messages=messages,
                user_id=state["user_id"],
                conversation_id=state["conversation_id"]
            )
            
            # Convert to dict format for response
            memory_updates = []
            for memory in extracted_memories:
                memory_updates.append({
                    "type": memory.memory_type,
                    "content": memory.content,
                    "importance": memory.importance,
                    "timestamp": memory.created_at.isoformat()
                })
            
            return memory_updates
            
        except Exception as e:
            self.logger.error(f"Memory extraction error: {e}")
            return []
    
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
    
    async def _quality_check_response(
        self, 
        response: str, 
        state: AgentState
    ) -> Dict[str, Any]:
        """Check response quality and suggest improvements"""
        
        issues = []
        
        # Check response length
        if len(response) < 20:
            issues.append("Response too short")
        elif len(response) > 1000 and state.get("elder_mode"):
            issues.append("Response too long for elder mode")
        
        # Check for appropriate tone
        if state.get("elder_mode"):
            # Check for elder-friendly language
            complex_words = ["utilize", "implement", "facilitate", "optimize"]
            for word in complex_words:
                if word.lower() in response.lower():
                    issues.append(f"Complex word detected: {word}")
        
        # Check for health-related safety
        if state.get("health_check_needed"):
            safety_phrases = ["contact your doctor", "healthcare provider", "medical advice"]
            has_safety_warning = any(phrase in response.lower() for phrase in safety_phrases)
            if not has_safety_warning:
                issues.append("Health query missing safety disclaimer")
        
        # Determine if revision is needed
        needs_revision = len(issues) > 2  # Only revise if multiple issues
        
        return {
            "needs_revision": needs_revision,
            "issues": issues,
            "quality_score": max(0, 10 - len(issues))
        }