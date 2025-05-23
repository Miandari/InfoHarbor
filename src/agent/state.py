"""
Agent state definition for LangGraph
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from datetime import datetime


class AgentState(TypedDict):
    """State schema for the elderly assistant agent"""
    
    # Core conversation state
    messages: List[BaseMessage]
    conversation_id: str
    user_id: str
    
    # Memory context
    memory_context: Optional[Dict[str, Any]]
    memory_updates: List[Dict[str, Any]]  # Track what was stored
    
    # Enhanced context from preprocessor
    enhanced_context: Optional[Dict[str, Any]]
    system_prompt: str
    
    # Tool tracking
    tools_used: List[str]
    tool_results: Dict[str, Any]
    
    # Processing metadata
    start_time: float
    preprocessing_done: bool
    agent_done: bool
    postprocessing_done: bool
    
    # Elder-specific features
    elder_mode: bool
    requires_clarification: bool
    health_check_needed: bool
    
    # Response formatting
    platform: str  # web, api, etc.
    response_format: str  # text, markdown, etc.
    
    # Final output
    final_response: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    retry_count: int