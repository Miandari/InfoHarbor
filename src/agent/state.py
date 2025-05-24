"""
State definition for the LangGraph agent workflow
"""
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow"""
    
    # Core conversation data - using add_messages for LangGraph Studio compatibility
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: str
    user_id: str
    
    # Memory and context
    memory_context: Optional[Dict[str, Any]]
    memory_updates: List[Dict[str, Any]]
    enhanced_context: Optional[Dict[str, Any]]
    system_prompt: str
    
    # Tool usage
    tools_used: List[str]
    tool_results: Dict[str, Any]
    
    # Timing and flow control
    start_time: float
    preprocessing_done: bool
    agent_done: bool
    postprocessing_done: bool
    
    # User preferences and modes
    elder_mode: bool
    requires_clarification: bool
    health_check_needed: bool
    platform: str
    response_format: str
    
    # Output and metadata
    final_response: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str]
    
    # Retry handling
    retry_count: int
    needs_revision: Optional[bool]
    revision_feedback: Optional[List[str]]