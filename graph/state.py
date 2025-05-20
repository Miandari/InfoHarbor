"""
State type definitions for the information assistant LangGraph.

This module implements modern best practices for LangGraph state management:
1. Using TypedDict for structured state definitions
2. Adding proper reducers (like operator.add) for lists that should be appended to
3. Including clear documentation for each state field
"""

import operator
from typing import List, Dict, Any, Annotated, TypedDict, Sequence, Union, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

class StateSnapshot(TypedDict, total=False):
    """A snapshot of state at a specific version point."""
    timestamp: str
    version: int
    description: Optional[str]
    state_data: Dict[str, Any]

class StateHistory(TypedDict):
    """History of state changes for undo/redo functionality."""
    snapshots: List[StateSnapshot]
    current_index: int  # Points to current position in history
    max_snapshots: int  # Maximum number of snapshots to keep

class InfoAssistantState(TypedDict):
    """
    The complete state structure for the information assistant.
    
    This TypedDict helps enforce consistent state structure throughout the app.
    Keeping all state fields properly typed helps prevent runtime errors.
    """
    # Core conversation state
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], operator.add]  # Uses concatenation reducer for messages
    reasoning: Annotated[List[str], operator.add]  # Reasoning chain for ReAct pattern, should be appended to
    
    # Task tracking
    current_task: Annotated[Optional[str], "Current task type being handled"]
    last_tool_used: Annotated[Optional[str], "The last tool that was used"]
    
    # History tracking for different content types
    podcast_history: Annotated[List[Dict], operator.add]  # History of podcast recommendations
    news_history: Annotated[List[Dict], operator.add]  # History of news searches
    food_order_history: Annotated[List[Dict], operator.add]  # History of food orders
    
    # Task-specific state
    food_order_state: Annotated[Optional[Literal["collecting_details", "completed", "error"]], "Current state of food ordering process"]
    
    # Context and working data
    context: Dict[str, Any]  # General context information
    tool_results: Annotated[Optional[Dict], "Results from the last tool used"]
    working_memory: Dict[str, Any]  # Working memory to store intermediate results and reasoning state
    
    # Planning and reflection
    next_actions: Annotated[List[Dict[str, Any]], operator.add]  # Planned next actions for the ReAct pattern
    pending_tools: Annotated[List[str], operator.add]  # Tools that still need to be called
    reflection: Annotated[Optional[str], "Agent's reflection on current progress"]
    
    # Versioning and history
    state_version: Annotated[Optional[int], "Version number of the state for tracking changes"]
    state_snapshots: Dict[str, Dict[str, Any]]
    state_history: StateHistory