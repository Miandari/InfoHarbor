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

class InfoAssistantState(TypedDict):
    """
    State for the information assistant workflow.
    
    Each field is annotated with its purpose and has appropriate reducer 
    functions assigned where needed.
    """
    # Conversation state
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], operator.add]  # Uses concatenation reducer for messages
    
    # History tracking with append reducers
    podcast_history: Annotated[List[Dict], operator.add]  # History of podcast recommendations
    news_history: Annotated[List[Dict], operator.add]  # History of news searches
    food_order_history: Annotated[List[Dict], operator.add]  # History of food orders
    
    # Task state fields (override by default)
    food_order_state: Annotated[Optional[Literal["collecting_details", "completed", "error"]], "Current state of food ordering process"]
    current_task: Annotated[Optional[Literal["podcast", "news", "food_order", "general"]], "Current task type being handled"]
    last_tool_used: Annotated[Optional[str], "The last tool that was used"]
    
    # User identity and memory
    user_id: Annotated[str, "Unique identifier for the current user"]
    user_memory: Annotated[Dict[str, Any], "User memory storage with preferences, facts, etc."]
    memory_updates: Annotated[List[Dict[str, Any]], operator.add]  # Tracked memory updates to be stored
    
    # Context and results
    context: Dict[str, Any]  # General context information
    tool_results: Annotated[Optional[Dict], "Results from the last tool used"]
    
    # ReAct pattern fields with appropriate reducers
    reasoning: Annotated[List[str], operator.add]  # Reasoning chain for ReAct pattern, should be appended to
    next_actions: Annotated[List[Dict[str, Any]], operator.add]  # Planned next actions for the ReAct pattern
    working_memory: Dict[str, Any]  # Working memory to store intermediate results and reasoning state
    pending_tools: Annotated[List[str], operator.add]  # Tools that still need to be called
    reflection: Annotated[Optional[str], "Agent's reflection on current progress"]
    
    # Version tracking for potential rollbacks
    state_version: Annotated[Optional[int], "Version number of the state for tracking changes"]