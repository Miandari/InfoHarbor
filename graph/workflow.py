"""
Workflow definition for the information assistant.

This module defines the LangGraph workflow structure:
- Nodes for processing user input
- Graph edges that define the flow between nodes
- Conditional routing based on state
"""

from typing import Annotated, Any, Dict, List, Optional, Sequence
import operator
import os

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import END, StateGraph

# Import ConfigSchema from schemas to avoid circular imports
from graph.schemas import ConfigSchema
from graph.state import InfoAssistantState
from graph.transitions import StateTransitions
from utils.direct_response import get_direct_response
from utils.middleware import (
    add_metadata,
    limit_conversation_history,
)

# Import nodes after schema imports to avoid circular dependency
from graph.nodes import (
    food_ordering_node,
    news_agent_node,
    podcast_agent_node,
    route_query,
    prepare_context,
)

# Import memory nodes
from memory.memory_nodes import (
    memory_retrieval_node,
    memory_extraction_node,
    memory_update_node,
    add_memory_context_to_prompt
)


def create_information_graph(override_nodes: Dict[str, Any] = None):
    """
    Create the LangGraph workflow for the information assistant.
    
    Args:
        override_nodes: Optional dictionary of custom node implementations for testing
        
    Returns:
        Compiled StateGraph ready to be run
    """
    # Create graph with state definition
    workflow = StateGraph(InfoAssistantState)
    
    # Add memory-related nodes
    workflow.add_node("memory_retrieval", override_nodes["memory_retrieval"] if override_nodes and "memory_retrieval" in override_nodes else memory_retrieval_node)
    workflow.add_node("add_memory_context", override_nodes["add_memory_context"] if override_nodes and "add_memory_context" in override_nodes else add_memory_context_to_prompt)
    workflow.add_node("memory_extraction", override_nodes["memory_extraction"] if override_nodes and "memory_extraction" in override_nodes else memory_extraction_node)
    workflow.add_node("memory_update", override_nodes["memory_update"] if override_nodes and "memory_update" in override_nodes else memory_update_node)
    
    # Add all the existing nodes to the graph
    workflow.add_node("route_query", override_nodes["route_query"] if override_nodes and "route_query" in override_nodes else route_query)
    workflow.add_node("prepare_context", override_nodes["prepare_context"] if override_nodes and "prepare_context" in override_nodes else prepare_context)
    workflow.add_node("food_ordering", override_nodes["food_ordering"] if override_nodes and "food_ordering" in override_nodes else food_ordering_node)
    workflow.add_node("podcast_agent", override_nodes["podcast_agent"] if override_nodes and "podcast_agent" in override_nodes else podcast_agent_node)
    workflow.add_node("news_agent", override_nodes["news_agent"] if override_nodes and "news_agent" in override_nodes else news_agent_node)
    
    # Define the workflow entry point - starts with memory retrieval for user context
    workflow.set_entry_point("memory_retrieval")
    
    # First, retrieve memories for the current user before anything else
    workflow.add_edge("memory_retrieval", "add_memory_context")
    
    # Add memory context to influence reasoning
    workflow.add_edge("add_memory_context", "route_query")
    
    # Routing from query router node
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state["current_task"],
        {
            "food_order": "food_ordering",
            "podcast": "podcast_agent",
            "news": "news_agent",
            "general": "prepare_context",
            None: "prepare_context",
        },
    )
    
    # All execution paths should extract and update memory before completion
    workflow.add_edge("prepare_context", "memory_extraction")
    workflow.add_edge("food_ordering", "memory_extraction")
    workflow.add_edge("podcast_agent", "memory_extraction")
    workflow.add_edge("news_agent", "memory_extraction")
    
    # Extract memory items from the conversation
    workflow.add_edge("memory_extraction", "memory_update")
    
    # Update user memory store and end
    workflow.add_edge("memory_update", END)
    
    # Note: middleware is applied manually in the handle_user_input function
    # since the current LangGraph version doesn't support middleware directly
    
    return workflow.compile()

# Create an alias for the create_information_graph function to match what main.py is importing
create_info_assistant = create_information_graph


def handle_user_input(graph: Any, state: Optional[InfoAssistantState], user_message: str, user_id: Optional[str] = None) -> InfoAssistantState:
    """
    Process user input and update the state using the graph.
    
    Args:
        graph: Compiled graph instance
        state: Current state or None for first interaction
        user_message: User's message text
        user_id: Optional user identifier for memory persistence
        
    Returns:
        Updated state after processing
    """
    # Create initial state if none exists
    if state is None:
        state = StateTransitions.create_clean_state()
        
    # Set user ID if provided for memory linkage
    if user_id:
        state = StateTransitions.identify_user(state, user_id)
    
    # Check if user input should be handled directly (without LLM)
    direct_response = get_direct_response(user_message)
    if direct_response:
        # Create simple direct response
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=user_message))
        messages.append(AIMessage(content=direct_response))
        return {**state, "messages": messages}
    
    # Check for task transitions based on user input
    state = StateTransitions.transition_from_task(state, user_message)
    
    # Add user message to state and run graph
    messages = state.get("messages", [])
    messages.append(HumanMessage(content=user_message))
    state = {**state, "messages": messages}
    
    # Apply middleware manually before processing
    state = limit_conversation_history(state)
    state = add_metadata(state)
    
    # Process and return the result
    result = graph.invoke(state)
    
    # Apply middleware manually after processing
    result = limit_conversation_history(result)
    result = add_metadata(result)
    
    # Verify tool state consistency
    result = verify_tools_state_consistency(result)
        
    return result

def verify_tools_state_consistency(state: InfoAssistantState) -> InfoAssistantState:
    """
    Verify and correct tool state consistency before returning the final state.
    
    This function ensures:
    1. No pending tools remain after graph completion
    2. Tool results and pending state are consistent
    
    Args:
        state: The current state after graph processing
        
    Returns:
        Corrected state with consistent tool tracking
    """
    # Check for pending tools
    if state.get("pending_tools", []):
        # Log warning for debugging
        print(f"Warning: Some tools were still pending after graph completion: {state['pending_tools']}")
        
        # Clean up state using the centralized method
        from graph.transitions import StateTransitions
        state = StateTransitions.clear_pending_tools(state)
        
        # Also ensure tool_results reflects this clean state
        if state.get("tool_results", {}).get("pending") is True:
            # Update the tool results to be consistent with pending_tools state
            tool_results = state.get("tool_results", {})
            tool_results["pending"] = False
            state = {**state, "tool_results": tool_results}
            
    return state