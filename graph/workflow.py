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
import uuid

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
    
    # Import the state coordinator for message deduplication
    from utils.state_coordinator import get_state_coordinator
    coordinator = get_state_coordinator()
    
    # Define message deduplication function - using our coordinator now
    def deduplicate_messages(state):
        """Middleware to prevent duplicate messages in the state."""
        return coordinator._sanitize_state(state)
    
    # Create node wrapper using the state coordinator
    def wrap_node_with_coordination(node_func, node_name=None):
        """Wrap a node with our state coordination system."""
        if node_name is None:
            # Try to get the function name
            node_name = getattr(node_func, "__name__", "unknown_node")
            
        # Get a function wrapped with our coordinator
        return coordinator.wrap_node_for_coordination(node_func, node_name)
    
    # Legacy deduplication wrapper - kept for backward compatibility
    def wrap_node_with_deduplication(node_func):
        def wrapped_node(state):
            # Add defensive check for non-dict state
            if not isinstance(state, dict):
                print(f"[NODE DEDUP] ERROR: Received non-dict state of type {type(state)} in {node_func.__name__}")
                # Create a clean state as fallback
                from graph.transitions import StateTransitions
                state = StateTransitions.create_clean_state()
                print("[NODE DEDUP] Created clean state as fallback")
                
            # Get original message count for diagnostics
            original_count = len(state.get("messages", []))
            
            # Apply deduplication before node execution
            deduplicated_state = deduplicate_messages(state)
            dedup_count = len(deduplicated_state.get("messages", []))
            
            if original_count != dedup_count:
                print(f"[NODE DEDUP] Pre-{node_func.__name__}: {original_count} → {dedup_count} messages")
                
            # Execute the original node function
            result = node_func(deduplicated_state)
            
            # Add defensive check for unexpected result type
            if result is None or isinstance(result, bool):
                print(f"[NODE DEDUP] ERROR: {node_func.__name__} returned {type(result)} instead of dict")
                # Return the input state as fallback
                return deduplicated_state
                
            # Apply deduplication after node execution
            if isinstance(result, dict) and "messages" in result:
                result_count = len(result.get("messages", []))
                final_result = deduplicate_messages(result)
                final_count = len(final_result.get("messages", []))
                
                if result_count != final_count:
                    print(f"[NODE DEDUP] Post-{node_func.__name__}: {result_count} → {final_count} messages")
                    
                return final_result
            return result
        return wrapped_node
    
    # Wrap all nodes with coordination instead of deduplication
    memory_retrieval = memory_retrieval_node
    add_memory_context = add_memory_context_to_prompt
    memory_extraction = memory_extraction_node
    memory_update = memory_update_node
    route_query_node = route_query
    prepare_context_node = prepare_context
    food_ordering = food_ordering_node
    podcast_agent = podcast_agent_node
    news_agent = news_agent_node
    
    # Apply wrappers based on configuration
    if override_nodes:
        memory_retrieval = override_nodes.get("memory_retrieval", wrap_node_with_coordination(memory_retrieval_node, "memory_retrieval"))
        add_memory_context = override_nodes.get("add_memory_context", wrap_node_with_coordination(add_memory_context_to_prompt, "add_memory_context"))
        memory_extraction = override_nodes.get("memory_extraction", wrap_node_with_coordination(memory_extraction_node, "memory_extraction"))
        memory_update = override_nodes.get("memory_update", wrap_node_with_coordination(memory_update_node, "memory_update"))
        route_query_node = override_nodes.get("route_query", wrap_node_with_coordination(route_query, "route_query"))
        prepare_context_node = override_nodes.get("prepare_context", wrap_node_with_coordination(prepare_context, "prepare_context"))
        food_ordering = override_nodes.get("food_ordering", wrap_node_with_coordination(food_ordering_node, "food_ordering"))
        podcast_agent = override_nodes.get("podcast_agent", wrap_node_with_coordination(podcast_agent_node, "podcast_agent"))
        news_agent = override_nodes.get("news_agent", wrap_node_with_coordination(news_agent_node, "news_agent"))
    else:
        memory_retrieval = wrap_node_with_coordination(memory_retrieval_node, "memory_retrieval")
        add_memory_context = wrap_node_with_coordination(add_memory_context_to_prompt, "add_memory_context")
        memory_extraction = wrap_node_with_coordination(memory_extraction_node, "memory_extraction")
        memory_update = wrap_node_with_coordination(memory_update_node, "memory_update")
        route_query_node = wrap_node_with_coordination(route_query, "route_query")
        prepare_context_node = wrap_node_with_coordination(prepare_context, "prepare_context")
        food_ordering = wrap_node_with_coordination(food_ordering_node, "food_ordering")
        podcast_agent = wrap_node_with_coordination(podcast_agent_node, "podcast_agent")
        news_agent = wrap_node_with_coordination(news_agent_node, "news_agent")
    
    # Add memory-related nodes with deduplication
    workflow.add_node("memory_retrieval", memory_retrieval)
    workflow.add_node("add_memory_context", add_memory_context)
    workflow.add_node("memory_extraction", memory_extraction)
    workflow.add_node("memory_update", memory_update)
    
    # Add all the existing nodes to the graph
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("prepare_context", prepare_context_node)
    workflow.add_node("food_ordering", food_ordering)
    workflow.add_node("podcast_agent", podcast_agent)
    workflow.add_node("news_agent", news_agent)
    
    # Define the workflow entry point - starts with memory retrieval for user context
    workflow.set_entry_point("memory_retrieval")
    
    # First, retrieve memories for the current user before anything else
    workflow.add_edge("memory_retrieval", "add_memory_context")
    
    # Add memory context to influence reasoning
    workflow.add_edge("add_memory_context", "route_query")
    
    # Routing from query router node
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state.get("current_task", "general"),  # Use get with default value of "general"
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


# Wrapper for LangSmith Studio compatibility
def create_info_assistant_for_studio():
    """
    Wrapper function for LangSmith Studio compatibility.
    
    This ensures that the same message handling logic used in the command line
    is also applied when running in LangSmith Studio.
    
    Returns:
        A wrapped graph that handles input messages properly
    """
    # Create the base graph
    base_graph = create_info_assistant()
    
    # Create a wrapper function that applies handle_user_input logic
    def studio_compatible_graph(state_or_message):
        # Print diagnostic info
        print(f"[LANGSMITH] Studio input: {state_or_message}")
        
        # Case 1: If we're called with just a string (typical in Studio)
        if isinstance(state_or_message, str):
            # Create a clean state and process using handle_user_input
            print("[LANGSMITH] Received string input, creating new state")
            result = handle_user_input(base_graph, None, state_or_message)
            return result
            
        # Case 2: If we're called with a dict that has 'messages' (Studio chat)
        elif isinstance(state_or_message, dict) and "messages" in state_or_message:
            # Extract last message if it exists
            messages = state_or_message.get("messages", [])
            last_message = None
            
            # Find the last human message
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("type") == "human":
                    last_message = msg.get("content", "")
                    break
                elif hasattr(msg, "__class__") and msg.__class__.__name__ == "HumanMessage":
                    last_message = msg.content
                    break
                    
            if last_message:
                print(f"[LANGSMITH] Extracted last message: {last_message}")
                # Process using existing state and handle_user_input
                result = handle_user_input(base_graph, state_or_message, last_message)
                return result
            else:
                print("[LANGSMITH] No human message found, passing state directly")
                # No human message found, just pass through
                return base_graph.invoke(state_or_message)
                
        # Default: Just invoke the base graph directly
        print("[LANGSMITH] Passing input directly to graph")
        return base_graph.invoke(state_or_message)
        
    # Return the wrapper function as the graph
    return studio_compatible_graph

# For LangSmith compatibility
if __name__ == "__main__":
    # This is what LangSmith will call
    graph = create_info_assistant_for_studio()


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
    # Add diagnostic logging
    print(f"\n[WORKFLOW] handle_user_input called with message: '{user_message}'")
    if state and "messages" in state:
        print(f"[WORKFLOW] Current state has {len(state['messages'])} messages before processing")
        
    # Flag to track if the user message has been added to the messages list
    message_added = False
    
    # Create initial state if none exists
    if state is None:
        state = StateTransitions.create_clean_state()
        print("[WORKFLOW] Created new clean state")
        
    # Check if we need to prompt for a user ID
    # This specifically handles the LangGraph Studio case where no user ID prompt occurs
    if not user_id and not state.get("user_id") and user_message != "__FIRST_MESSAGE__":
        # First message after Studio startup - prompt for user ID
        if user_message.lower().startswith("my id is ") or user_message.lower().startswith("user id:"):
            # Extract user ID from the message
            try:
                extracted_id = user_message.split(":", 1)[1].strip() if ":" in user_message else user_message[9:].strip()
                if extracted_id:
                    user_id = extracted_id
                    messages = state.get("messages", [])
                    messages.append(HumanMessage(content=user_message))
                    message_added = True
                    messages.append(AIMessage(content=f"Thank you! I've set your user ID to: {user_id}. This will be used to store your memories. How can I help you today?"))
                    print(f"[WORKFLOW] User ID set to: {user_id}, returning early with {len(messages)} messages")
                    return {**state, "messages": messages, "user_id": user_id}
            except Exception as e:
                print(f"[WORKFLOW] Error extracting user ID: {e}")
                
        # No valid user ID in message, so prompt for one
        if not user_message.startswith("__"):  # Ignore internal messages
            messages = state.get("messages", [])
            # Only add the user message if it's not a system message
            if not user_message.startswith("__"):
                messages.append(HumanMessage(content=user_message))
                message_added = True
                print(f"[WORKFLOW] Added user message for ID prompt, message_added={message_added}")
            
            id_prompt = "Welcome! To maintain your memories across sessions, I need a user ID. Please provide one by typing 'My ID is: [your-id]' or press Enter for a new random ID."
            messages.append(AIMessage(content=id_prompt))
            print(f"[WORKFLOW] Added ID prompt, returning early with {len(messages)} messages")
            return {**state, "messages": messages}
    
    # If user just pressed Enter for ID, generate a random one
    if user_message.strip() == "" and not user_id and not state.get("user_id"):
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=""))
        message_added = True
        messages.append(AIMessage(content=f"I've generated a new user ID for you: {user_id}. This will be used to store your memories. How can I help you today?"))
        print(f"[WORKFLOW] Generated new user ID: {user_id}, returning early with {len(messages)} messages")
        return {**state, "messages": messages, "user_id": user_id}
        
    # Set user ID if provided for memory linkage
    if user_id:
        state = StateTransitions.identify_user(state, user_id)
        print(f"[WORKFLOW] User ID set in state: {user_id}")
    
    # Check if user input should be handled directly (without LLM)
    direct_response = get_direct_response(user_message)
    if direct_response:
        # Create simple direct response
        messages = state.get("messages", [])
        # Only add the message if it wasn't already added
        if not message_added:
            messages.append(HumanMessage(content=user_message))
            print("[WORKFLOW] Added user message for direct response")
        messages.append(AIMessage(content=direct_response))
        print(f"[WORKFLOW] Direct response created, returning with {len(messages)} messages")
        return {**state, "messages": messages}
    
    # Check for task transitions based on user input
    state = StateTransitions.transition_from_task(state, user_message)
    
    # Add user message to state ONLY if it hasn't been added yet
    messages = state.get("messages", [])
    print(f"[WORKFLOW] Before adding user message, state has {len(messages)} messages")
    if not message_added:
        messages.append(HumanMessage(content=user_message))
        state = {**state, "messages": messages}
        print(f"[WORKFLOW] Added user message, now state has {len(state['messages'])} messages")
    
    # Initialize or update fingerprinting system to prevent duplicates
    from utils.middleware import track_message_fingerprints
    state = track_message_fingerprints(state)
    
    # Add a flag to state to prevent nodes from adding the message again
    state = {**state, "message_processed": True}
    
    # Apply middleware manually before processing
    state = limit_conversation_history(state)
    state = add_metadata(state)
    
    # Process and return the result
    print("[WORKFLOW] Invoking graph with state")
    result = graph.invoke(state)
    
    print(f"[WORKFLOW] Graph returned result with {len(result.get('messages', []))} messages")
    
    # Apply middleware manually after processing
    result = limit_conversation_history(result)
    result = add_metadata(result)
    
    # Verify tool state consistency
    result = verify_tools_state_consistency(result)
    
    # Remove the processing flag if it exists
    if "message_processed" in result:
        result = {k: v for k, v in result.items() if k != "message_processed"}
    
    # ENHANCED DEDUPLICATION: Apply a final, strict deduplication pass
    result = apply_strict_deduplication(result)
    
    # Add final diagnostic count
    print(f"[WORKFLOW] Final result has {len(result.get('messages', []))} messages")
    
    return result

def apply_strict_deduplication(state: InfoAssistantState) -> InfoAssistantState:
    """
    Apply strict deduplication to messages in the state.
    This is more aggressive than the regular deduplication in nodes.
    
    Args:
        state: The current state
        
    Returns:
        State with strictly deduplicated messages
    """
    if "messages" not in state:
        return state
        
    # Track seen message content keyed by ONLY content (ignoring type)
    # This is more strict than the node-level deduplication
    seen_contents = {}
    deduplicated_messages = []
    duplicate_count = 0
    duplicate_summary = {}
    
    for msg in state["messages"]:
        # Extract content consistently regardless of message type
        if isinstance(msg, dict):
            content = msg.get("content", "")
            msg_type = msg.get("type", "unknown")
        else:
            content = getattr(msg, "content", str(msg))
            msg_type = msg.__class__.__name__
            
        # Skip empty messages
        if not content.strip():
            continue
            
        # Check if we've seen identical content before
        if content not in seen_contents:
            seen_contents[content] = True
            deduplicated_messages.append(msg)
        else:
            duplicate_count += 1
            # Track statistics on duplicate types
            if msg_type in duplicate_summary:
                duplicate_summary[msg_type] += 1
            else:
                duplicate_summary[msg_type] = 1
    
    # Only log if we found duplicates
    if duplicate_count > 0:
        # Enhanced logging with duplicate type information
        print(f"[WORKFLOW] Strict deduplication: {len(state['messages'])} messages -> {len(deduplicated_messages)} messages")
        print(f"[WORKFLOW] Removed {duplicate_count} duplicates")
        
        # Log duplicate summary if any found
        if duplicate_summary:
            print(f"[WORKFLOW] Duplicate types: {duplicate_summary}")
    
    return {**state, "messages": deduplicated_messages}

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