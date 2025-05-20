"""
Main entry point for the Information Assistant application.
"""
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import argparse
import uuid

# Import the workflow
from graph.workflow import create_info_assistant, ConfigSchema, handle_user_input

# Import the necessary types for visualization
from graph.state import InfoAssistantState  # Import state type from your state.py file
from langgraph.graph import StateGraph, END  # Import StateGraph and END from langgraph

# Import the new centralized state transitions module
from graph.transitions import StateTransitions

# Import the memory manager for direct access when needed
from memory.memory_manager import MemoryManager

# Import debug utilities
from utils.direct_response import set_debug_mode, debug_log, set_verbose_mode

# Initialize memory-related constants
MEMORY_STORAGE_PATH = os.getenv("MEMORY_STORAGE_PATH", "./memory_storage")
USER_SESSION_FILE = os.getenv("USER_SESSION_FILE", "./user_sessions.json")

# Load environment variables
load_dotenv()

# Initialize memory manager as needed
_memory_manager = None

def get_memory_manager(debug=False) -> MemoryManager:
    """Get or initialize the memory manager singleton."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(storage_path=MEMORY_STORAGE_PATH, debug=debug)
    return _memory_manager

def run_info_assistant(query: str, state=None, conversation_id: Optional[str] = None, 
                      user_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None, 
                      debug: bool = False):
    """
    Run the information assistant with a query.
    
    Args:
        query: The user's query
        state: Optional previous state to continue a conversation
        conversation_id: Optional ID for session tracking
        user_id: Optional user identifier for memory persistence
        config: Optional configuration to override defaults
        debug: Optional flag to enable debug mode
        
    Returns:
        Tuple of (response_content, updated_state)
    """
    # Import direct_response utility to use as fallback if needed
    from utils.direct_response import get_direct_answer
    # Import formatting functions
    from utils.formatting import format_news_response, format_podcast_response
    
    debug_log("\n==== RUN_INFO_ASSISTANT CALLED ====")
    debug_log(f"Query: {query}")
    debug_log(f"User ID: {user_id}")
    
    # Create or get unique user ID if not provided
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        debug_log(f"Generated new user ID: {user_id}")
    
    # Initialize memory manager with debug flag
    memory_manager = get_memory_manager(debug=debug)
    
    # Check if this is a food ordering continuation
    if state and state.get("current_task") == "food_order" and state.get("food_order_state") == "collecting_details":
        debug_log("Food order continuation detected, using food ordering tool")
        from tools.food_tools import process_food_order
        
        try:
            # Process the food order using invoke() instead of direct calling
            result = process_food_order.invoke({"order_text": query})
            
            # Use the centralized state transition function for food order completion
            updated_state = StateTransitions.handle_food_order_completion(state, query, result)
            
            # Set or update user ID for memory
            updated_state = StateTransitions.identify_user(updated_state, user_id)
            
            # Extract the response from the updated state
            ai_messages = [msg for msg in updated_state["messages"] if isinstance(msg, AIMessage)]
            response = ai_messages[-1].content if ai_messages else "Your food order has been processed."
            
            debug_log("Food order processed directly")
            return response, updated_state
            
        except Exception as e:
            debug_log(f"Error processing food order: {str(e)}")
    
    # Create the info assistant graph
    app = create_info_assistant()
    
    # Set configuration for the workflow - critical for LangGraph Cloud/Studio compatibility
    if not config:
        from config import DEFAULT_MODEL
        config = {
            "configurable": {  # Wrap in configurable for Cloud compatibility
                "model_name": DEFAULT_MODEL,
                "system_prompt": "You are a helpful assistant specialized in providing information about podcasts and news.",
            }
        }
    
    if state is None:
        debug_log("Creating new state (first turn)")
        # Use the centralized state creation function to get a fresh state
        initial_state = StateTransitions.create_clean_state()
        
        # Set the user ID for memory
        initial_state["user_id"] = user_id
        
        # Add the first message
        initial_state["messages"] = [HumanMessage(content=query)]
        
        # Add conversation ID if provided
        if conversation_id:
            initial_state["context"] = {"conversation_id": conversation_id}
            
        # Check if this is a food ordering intent
        intent = StateTransitions.determine_intent(query)
        
        if intent == "food_order":
            debug_log("Food ordering intent detected in first message")
            response = "I'd be happy to help you order food! Please provide your complete order details including:\n\n" + \
                  "1. Restaurant name\n" + \
                  "2. Items you'd like to order (including quantities)\n" + \
                  "3. Delivery or pickup preference\n" + \
                  "4. Delivery address (if applicable)\n" + \
                  "5. Any special instructions"
                  
            # Use state transitions to update state
            initial_state = {
                **initial_state,
                "messages": initial_state["messages"] + [AIMessage(content=response)],
                "current_task": "food_order", 
                "food_order_state": "collecting_details"
            }
            
            debug_log("Food ordering prompt sent directly")
            return response, initial_state
    else:
        # Continue existing conversation
        messages_count = len(state['messages']) if 'messages' in state else 0
        debug_log(f"Continuing conversation with existing state. Message count before: {messages_count}")
        
        # Ensure the state has a user ID
        if not state.get("user_id"):
            state["user_id"] = user_id
            
        # Use the centralized state transitions to handle task transitions
        state = StateTransitions.transition_from_task(state, query)
        
        # Check if the previous exchange was about food ordering (prompt sent, now getting details)
        if state.get("current_task") == "food_order" and state.get("food_order_state") == "collecting_details":
            debug_log("Processing food order details")
            from tools.food_tools import process_food_order
            
            try:
                # Process the food order using invoke() instead of direct calling
                result = process_food_order.invoke({"order_text": query})
                
                # Use the centralized state transition function for food order completion
                updated_state = StateTransitions.handle_food_order_completion(state, query, result)
                
                # Extract the response from the updated state
                ai_messages = [msg for msg in updated_state["messages"] if isinstance(msg, AIMessage)]
                response = ai_messages[-1].content if ai_messages else "Your food order has been processed."
                
                debug_log("Food order processed directly")
                return response, updated_state
                
            except Exception as e:
                debug_log(f"Error processing food order: {str(e)}")
                import traceback
                debug_log(traceback.format_exc())
                
                # Error handling
                response = f"I encountered an issue while processing your food order: {str(e)}. Would you like to try again?"
                updated_state = {
                    **state,
                    "messages": state["messages"] + [HumanMessage(content=query), AIMessage(content=response)],
                    "current_task": "food_order",
                    "food_order_state": "error"
                }
                return response, updated_state
        
        # Make a deep copy of state to avoid modifying the original
        initial_state = {
            "messages": list(state["messages"]) + [HumanMessage(content=query)],
            "podcast_history": state.get("podcast_history", []),
            "news_history": state.get("news_history", []),
            "food_order_history": state.get("food_order_history", []),
            "food_order_state": state.get("food_order_state", None),
            "current_task": state.get("current_task"),
            "last_tool_used": state.get("last_tool_used"),
            "user_id": state.get("user_id", user_id),
            "user_memory": state.get("user_memory", {}),
            "memory_updates": state.get("memory_updates", []),
            "context": state.get("context", {}),
            "tool_results": state.get("tool_results", {}),
            "reasoning": state.get("reasoning", []),
            "next_actions": state.get("next_actions", []),
            "working_memory": state.get("working_memory", {}),
            "pending_tools": state.get("pending_tools", []),
            "reflection": state.get("reflection", None),
            "state_version": state.get("state_version", 0) + 1  # Increment version number
        }
    
    # Process the query using the workflow, with user ID for memory
    debug_log("Invoking LangGraph workflow with memory support")
    result = handle_user_input(app, initial_state, query, user_id)
    
    debug_log(f"LangGraph workflow returned. Messages in result: {len(result.get('messages', []))}")
    
    # Track how many messages were added during processing
    original_msg_count = len(initial_state["messages"])
    result_msg_count = len(result.get("messages", []))
    added_msg_count = result_msg_count - original_msg_count
    debug_log(f"Messages added during processing: {added_msg_count}")
    
    # Log ReAct-related information for debugging
    reasoning_count = len(result.get("reasoning", []))
    debug_log(f"Reasoning steps recorded: {reasoning_count}")
    if result.get("reflection"):
        debug_log(f"Reflection: {result.get('reflection')[:100]}...")
    
    # Check for newly added AI messages
    ai_messages = [msg for msg in result.get("messages", [])[-added_msg_count:] 
                  if isinstance(msg, AIMessage)]
    
    debug_log(f"Found {len(ai_messages)} new AI messages")
    
    # If we have new AI messages, use the last one as our response
    if ai_messages:
        debug_log(f"Using last AI message: {ai_messages[-1].content[:50]}...")
        response = ai_messages[-1].content
    else:
        # No new AI messages found, check if tool_results are present
        debug_log("No new AI messages found, using direct_response utility as fallback")
        response = get_direct_answer(query)
        # Add this response to the result state
        if "messages" in result:
            result["messages"].append(AIMessage(content=response))
            
        # Check if this is a food ordering intent detected by direct_response
        if "order food" in response.lower() or "restaurant name" in response.lower():
            debug_log("Food ordering prompt detected in direct_response output")
            # Update the state to include food ordering state
            result["current_task"] = "food_order"
            result["food_order_state"] = "collecting_details"
    
    debug_log(f"FINAL RESPONSE: {response[:100]}...")
    debug_log("==== RUN_INFO_ASSISTANT COMPLETE ====\n")
    
    # Return the response and updated state
    return response, result

# User session storage for interactive mode
user_sessions = {}

def get_or_create_user_session(user_id=None):
    """Get or create a user session."""
    import json
    
    global user_sessions
    
    # Generate random user ID if none provided
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    # Check if we have this user in memory
    if user_id in user_sessions:
        return user_id, user_sessions[user_id]
    
    # Try to load from file
    if os.path.exists(USER_SESSION_FILE):
        try:
            with open(USER_SESSION_FILE, 'r') as f:
                stored_sessions = json.load(f)
                user_sessions.update(stored_sessions)
                
                if user_id in user_sessions:
                    return user_id, user_sessions[user_id]
        except Exception as e:
            debug_log(f"Error loading user sessions: {e}")
    
    # Create new session
    user_sessions[user_id] = None  # No state yet
    
    # Try to save to file
    try:
        with open(USER_SESSION_FILE, 'w') as f:
            json.dump(user_sessions, f)
    except Exception as e:
        debug_log(f"Error saving user sessions: {e}")
    
    return user_id, None

def save_user_session(user_id, state):
    """Save user session state."""
    import json
    
    global user_sessions
    
    # Update in-memory
    user_sessions[user_id] = state
    
    # Try to save to file
    try:
        with open(USER_SESSION_FILE, 'w') as f:
            # We can't directly serialize the state with JSON
            # Just save the user IDs - actual state is kept in memory
            sessions_to_save = {uid: True for uid in user_sessions.keys()}
            json.dump(sessions_to_save, f)
    except Exception as e:
        debug_log(f"Error saving user sessions: {e}")

def interactive_chat_session():
    """Run an interactive chat session with the agent."""
    print("\nWelcome to the Information Assistant with Memory!")
    print("This assistant remembers your preferences and important facts across conversations.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to start a new conversation but keep your memory.")
    print("Type 'forget' to clear all your memory data.\n")
    
    # Get or create user session
    user_id = input("Enter your user ID (or press Enter for a new ID): ").strip()
    user_id, state = get_or_create_user_session(user_id if user_id else None)
    
    print(f"\nYour user ID is: {user_id}")
    print("You can use this ID in future sessions to access your memory.")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == "clear":
                # Clear conversation but keep user ID
                state = None
                print("\nConversation cleared. Starting fresh, but I'll remember you!")
                continue
                
            if user_input.lower() == "forget":
                # Clear memory and conversation
                memory_manager = get_memory_manager()
                try:
                    # Get user collection and delete documents
                    collection = memory_manager.memory_store.get_collection(user_id)
                    collection.delete(delete_all=True)
                    print("\nI've cleared all your memory data. Starting with a clean slate!")
                except Exception as e:
                    print(f"\nI had trouble clearing your memory: {str(e)}")
                
                # Clear conversation state
                state = None
                continue
                
            if not user_input:
                print("Please enter a question or command.")
                continue
                
            try:
                debug_log("MAIN.PY - User query: " + user_input)
                response, state = run_info_assistant(user_input, state, user_id=user_id)
                
                # Save user session after each interaction
                save_user_session(user_id, state)
                
                debug_log("MAIN.PY - Response received: " + response[:100])
                print(f"\nAssistant: {response}")
                
                # Quick sanity check - warn if response equals query (feedback loop)
                if response == user_input:
                    debug_log("WARNING: RESPONSE EQUALS USER QUERY!")
                    
            except Exception as e:
                import traceback
                print(f"\nSorry, I encountered an error: {str(e)}")
                debug_log(f"ERROR in interactive session: {str(e)}")
                debug_log(traceback.format_exc())
                
    except KeyboardInterrupt:
        print("\n\nSession ended by user. Goodbye!")
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Information Assistant with memory, podcast and news capabilities")
    parser.add_argument("query", nargs="*", help="The query to ask the assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode with console debug messages")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--visualize", action="store_true", help="Generate a visualization of the workflow")
    parser.add_argument("--user", type=str, help="User ID for memory retrieval")
    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    args = parse_args()
    
    # Set debug and verbose modes if requested
    if args.debug:
        set_debug_mode(True)
        print("Debug mode enabled. Logging to debug_log.txt")
    
    # Set verbose mode separately from debug mode
    set_verbose_mode(args.verbose)
    if args.verbose:
        print("Verbose mode enabled. Debug messages will be displayed in the console.")
    
    # Check if we're in visualization mode
    if args.visualize:
        # Import and use the visualization function from visualize_graph.py
        try:
            from visualize_graph import visualize_workflow
            
            # Generate the visualization
            visualize_workflow()
            
        except ImportError:
            print("Graphviz not installed. Install with: pip install graphviz")
    
    # Check if we're in single query mode or interactive mode
    elif args.query and not args.interactive:
        # Single query mode with possible user ID
        user_query = " ".join(args.query)
        user_id = args.user if args.user else None
        
        # Get or create user session if user ID provided
        if user_id:
            user_id, state = get_or_create_user_session(user_id)
        else:
            # Generate new user ID for single query
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            state = None
            
        response, updated_state = run_info_assistant(user_query, state, user_id=user_id)
        
        # Save the updated state
        if args.user:  # Only save if user explicitly provided a user ID
            save_user_session(user_id, updated_state)
            
        print(f"\nAssistant: {response}")
        if not args.user:
            print(f"\nNote: To continue this conversation with memory, use --user {user_id}")
    else:
        # Interactive chat mode
        interactive_chat_session()