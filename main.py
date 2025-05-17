"""
Main entry point for the Information Assistant application.
"""
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import argparse

# Import the workflow
from graph.workflow import create_info_assistant

# Import the necessary types for visualization
from graph.state import InfoAssistantState  # Import state type from your state.py file
from langgraph.graph import StateGraph, END  # Import StateGraph and END from langgraph

# Import debug utilities
from utils.direct_response import set_debug_mode, debug_log

# Load environment variables
load_dotenv()

def run_info_assistant(query: str, state=None, conversation_id: Optional[str] = None):
    """
    Run the information assistant with a query.
    
    Args:
        query: The user's query
        state: Optional previous state to continue a conversation
        conversation_id: Optional ID for session tracking
        
    Returns:
        Tuple of (response_content, updated_state)
    """
    # Import direct_response utility to use as fallback if needed
    from utils.direct_response import get_direct_answer
    # Import formatting functions
    from utils.formatting import format_news_response, format_podcast_response
    
    debug_log("\n==== RUN_INFO_ASSISTANT CALLED ====")
    debug_log(f"Query: {query}")
    
    # Create the info assistant graph
    app = create_info_assistant()
    
    if state is None:
        debug_log("Creating new state (first turn)")
        # Start a new conversation with initial state
        messages = [HumanMessage(content=query)]
        
        # Create context with conversation ID if provided
        context = {}
        if conversation_id:
            context["conversation_id"] = conversation_id
        
        # Initial state matching InfoAssistantState structure
        initial_state = {
            "messages": messages,
            "podcast_history": [],
            "news_history": [],
            "current_task": None,
            "last_tool_used": None,
            "context": context,
        }
    else:
        # Continue existing conversation
        debug_log(f"Continuing conversation with existing state ({len(state['messages'])} messages)")
        # Make a deep copy of state to avoid modifying the original
        initial_state = {
            "messages": list(state["messages"]) + [HumanMessage(content=query)],
            "podcast_history": state.get("podcast_history", []),
            "news_history": state.get("news_history", []),
            "current_task": state.get("current_task"),
            "last_tool_used": state.get("last_tool_used"),
            "context": state.get("context", {}),
            "tool_results": state.get("tool_results", {})
        }
    
    # Call the graph
    debug_log("Invoking LangGraph workflow")
    result = app.invoke(initial_state)
    
    # CRITICAL: Check for tool results first - this is the most reliable approach
    tool_results = result.get("tool_results", {})
    current_task = result.get("current_task")
    debug_log(f"Current task after processing: {current_task}")
    debug_log(f"Tool results present: {bool(tool_results)}")
    
    # DIRECT TOOL RESULT HANDLING APPROACH:
    # Instead of trying to find messages in the state, we'll directly use
    # the tool results and format them appropriately
    
    if current_task == "news" and tool_results and tool_results.get("type") == "news":
        debug_log("NEWS TASK DETECTED - Using news formatting")
        news_data = tool_results.get("data", {})
        topic = tool_results.get("topic", "")
        days_back = tool_results.get("days_back", 7)
        error = tool_results.get("error")
        
        if error:
            response = f"I encountered an issue while searching for news about '{topic}': {error}. Would you like to try a different search?"
        else:
            # Format news response
            response = format_news_response(news_data, topic, days_back)
        
        debug_log(f"NEWS RESPONSE: {response[:100]}...")
        
    elif current_task == "podcast" and tool_results and tool_results.get("type") == "podcast":
        debug_log("PODCAST TASK DETECTED - Using podcast formatting")
        podcast_data = tool_results.get("data", [])
        response = format_podcast_response({"recommendations": podcast_data})
        debug_log(f"PODCAST RESPONSE: {response[:100]}...")
        
    else:
        # For all other cases, check for AI messages or use direct response
        ai_messages = [msg for msg in result.get("messages", []) if isinstance(msg, AIMessage)]
        
        if ai_messages:
            debug_log(f"Using last AI message from {len(ai_messages)} messages")
            response = ai_messages[-1].content
        else:
            debug_log("No AI messages or tool results, using direct response")
            response = get_direct_answer(query)
            # Add this response to the result state
            result["messages"].append(AIMessage(content=response))
    
    debug_log(f"FINAL RESPONSE: {response[:100]}...")
    debug_log("==== RUN_INFO_ASSISTANT COMPLETE ====\n")
    
    # Return the response and updated state
    return response, result

def interactive_chat_session():
    """Run an interactive chat session with the agent."""
    print("Welcome to the Information Assistant!")
    print("You can chat with the assistant about podcasts, news, or ask general questions.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialize conversation state
    state = None
    
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check if user wants to exit
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("\nAssistant: Goodbye! Have a great day!")
            break
        
        try:
            # Get response and updated state
            response, state = run_info_assistant(user_query, state)
            
            # Debug logging
            debug_log(f"MAIN.PY - User query: {user_query}")
            debug_log(f"MAIN.PY - Response received: {response}")
            
            # Print the response
            print(f"\nAssistant: {response}")
            
            # Extra debug - check if response equals user query
            if response.lower().strip() == user_query.lower().strip():
                debug_log("WARNING: RESPONSE EQUALS USER QUERY!")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's start a new conversation.")
            state = None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Information Assistant")
    parser.add_argument("--visualize", action="store_true", help="Generate a visualization of the workflow")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("query", nargs="*", help="Query to ask the assistant (in non-interactive mode)")
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    args = parse_args()
    
    # Set debug mode if requested
    if args.debug:
        set_debug_mode(True)
        print("Debug mode enabled. Logging to debug_log.txt")
    
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
        # Single query mode
        user_query = " ".join(args.query)
        response, _ = run_info_assistant(user_query)
        print(f"\nAssistant: {response}")
    else:
        # Interactive chat mode
        interactive_chat_session()