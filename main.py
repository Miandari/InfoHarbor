"""
Main entry point for the Information Assistant application.
"""
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import argparse

# Import the workflow
from graph.workflow import create_info_assistant, ConfigSchema

# Import the necessary types for visualization
from graph.state import InfoAssistantState  # Import state type from your state.py file
from langgraph.graph import StateGraph, END  # Import StateGraph and END from langgraph

# Import debug utilities
from utils.direct_response import set_debug_mode, debug_log, set_verbose_mode

# Load environment variables
load_dotenv()

def run_info_assistant(query: str, state=None, conversation_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """
    Run the information assistant with a query.
    
    Args:
        query: The user's query
        state: Optional previous state to continue a conversation
        conversation_id: Optional ID for session tracking
        config: Optional configuration to override defaults
        
    Returns:
        Tuple of (response_content, updated_state)
    """
    # Import direct_response utility to use as fallback if needed
    from utils.direct_response import get_direct_answer
    # Import formatting functions
    from utils.formatting import format_news_response, format_podcast_response
    
    debug_log("\n==== RUN_INFO_ASSISTANT CALLED ====")
    debug_log(f"Query: {query}")
    
    # Check if this is a food ordering continuation
    if state and state.get("current_task") == "food_order" and state.get("food_order_state") == "collecting_details":
        debug_log("Food order continuation detected, using food ordering tool")
        from tools.food_tools import process_food_order
        
        try:
            # Process the food order using invoke() instead of direct calling
            result = process_food_order.invoke({"order_text": query})
            
            # Format a nice response with order confirmation
            response = "Thank you! I've sent your food order via Telegram. Here's what I sent:\n\n" + \
                      f"{query}\n\n" + \
                      "Your order has been submitted. Is there anything else you need help with?"
            
            # Update the state
            food_order_history = state.get("food_order_history", [])
            food_order_history.append({"order_text": query})
            
            updated_state = {
                **state,
                "messages": state["messages"] + [AIMessage(content=response)],
                "current_task": "food_order",
                "last_tool_used": "food_order_tool",
                "food_order_state": "completed",
                "food_order_history": food_order_history,
                "tool_results": {
                    "type": "food_order",
                    "data": result,
                    "pending": False
                }
            }
            
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
            "food_order_history": [],
            "food_order_state": None,
            "current_task": None,
            "last_tool_used": None,
            "context": context,
            "tool_results": {}
        }
        
        # Check if this is a food ordering intent
        query_lower = query.lower()
        food_order_keywords = ["order food", "order a pizza", "food delivery", "order pizza", "get food", 
                          "place an order", "hungry", "deliver food", "takeout", "food order",
                          "pizza", "want to order", "want pizza"]
                          
        is_food_order = any(keyword in query_lower for keyword in food_order_keywords) or \
                      ("want" in query_lower and "pizza" in query_lower)
                      
        if is_food_order:
            debug_log("Food ordering intent detected in first message")
            response = "I'd be happy to help you order food! Please provide your complete order details including:\n\n" + \
                  "1. Restaurant name\n" + \
                  "2. Items you'd like to order (including quantities)\n" + \
                  "3. Delivery or pickup preference\n" + \
                  "4. Delivery address (if applicable)\n" + \
                  "5. Any special instructions"
                  
            updated_state = {
                **initial_state,
                "messages": messages + [AIMessage(content=response)],
                "current_task": "food_order", 
                "food_order_state": "collecting_details"
            }
            
            debug_log("Food ordering prompt sent directly")
            return response, updated_state
            
    else:
        # Continue existing conversation
        messages_count = len(state['messages']) if 'messages' in state else 0
        debug_log(f"Continuing conversation with existing state. Message count before: {messages_count}")
        
        # Check if the previous exchange was about food ordering (prompt sent, now getting details)
        if state.get("current_task") == "food_order" and state.get("food_order_state") == "collecting_details":
            debug_log("Processing food order details")
            from tools.food_tools import process_food_order
            
            try:
                # Process the food order using invoke() instead of direct calling
                result = process_food_order.invoke({"order_text": query})
                
                # Format a nice response with order confirmation
                response = "Thank you! I've sent your food order via Telegram. Here's what I sent:\n\n" + \
                          f"{query}\n\n" + \
                          "Your order has been submitted. Is there anything else you need help with?"
                
                # Update the state
                food_order_history = state.get("food_order_history", [])
                food_order_history.append({"order_text": query})
                
                updated_state = {
                    **state,
                    "messages": state["messages"] + [HumanMessage(content=query), AIMessage(content=response)],
                    "current_task": "food_order",
                    "last_tool_used": "food_order_tool",
                    "food_order_state": "completed",
                    "food_order_history": food_order_history,
                    "tool_results": {
                        "type": "food_order",
                        "data": result,
                        "pending": False
                    }
                }
                
                debug_log("Food order processed successfully")
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
            "context": state.get("context", {}),
            "tool_results": state.get("tool_results", {})
        }
    
    # Call the graph with the config parameter - critical for LangGraph Cloud/Studio
    debug_log("Invoking LangGraph workflow")
    result = app.invoke(initial_state, config=config)
    
    debug_log(f"LangGraph workflow returned. Messages in result: {len(result.get('messages', []))}")
    
    # Track how many messages were added during processing
    original_msg_count = len(initial_state["messages"])
    result_msg_count = len(result.get("messages", []))
    added_msg_count = result_msg_count - original_msg_count
    debug_log(f"Messages added during processing: {added_msg_count}")
    
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

def interactive_chat_session():
    """Run an interactive chat session with the agent."""
    print("\nWelcome to the Information Assistant!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to start a new conversation.\n")
    
    state = None
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == "clear":
                state = None
                print("\nConversation cleared. Starting fresh!")
                continue
                
            if not user_input:
                print("Please enter a question or command.")
                continue
                
            try:
                debug_log("MAIN.PY - User query: " + user_input)
                response, state = run_info_assistant(user_input, state)
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
    parser = argparse.ArgumentParser(description="Information Assistant with podcast and news capabilities")
    parser.add_argument("query", nargs="*", help="The query to ask the assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode with console debug messages")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--visualize", action="store_true", help="Generate a visualization of the workflow")
    
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
        # Single query mode
        user_query = " ".join(args.query)
        response, _ = run_info_assistant(user_query)
        print(f"\nAssistant: {response}")
    else:
        # Interactive chat mode
        interactive_chat_session()