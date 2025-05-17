"""
Main entry point for the Information Assistant application.
"""
import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import the workflow
from graph.workflow import create_info_assistant

# Import the necessary types for visualization
from graph.state import InfoAssistantState  # Import state type from your state.py file
from langgraph.graph import StateGraph, END  # Import StateGraph and END from langgraph

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
    # Create the info assistant graph
    app = create_info_assistant()
    
    if state is None:
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
        state["messages"].append(HumanMessage(content=query))
        initial_state = state
    
    # Call the graph
    result = app.invoke(initial_state)
    
    # Return the last message and updated state
    return result["messages"][-1].content, result

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
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's start a new conversation.")
            state = None

if __name__ == "__main__":
    import sys
    
    # Check if we're in visualization mode
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        # Import and use the visualization function from visualize_graph.py
        try:
            from visualize_graph import visualize_workflow
            
            # Generate the visualization
            visualize_workflow()
            
        except ImportError:
            print("Graphviz not installed. Install with: pip install graphviz")
    
    # Check if we're in single query mode or interactive mode
    elif len(sys.argv) > 1 and sys.argv[1] != "--interactive":
        # Single query mode
        user_query = " ".join(sys.argv[1:])
        response, _ = run_info_assistant(user_query)
        print(f"\nAssistant: {response}")
    else:
        # Interactive chat mode
        interactive_chat_session()