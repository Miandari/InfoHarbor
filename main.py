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

def run_info_assistant(query: str, conversation_id: Optional[str] = None):
    """
    Run the information assistant with a query.
    
    Args:
        query: The user's query
        conversation_id: Optional ID for session tracking
        
    Returns:
        The assistant's response
    """
    # Create the info assistant graph
    app = create_info_assistant()
    
    # Create initial state
    messages = [HumanMessage(content=query)]
    
    # Create context with conversation ID if provided
    context = {}
    if conversation_id:
        context["conversation_id"] = conversation_id
    
    # Call the graph
    result = app.invoke({"messages": messages, "context": context})
    
    # Return the last message (assistant's response)
    return result["messages"][-1].content

if __name__ == "__main__":
    import sys
    
    # Check if we're in visualization mode
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        # Visualize the workflow
        try:
            from graphviz import Digraph
            
            # Create the graph (uncompiled to preserve structure)
            workflow = StateGraph(InfoAssistantState)
            workflow.add_node("main_agent", lambda x: x)  # Placeholder function
            workflow.set_entry_point("main_agent")
            workflow.add_edge("main_agent", END)
            
            # Create visualization
            dot = Digraph(comment='Info Assistant Workflow')
            dot.attr(rankdir='LR')  # Left to right layout
            
            # Add nodes
            dot.node("main_agent", "Main Agent\n(All Tools)", shape='box', style='filled', fillcolor='lightblue')
            dot.node("END", "END", shape='doublecircle', style='filled', fillcolor='lightgray')
            
            # Add edge
            dot.edge("main_agent", "END", penwidth='1.5')
            
            # Save the visualization
            dot.render("info_assistant_workflow", format='png', cleanup=True)
            print("Graph visualization saved to: info_assistant_workflow.png")
            
        except ImportError:
            print("Graphviz not installed. Install with: pip install graphviz")
    
    else:
        # Interactive mode
        user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your query: ")
        response = run_info_assistant(user_query)
        print(f"\nAssistant: {response}")