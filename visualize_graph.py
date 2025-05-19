"""
Direct workflow visualization script that avoids circular imports
"""
import os
import sys
from graphviz import Digraph

# Import directly from the workflow module
# This bypasses the circular dependency
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.workflow import create_info_assistant

print("Creating the info assistant workflow...")
# Create a new instance of the workflow (uncompiled)
try:
    workflow = create_info_assistant()
    print(f"Successfully retrieved workflow")
except Exception as e:
    print(f"Error creating workflow: {e}")
    sys.exit(1)

# Print the basic info about the workflow
print(f"Workflow type: {type(workflow).__name__}")
print(f"Nodes: {list(workflow.nodes) if hasattr(workflow, 'nodes') else 'Not accessible'}")

# Create visualization function for the cyclic workflow with main agent control
def visualize_workflow(output_file="info_assistant_workflow"):
    """Create a visualization for the main agent with tool nodes that report back"""
    dot = Digraph(comment='Info Assistant Workflow')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Define the nodes
    dot.node("_start_", shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.node("main_agent", "Main Agent\n(Conversation Controller)", shape='box', style='filled', fillcolor='lightblue')
    dot.node("podcast_tools", "Podcast Tools\n(Specialized Agent)", shape='box', style='filled', fillcolor='lightpink')
    dot.node("news_tools", "News Tools\n(Specialized Agent)", shape='box', style='filled', fillcolor='lightyellow')
    dot.node("food_order", "Food Order Tools\n(Order Processing)", shape='box', style='filled', fillcolor='lightcoral')
    dot.node("respond", "Response Handler\n(Formats Responses)", shape='box', style='filled', fillcolor='lightgreen')
    dot.node("END", shape='doublecircle', style='filled', fillcolor='lightgray')
    
    # Add the edges
    dot.edge("_start_", "main_agent", penwidth='1.5')
    dot.edge("main_agent", "podcast_tools", label="podcast intent", penwidth='1.5')
    dot.edge("main_agent", "news_tools", label="news intent", penwidth='1.5')
    dot.edge("main_agent", "food_order", label="food order intent", penwidth='1.5')
    dot.edge("main_agent", "respond", label="direct response", penwidth='1.5')
    dot.edge("main_agent", "END", label="end", penwidth='0.8', style="dashed")
    
    # Tools report back to respond handler
    dot.edge("podcast_tools", "respond", penwidth='1.5')
    dot.edge("news_tools", "respond", penwidth='1.5')
    dot.edge("food_order", "respond", penwidth='1.5')
    
    # Respond handler goes back to main agent for next turn
    dot.edge("respond", "main_agent", label="next turn", penwidth='1.5')
    
    # Save the visualization
    try:
        dot.render(output_file, format='png', cleanup=True)
        print(f"Graph visualization saved to: {output_file}.png")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        # Try a simpler approach
        try:
            dot.save(f"{output_file}.dot")
            print(f"Graph dot file saved to: {output_file}.dot")
            print("You can render this with: dot -Tpng -o workflow.png info_assistant_workflow.dot")
        except Exception as e2:
            print(f"Could not save dot file either: {e2}")
    
    return dot

# Generate the visualization
print("Generating visualization...")
graph_viz = visualize_workflow()

print("Visualization process complete!")