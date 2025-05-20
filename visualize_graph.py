"""
Direct workflow visualization script for memory-enhanced ReAct-based workflow
"""
import os
import sys
from graphviz import Digraph

# Import directly from the workflow module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph.workflow import create_info_assistant

print("Creating the memory-enhanced info assistant workflow...")
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

# Create visualization function for the memory-enhanced workflow
def visualize_workflow(output_file="info_assistant_workflow"):
    """Create a visualization for the memory-enhanced workflow with reasoning, action, and observation"""
    dot = Digraph(comment='Memory-Enhanced Info Assistant Workflow')
    dot.attr(rankdir='TB')  # Top to bottom layout
    
    # Define node clusters
    with dot.subgraph(name="cluster_memory") as memory:
        memory.attr(label="Memory System", style="filled", color="lightblue")
        memory.node("memory_retrieval", "Memory Retrieval\n(Load User Context)", shape='box', style='filled', fillcolor='lightcyan')
        memory.node("add_memory_context", "Add Memory Context\n(To Reasoning)", shape='box', style='filled', fillcolor='lightcyan')
        memory.node("memory_extraction", "Memory Extraction\n(Identify New Facts)", shape='box', style='filled', fillcolor='lightcyan')
        memory.node("memory_update", "Memory Update\n(Store New Facts)", shape='box', style='filled', fillcolor='lightcyan')
    
    with dot.subgraph(name="cluster_core") as core:
        core.attr(label="Core Processing", style="filled", color="lightgreen")
        core.node("route_query", "Route Query\n(Intent Detection)", shape='box', style='filled', fillcolor='lightgreen')
        core.node("prepare_context", "General Processing\n(Default Handler)", shape='box', style='filled', fillcolor='lightgreen')
    
    with dot.subgraph(name="cluster_tools") as tools:
        tools.attr(label="Specialized Tools", style="filled", color="lightyellow")
        tools.node("podcast_agent", "Podcast Agent", shape='box', style='filled', fillcolor='lightsalmon')
        tools.node("news_agent", "News Agent", shape='box', style='filled', fillcolor='lightyellow')
        tools.node("food_ordering", "Food Ordering", shape='box', style='filled', fillcolor='lightcoral')
    
    # Define start and end nodes
    dot.node("_start_", shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.node("END", shape='doublecircle', style='filled', fillcolor='lightgray')
    
    # Add the edges for memory flow
    dot.edge("_start_", "memory_retrieval", penwidth='1.5')
    dot.edge("memory_retrieval", "add_memory_context", penwidth='1.5')
    dot.edge("add_memory_context", "route_query", penwidth='1.5')
    
    # Routing edges from intent detection
    dot.edge("route_query", "food_ordering", label="food order", penwidth='1.5')
    dot.edge("route_query", "podcast_agent", label="podcast", penwidth='1.5')
    dot.edge("route_query", "news_agent", label="news", penwidth='1.5')
    dot.edge("route_query", "prepare_context", label="general/none", penwidth='1.5')
    
    # All tool endpoints go to memory extraction
    dot.edge("prepare_context", "memory_extraction", penwidth='1.5')
    dot.edge("food_ordering", "memory_extraction", penwidth='1.5')
    dot.edge("podcast_agent", "memory_extraction", penwidth='1.5')
    dot.edge("news_agent", "memory_extraction", penwidth='1.5')
    
    # Memory processing flow - extract then update
    dot.edge("memory_extraction", "memory_update", penwidth='1.5')
    dot.edge("memory_update", "END", penwidth='1.5')
    
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