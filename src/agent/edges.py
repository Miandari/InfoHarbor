"""
Edge logic and conditions for Elder Care Assistant
"""
from typing import Literal
from src.agent.state import AgentState


def should_use_tool(state: AgentState) -> Literal["use_tool", "generate_response"]:
    """Determine if tools should be used based on user input"""
    user_input = state.user_input.lower()
    
    # Check for tool-triggering keywords
    tool_keywords = [
        "news", "current events", "podcast", "listen", "audio",
        "health", "medication", "doctor", "weather", "search"
    ]
    
    if any(keyword in user_input for keyword in tool_keywords):
        return "use_tool"
    
    return "generate_response"


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if processing should continue"""
    # For now, always end after tool execution
    # In the future, this could support multi-step tool usage
    return "end"