"""
State type definitions for the information assistant LangGraph.
"""

from typing import List, Dict, Any, Annotated, TypedDict, Sequence, Union, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class InfoAssistantState(TypedDict):
    """State for the information assistant workflow."""
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage, ToolMessage]], "The messages in the conversation"]
    podcast_history: Annotated[List[Dict], "History of podcast recommendations"]
    news_history: Annotated[List[Dict], "History of news searches"]
    food_order_history: Annotated[List[Dict], "History of food orders"]
    food_order_state: Annotated[Optional[Literal["collecting_details", "completed", "error"]], "Current state of food ordering process"]
    current_task: Annotated[Optional[Literal["podcast", "news", "food_order", "general"]], "Current task type being handled"]
    last_tool_used: Annotated[Optional[str], "The last tool that was used"]
    context: Dict[str, Any]
    tool_results: Annotated[Optional[Dict], "Results from the last tool used"]