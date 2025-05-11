"""
Utility functions for formatting responses in the information assistant.
"""

from typing import List, Dict, Any, Sequence, Union
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def format_podcast_response(podcast_data: Dict[str, Any]) -> str:
    """Format podcast recommendation results for user-friendly display."""
    recommendations = podcast_data.get("recommendations", [])
    if not recommendations:
        return "I couldn't find any podcasts matching your criteria. Would you like to try a different search?"
    
    response = f"Here are some podcast recommendations for you:\n\n"
    
    for i, podcast in enumerate(recommendations[:5], 1):
        name = podcast.get("name", "Unknown")
        description = podcast.get("description", "No description available")
        # Truncate long descriptions
        if len(description) > 150:
            description = description[:147] + "..."
            
        publisher = podcast.get("publisher", "")
        listen_notes_url = podcast.get("listen_notes_url", "")
        
        response += f"**{i}. {name}**\n"
        if publisher:
            response += f"Publisher: {publisher}\n"
        response += f"{description}\n"
        
        # Add episode information if available
        episodes = podcast.get("episodes", [])
        if episodes:
            response += "\nRecent episodes:\n"
            for j, episode in enumerate(episodes[:2], 1):
                episode_title = episode.get("title", "Untitled episode")
                response += f"- {episode_title}\n"
        
        if listen_notes_url:
            response += f"[Listen Notes Link]({listen_notes_url})\n"
            
        response += "\n"
    
    response += "Would you like more details about any of these podcasts or recommendations on a different topic?"
    return response

def format_news_response(news_data: Dict[str, Any], topic: str, days_back: int) -> str:
    """Format news search results for user-friendly display."""
    articles = news_data.get("articles", [])
    summary = news_data.get("summary", "")
    
    if not articles:
        return f"I couldn't find any recent news about '{topic}' in the past {days_back} days. Would you like me to search for a different topic or time period?"
    
    response = f"Here's what I found about '{topic}' from the past {days_back} days:\n\n"
    
    if summary and summary != "No summary available":
        response += f"**Summary**: {summary}\n\n"
        
    response += "**Top Articles:**\n"
    for i, article in enumerate(articles[:5], 1):
        title = article.get("title", "Untitled")
        source = article.get("source", "Unknown source")
        date = article.get("date", "Unknown date")
        url = article.get("url", "")
        
        response += f"{i}. **{title}**\n"
        response += f"   Source: {source} | Date: {date}\n"
        if url:
            response += f"   [Read More]({url})\n"
        response += "\n"
        
    response += "Would you like more details on any of these stories or news on a related topic?"
    return response

def format_response(messages: Sequence[Union[HumanMessage, AIMessage, ToolMessage]]) -> List[Union[HumanMessage, AIMessage, ToolMessage]]:
    """
    Format the response messages from the agent.
    This is a more general version that handles all response types.
    
    Args:
        messages: The messages from the agent
        
    Returns:
        The formatted messages
    """
    # Simply return the messages as-is for now
    # You can add more sophisticated formatting logic here if needed
    return messages

