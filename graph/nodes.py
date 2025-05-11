"""
LangGraph processing nodes for the information assistant.
"""

from typing import Dict, Any, List, Sequence, Union, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config import DEFAULT_MODEL, TEMPERATURE
from graph.state import InfoAssistantState
from tools.podcast_tools import (
    get_podcast_recommendations, 
    get_podcast_details,
    get_similar_podcasts, 
    get_topic_podcasts
)
from tools.news_tools import get_recent_news
from utils.formatting import format_podcast_response, format_news_response

# Define decision functions for routing
def determine_intent(state: InfoAssistantState) -> str:
    """Determine the user's intent from their message."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "general_query"
    
    content = last_message.content.lower()
    
    # Check for podcast-related intents
    if any(term in content for term in ["podcast", "episode", "listen", "audio show", "similar to"]):
        return "podcast_search"
    
    # Check for news-related intents
    if any(term in content for term in ["news", "recent events", "happened recently", "latest on", "update me"]):
        return "news_search"
    
    # Default to general query
    return "general_query"

# Processing nodes for different intents
def process_podcast_request(state: InfoAssistantState) -> InfoAssistantState:
    """Handle podcast-related requests using the appropriate tool."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Create tool executor with all podcast tools
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=TEMPERATURE)
    podcast_tools = [
        get_podcast_recommendations,
        get_podcast_details,
        get_similar_podcasts,
        get_topic_podcasts
    ]
    
    # Create specialized podcast agent
    podcast_prompt = """You are a podcast recommendation specialist. Your goal is to help the user find the perfect podcasts 
    based on their interests and queries. You have access to several specialized tools for podcast discovery.
    
    Think step by step about which podcast tool would best serve the user's request."""
    
    agent = create_tool_calling_agent(llm, podcast_tools, podcast_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=podcast_tools)
    
    # Execute the agent
    result = agent_executor.invoke({"messages": [last_message]})
    
    # Extract tool results and update history
    tool_outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    
    if tool_outputs:
        # We have podcast results to add to history
        podcast_history = state.get("podcast_history", [])
        for tool_msg in tool_outputs:
            if isinstance(tool_msg.content, str):
                # Parse string content if needed
                try:
                    import json
                    content = json.loads(tool_msg.content)
                except:
                    content = {"text_result": tool_msg.content}
            else:
                content = tool_msg.content
                
            podcast_history.append(content)
        
        # Get final AI message
        final_message = result["messages"][-1] if result["messages"] else AIMessage(content="I found some podcast information for you.")
        
        return {
            **state,
            "messages": messages + [final_message],
            "podcast_history": podcast_history,
            "current_task": "podcast",
            "last_tool_used": "podcast_tool"
        }
    
    # If no tool outputs, just return the AI message
    return {
        **state,
        "messages": messages + result["messages"],
        "current_task": "podcast",
        "last_tool_used": "podcast_tool"
    }

def process_news_request(state: InfoAssistantState) -> InfoAssistantState:
    """Handle news-related requests using the news tool."""
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content.lower()
    
    # Extract the topic
    topic = ""
    if "news about" in content:
        topic = content.split("news about")[-1].strip()
    elif "latest on" in content:
        topic = content.split("latest on")[-1].strip()
    elif "updates on" in content:
        topic = content.split("updates on")[-1].strip()
    else:
        # Default to using the whole query minus common words
        common_words = ["news", "recent", "latest", "update", "tell", "me", "about"]
        topic_words = [word for word in content.split() if word.lower() not in common_words]
        topic = " ".join(topic_words)
    
    # Extract time range if specified
    days_back = 7  # Default
    if "last week" in content:
        days_back = 7
    elif "last month" in content:
        days_back = 30
    elif "last day" in content or "yesterday" in content:
        days_back = 1
    
    # Create LLM for processing
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=TEMPERATURE)
    
    # Use tool to get news
    try:
        news_result = get_recent_news(topic=topic, days_back=days_back)
        
        # Update news history
        news_history = state.get("news_history", [])
        news_history.append(news_result)
        
        # Format a nice response
        response = format_news_response(news_result, topic, days_back)
        
        return {
            **state,
            "messages": messages + [AIMessage(content=response)],
            "news_history": news_history,
            "current_task": "news",
            "last_tool_used": "news_tool"
        }
    except Exception as e:
        # Handle errors
        return {
            **state,
            "messages": messages + [AIMessage(content=f"I encountered an issue while searching for news about '{topic}': {str(e)}. Would you like to try a different search?")],
            "current_task": "news",
            "last_tool_used": "news_tool"
        }

def process_general_request(state: InfoAssistantState) -> InfoAssistantState:
    """Handle general requests that could use any tool."""
    # Create a general assistant with access to all tools
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=TEMPERATURE)
    
    # Collect all tools
    all_tools = [
        get_podcast_recommendations,
        get_podcast_details,
        get_similar_podcasts,
        get_topic_podcasts,
        get_recent_news
    ]
    
    # Create the agent with tool-calling capability
    agent_prompt = """You are an intelligent assistant that helps users find information about podcasts and recent news.
    You have access to specialized tools for podcast recommendations and news searches.
    
    - For podcast requests: Help users find recommendations, get details about specific podcasts, or find similar podcasts
    - For news requests: Help users find recent articles on specific topics
    - For general queries: Engage in helpful conversation and determine if any of your tools might be useful
    
    Think step by step about which tool would best serve the user's request and only use a tool if it's truly needed."""
    
    agent = create_tool_calling_agent(llm, all_tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=all_tools)
    
    # Execute agent
    result = agent_executor.invoke({"messages": [state["messages"][-1]]})
    
    # Update state based on the tool used
    tool_outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    
    if tool_outputs:
        # Determine which type of tool was used
        if any("podcast" in str(msg.content).lower() for msg in tool_outputs):
            current_task = "podcast"
            
            # Update podcast history
            podcast_history = state.get("podcast_history", [])
            for tool_msg in tool_outputs:
                if "podcast" in str(tool_msg.content).lower():
                    if isinstance(tool_msg.content, str):
                        try:
                            import json
                            content = json.loads(tool_msg.content)
                        except:
                            content = {"text_result": tool_msg.content}
                    else:
                        content = tool_msg.content
                    
                    podcast_history.append(content)
            
            return {
                **state,
                "messages": state["messages"] + result["messages"],
                "podcast_history": podcast_history,
                "current_task": current_task,
                "last_tool_used": "podcast_tool"
            }
            
        elif any("news" in str(msg.content).lower() for msg in tool_outputs):
            current_task = "news"
            
            # Update news history
            news_history = state.get("news_history", [])
            for tool_msg in tool_outputs:
                if "news" in str(tool_msg.content).lower():
                    if isinstance(tool_msg.content, str):
                        try:
                            import json
                            content = json.loads(tool_msg.content)
                        except:
                            content = {"text_result": tool_msg.content}
                    else:
                        content = tool_msg.content
                    
                    news_history.append(content)
            
            return {
                **state,
                "messages": state["messages"] + result["messages"],
                "news_history": news_history,
                "current_task": current_task,
                "last_tool_used": "news_tool"
            }
    
    # Default case - no specific tool identified
    return {
        **state,
        "messages": state["messages"] + result["messages"],
        "current_task": "general",
        "last_tool_used": None
    }