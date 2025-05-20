"""
LangGraph processing nodes for the information assistant.
"""

from typing import Dict, Any, List, Sequence, Union, Optional, Literal, TypedDict
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config import DEFAULT_MODEL, TEMPERATURE, OPENAI_API_KEY
from graph.state import InfoAssistantState
from graph.schemas import ConfigSchema
from tools.podcast_tools import (
    get_podcast_recommendations, 
    get_podcast_details,
    get_similar_podcasts, 
    get_topic_podcasts
)
from tools.news_tools import get_recent_news
from tools.food_tools import process_food_order
from utils.formatting import format_podcast_response, format_news_response

# Define decision functions for routing
def determine_intent(state: InfoAssistantState) -> str:
    """Determine the user's intent from their message."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "general_query"
    
    content = last_message.content.lower()
    
    # Check for food ordering intents
    if any(term in content for term in ["order food", "order a pizza", "food delivery", "order pizza", "get food"]):
        return "food_order"
    
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

def process_food_order_request(state: InfoAssistantState) -> InfoAssistantState:
    """Handle food ordering requests."""
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content
    
    # Import the centralized state transitions
    from graph.transitions import StateTransitions
    
    # If this is the first food order interaction, always prompt the user for details
    if state.get("current_task") != "food_order" or not state.get("food_order_state"):
        # Create initial prompt for order details
        response = "I'd be happy to help you order food! Please provide your complete order details including:\n\n" + \
                   "1. Restaurant name\n" + \
                   "2. Items you'd like to order (including quantities)\n" + \
                   "3. Delivery or pickup preference\n" + \
                   "4. Delivery address (if applicable)\n" + \
                   "5. Any special instructions"
        
        return {
            **state,
            "messages": messages + [AIMessage(content=response)],
            "current_task": "food_order",
            "last_tool_used": None,
            "food_order_state": "collecting_details",
            "tool_results": {
                "type": "food_order",
                "pending": False
            }
        }
    
    # If the user has provided order details, mark food tools as pending
    state = StateTransitions.add_to_pending_tools(state, "food_order")
    
    try:
        # Process the food order using the food ordering tool
        result = process_food_order(order_text=content)
        
        # Use the centralized state transition function to handle food order completion
        completed_state = StateTransitions.handle_food_order_completion(state, content, result)
        
        # Verify that handle_food_order_completion properly cleared pending tools
        # If not, explicitly remove it here as a safety check
        if "food_order" in completed_state.get("pending_tools", []):
            completed_state = StateTransitions.remove_from_pending_tools(completed_state, "food_order")
        
        return completed_state
    except Exception as e:
        # Handle errors in order processing using centralized method
        return StateTransitions.handle_error_state(state, f"Error processing food order: {str(e)}", "food_order")

def extract_order_details(message: str) -> Dict[str, Any]:
    """Extract food order details from a user message."""
    # Create simple parsing logic for order messages
    # This is a simplified implementation - in a real system you might use an LLM for this
    details = {}
    lines = message.split("\n")
    
    # Try to extract structured format like "1. Restaurant 2. Items..."
    numbered_format = any(line.strip().startswith(("1.", "1:")) for line in lines)
    
    if numbered_format:
        # Process numbered format
        for line in lines:
            line = line.strip()
            if line.startswith("1.") or line.startswith("1:") or "restaurant" in line.lower():
                # Extract restaurant name
                parts = line.split(".", 1) if "." in line else line.split(":", 1)
                if len(parts) > 1:
                    details["restaurant"] = parts[1].strip()
                else:
                    # Try to extract after "restaurant"
                    parts = line.lower().split("restaurant", 1)
                    if len(parts) > 1:
                        details["restaurant"] = parts[1].strip()
            elif line.startswith("2.") or line.startswith("2:") or "item" in line.lower():
                # Extract items
                parts = line.split(".", 1) if "." in line else line.split(":", 1)
                if len(parts) > 1:
                    items_text = parts[1].strip()
                    details["items"] = [items_text]  # Simple version, not parsing multiple items
            elif line.startswith("3.") or line.startswith("3:") or "delivery" in line.lower() or "pickup" in line.lower():
                # Extract delivery method
                if "delivery" in line.lower():
                    details["delivery_method"] = "delivery"
                elif "pickup" in line.lower():
                    details["delivery_method"] = "pickup"
            elif line.startswith("4.") or line.startswith("4:") or "address" in line.lower():
                # Extract address for delivery
                parts = line.split(".", 1) if "." in line else line.split(":", 1)
                if len(parts) > 1:
                    details["address"] = parts[1].strip()
            elif line.startswith("5.") or line.startswith("5:") or "instruction" in line.lower():
                # Extract special instructions
                parts = line.split(".", 1) if "." in line else line.split(":", 1)
                if len(parts) > 1:
                    details["special_instructions"] = parts[1].strip()
    else:
        # Try to extract from unstructured text
        # Use simple heuristics - in reality, you'd use an LLM for this
        
        # Extract restaurant
        restaurant_indicators = ["from", "at", "restaurant", "order from"]
        for indicator in restaurant_indicators:
            if indicator in message.lower():
                parts = message.lower().split(indicator, 1)
                if len(parts) > 1:
                    # Take the text after the indicator until the next common word
                    restaurant_text = parts[1].strip().split()[0:3]
                    details["restaurant"] = " ".join(restaurant_text).strip(".,;: ")
                    break
        
        # Extract items (simplistic)
        if "pizza" in message.lower():
            details["items"] = ["pizza"]
            # Try to get quantity
            if "one" in message.lower() or "1" in message:
                details["items"] = ["1 Pizza"]
        
        # Extract delivery method
        if "delivery" in message.lower():
            details["delivery_method"] = "delivery"
        elif "pickup" in message.lower():
            details["delivery_method"] = "pickup"
        
        # Extract address if present (simplistic)
        address_indicators = ["address", "deliver to", "location"]
        for indicator in address_indicators:
            if indicator in message.lower():
                parts = message.lower().split(indicator, 1)
                if len(parts) > 1:
                    # Take the text after the indicator
                    address_text = parts[1].strip().split()[0:6]
                    details["address"] = " ".join(address_text).strip(".,;: ")
                    break
    
    # Set defaults for missing fields
    if "delivery_method" not in details:
        details["delivery_method"] = "delivery"  # Default to delivery
    
    return details

def format_order_confirmation(order_details: Dict[str, Any]) -> str:
    """Format a user-friendly order confirmation message."""
    confirmation = f"Got it! Here's your order summary:\n\n"
    confirmation += f"- Restaurant: {order_details['restaurant']}\n"
    
    # Format items
    confirmation += "- Pizza: "
    if isinstance(order_details['items'], list):
        confirmation += ", ".join(order_details['items'])
    else:
        confirmation += str(order_details['items'])
    confirmation += "\n"
    
    # Add delivery info
    confirmation += f"- Order type: {order_details['delivery_method'].capitalize()}\n"
    
    if order_details.get('address'):
        confirmation += f"- Delivery address: {order_details['address']}\n"
    
    if order_details.get('special_instructions'):
        confirmation += f"- Special instructions: {order_details['special_instructions']}\n"
    
    # Add confirmation
    confirmation += "\nI've sent your order details via Telegram! Is there anything else you need help with?"
    
    return confirmation

def process_general_request(state: InfoAssistantState) -> InfoAssistantState:
    """Handle general requests that could use any tool."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # For purely general queries, use direct OpenAI API instead of LangChain tools
    if not any(term in last_message.content.lower() for term in ["podcast", "episode", "news", "update"]):
        try:
            # Import the OpenAI client
            from openai import OpenAI
            import os
            
            # Create client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Format conversation history for OpenAI
            openai_messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer questions directly and comprehensively without repeating the question back to the user. Be informative and concise."}
            ]
            
            # Add recent conversation history (limit to 5 exchanges to avoid token limits)
            history_messages = messages[-10:]  # Last 5 exchanges (10 messages)
            for msg in history_messages:
                if isinstance(msg, HumanMessage):
                    openai_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    openai_messages.append({"role": "assistant", "content": msg.content})
            
            # Call OpenAI API directly
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=TEMPERATURE,
                messages=openai_messages
            )
            
            # Extract response content
            answer = response.choices[0].message.content
            
            # Return updated state with the new message
            return {
                **state,
                "messages": messages + [AIMessage(content=answer)],
                "current_task": "general",
                "last_tool_used": None
            }
            
        except Exception as e:
            # Fall back to tool-based approach if direct API fails
            print(f"Direct API call failed: {e}. Falling back to tool-based approach.")
    
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
    # FIX: Use a proper prompt template instead of a string
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant that helps users find information about podcasts and recent news.
        You have access to specialized tools for podcast recommendations and news searches.
        
        - For podcast requests: Help users find recommendations, get details about specific podcasts, or find similar podcasts
        - For news requests: Help users find recent articles on specific topics
        - For general queries: Engage in helpful conversation and determine if any of your tools might be useful
        
        Think step by step about which tool would best serve the user's request and only use a tool if it's truly needed."""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
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

def react_agent(state: InfoAssistantState, config: ConfigSchema) -> Dict[str, Any]:
    """
    Main ReAct agent that employs reasoning-action-observation pattern.
    
    This function:
    1. Examines the latest message and previous context
    2. Thinks step by step about what needs to be done
    3. Plans and takes actions through specialized tools
    4. Observes results and adapts the approach as needed
    """
    try:
        from utils.direct_response import debug_log, verbose_print
        debug_log("REACT_AGENT - Entering react_agent node")
    except ImportError:
        try:
            from utils.direct_response import verbose_print
            verbose_print("REACT_AGENT - Debug logging not available")
        except ImportError:
            pass
    
    try:
        messages = state["messages"]
        if not messages:
            try:
                verbose_print("REACT_AGENT - No messages, ending flow")
            except:
                pass
            return {"next": "END"}
        
        # Initialize ReAct-specific state fields if they don't exist
        reasoning = state.get("reasoning", [])
        working_memory = state.get("working_memory", {})
        next_actions = state.get("next_actions", [])
        pending_tools = state.get("pending_tools", [])
        
        # Get the last user message
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return {"next": "prepare_context"}  # Changed from "respond" to match workflow
        
        # Get the query text    
        query = last_message.content
            
        # Import the centralized state transitions
        from graph.transitions import StateTransitions
        
        # Apply state transition logic based on the new query
        state = StateTransitions.transition_from_task(state, query)
        
        # Determine the intent using the centralized logic
        intent = StateTransitions.determine_intent(query)
        debug_log(f"REACT_AGENT - Detected intent: {intent} for query: {query}")
        
        # CRITICAL FIX: Explicitly set the current_task in the state based on detected intent
        if intent == "food_order":
            state["current_task"] = "food_order"
            debug_log("REACT_AGENT - Explicitly setting current_task to 'food_order'")
            return {"next": "food_ordering"}  # Directly route to food_ordering
        elif intent == "news":
            state["current_task"] = "news"
            debug_log("REACT_AGENT - Explicitly setting current_task to 'news'")
            return {"next": "news_agent"}  # Directly route to news_agent
        elif intent == "podcast":
            state["current_task"] = "podcast"
            debug_log("REACT_AGENT - Explicitly setting current_task to 'podcast'")
            return {"next": "podcast_agent"}  # Directly route to podcast_agent
        else:
            state["current_task"] = "general"
            debug_log("REACT_AGENT - Explicitly setting current_task to 'general'")
            return {"next": "prepare_context"}  # Default to prepare_context
    
    except Exception as e:
        import traceback
        error_msg = f"Error in react_agent: {str(e)}\n{traceback.format_exc()}"
        try:
            from utils.direct_response import debug_log
            debug_log(error_msg)
        except:
            print(error_msg)
        
        # Default to prepare_context node on error (changed from "respond")
        return {"next": "prepare_context"}

def podcast_agent_node(state: InfoAssistantState) -> InfoAssistantState:
    """Process podcast-related requests using specialized podcast tools."""
    from graph.transitions import StateTransitions
    
    try:
        # Mark podcast tools as pending before execution
        state = StateTransitions.add_to_pending_tools(state, "podcast")
        
        # Process the podcast request
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
        
        # Mark podcast tools as complete
        state = StateTransitions.remove_from_pending_tools(state, "podcast")
        
        return {
            **state,
            "messages": messages + [final_message],
            "podcast_history": podcast_history,
            "current_task": "podcast",
            "last_tool_used": "podcast_tool",
            "tool_results": {
                "type": "podcast",
                "pending": False
            }
        }
        
        # If no tool outputs, just return the AI message
        # Mark podcast tools as complete
        state = StateTransitions.remove_from_pending_tools(state, "podcast")
        
        return {
            **state,
            "messages": messages + result["messages"],
            "current_task": "podcast",
            "last_tool_used": "podcast_tool",
            "tool_results": {
                "type": "podcast",
                "pending": False
            }
        }
    except Exception as e:
        # Use the centralized error handling method
        return StateTransitions.handle_error_state(state, f"Error in podcast tool: {str(e)}", "podcast")

def news_agent_node(state: InfoAssistantState) -> InfoAssistantState:
    """Process news-related requests using the news tool."""
    from graph.transitions import StateTransitions
    from utils.direct_response import debug_log
    import os
    import sys
    
    try:
        debug_log("NEWS_AGENT - Processing news request")
        
        # Mark news tools as pending before execution
        state = StateTransitions.add_to_pending_tools(state, "news")
        
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content.lower()
        
        # Extract the topic with improved parsing
        topic = ""
        if "news about" in content:
            topic = content.split("news about")[-1].strip()
        elif "latest on" in content:
            topic = content.split("latest on")[-1].strip()
        elif "updates on" in content:
            topic = content.split("updates on")[-1].strip()
        elif "tell me about" in content:
            topic = content.split("tell me about")[-1].strip()
        elif "important" in content and "news" in content:
            # For queries like "most important news"
            topic = "current events"
        elif "what's happening" in content or "what is happening" in content:
            topic = "current events"
        else:
            # Default to using the whole query minus common words
            common_words = ["news", "recent", "latest", "update", "tell", "me", "about", 
                           "the", "in", "on", "for", "of", "a", "an", "this", "last", 
                           "most", "important", "what", "is", "are", "were", "was", "happened"]
            topic_words = [word for word in content.split() if word.lower() not in common_words]
            topic = " ".join(topic_words) if topic_words else "current events"
        
        # If topic is empty after filtering, use a default
        if not topic.strip():
            topic = "current events"
            
        debug_log(f"NEWS_AGENT - Extracted topic: {topic}")
        
        # Extract time range if specified
        days_back = 7  # Default
        if "last week" in content:
            days_back = 7
        elif "last month" in content:
            days_back = 30
        elif "last day" in content or "yesterday" in content:
            days_back = 1
            
        debug_log(f"NEWS_AGENT - Using {days_back} days back for search")
        
        # Check if Tavily API key is available - DIAGNOSTIC PRINT ADDED
        from config import TAVILY_API_KEY
        tavily_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
        
        # Print diagnostic info (censored for security)
        if tavily_key:
            debug_log(f"NEWS_AGENT - Tavily API key found: {tavily_key[:4]}...{tavily_key[-4:] if len(tavily_key) > 8 else ''}") 
        else:
            debug_log("NEWS_AGENT - ERROR: No Tavily API key found")
            print("NEWS ERROR: No Tavily API key found in environment variables or config", file=sys.stderr)
            raise ValueError("Tavily API key is not configured. Please set the TAVILY_API_KEY environment variable.")
        
        debug_log(f"NEWS_AGENT - Calling get_recent_news for topic: {topic}")
        
        # Use tool to get news
        from tools.news_tools import get_recent_news
        
        # ADDED: Try direct invocation with explicit parameters
        news_result = get_recent_news.invoke({"topic": topic, "days_back": days_back})
        
        debug_log(f"NEWS_AGENT - Got news results with {news_result.get('article_count', 0)} articles")
        
        # Update news history
        news_history = state.get("news_history", [])
        news_history.append(news_result)
        
        # Format a nice response
        from utils.formatting import format_news_response
        response = format_news_response(news_result, topic, days_back)
        
        # Mark news tools as complete
        state = StateTransitions.remove_from_pending_tools(state, "news")
        
        debug_log("NEWS_AGENT - Successfully processed news request")
        
        return {
            **state,
            "messages": messages + [AIMessage(content=response)],
            "news_history": news_history,
            "current_task": "news",
            "last_tool_used": "news_tool",
            "tool_results": {
                "type": "news",
                "pending": False
            }
        }
    except Exception as e:
        import traceback
        error_msg = f"Error in news_agent_node: {str(e)}\n{traceback.format_exc()}"
        debug_log(error_msg)
        print(f"NEWS ERROR: {str(e)}", file=sys.stderr)
        
        # Show more detailed diagnostic info in the response for debugging
        diagnostic_msg = "I'm having trouble accessing news updates right now. There seems to be an issue with "
        
        if "Tavily API" in str(e) or "api_key" in str(e).lower():
            diagnostic_msg += "the news API configuration. The Tavily API key may be missing or invalid."
            # Add the actual error for debugging
            diagnostic_msg += f"\n\nTechnical details (for developers): {str(e)}"
            
            return {
                **state,
                "messages": messages + [AIMessage(content=diagnostic_msg)],
                "current_task": "general",
                "last_tool_used": None
            }
        else:
            diagnostic_msg += f"processing your news request. Technical details: {str(e)}"
            
            return {
                **state,
                "messages": messages + [AIMessage(content=diagnostic_msg)],
                "current_task": "general", 
                "last_tool_used": None
            }
def food_ordering_node(state: InfoAssistantState) -> InfoAssistantState:
    """Handle food ordering requests."""
    from graph.transitions import StateTransitions
    
    try:
        messages = state["messages"]
        last_message = messages[-1]
        content = last_message.content
        
        # If this is the first food order interaction, prompt the user for details
        if state.get("current_task") != "food_order" or not state.get("food_order_state"):
            # Create initial prompt for order details
            response = "I'd be happy to help you order food! Please provide your complete order details including:\n\n" + \
                       "1. Restaurant name\n" + \
                       "2. Items you'd like to order (including quantities)\n" + \
                       "3. Delivery or pickup preference\n" + \
                       "4. Delivery address (if applicable)\n" + \
                       "5. Any special instructions"
            
            return {
                **state,
                "messages": messages + [AIMessage(content=response)],
                "current_task": "food_order",
                "last_tool_used": None,
                "food_order_state": "collecting_details",
                "tool_results": {
                    "type": "food_order",
                    "pending": False
                }
            }
        
        # If the user has provided order details, mark food tools as pending
        state = StateTransitions.add_to_pending_tools(state, "food_order")
        
        try:
            # Process the food order using the food ordering tool
            result = process_food_order(order_text=content)
            
            # Use the centralized state transition function to handle food order completion
            completed_state = StateTransitions.handle_food_order_completion(state, content, result)
            
            # Verify that handle_food_order_completion properly cleared pending tools
            # If not, explicitly remove it here as a safety check
            if "food_order" in completed_state.get("pending_tools", []):
                completed_state = StateTransitions.remove_from_pending_tools(completed_state, "food_order")
            
            return completed_state
        except Exception as e:
            # Handle errors in order processing using centralized method
            return StateTransitions.handle_error_state(state, f"Error processing food order: {str(e)}", "food_order")
    
    except Exception as e:
        # Handle any unexpected errors in the node itself
        return StateTransitions.handle_error_state(state, f"Unexpected error in food ordering node: {str(e)}", "food_order")

def route_query(state: InfoAssistantState) -> Dict[str, Any]:
    """
    Route the user's query to the appropriate processing node based on intent detection.
    
    Args:
        state: Current conversation state
        
    Returns:
        Dictionary with "next" key indicating which node to route to
    """
    # Import necessary functions
    from utils.direct_response import debug_log
    from graph.transitions import StateTransitions
    
    try:
        debug_log("ROUTE_QUERY - Entering routing node")
        
        if not state.get("messages", []):
            debug_log("ROUTE_QUERY - No messages, ending flow")
            return {"next": "END"}
        
        # Get the last user message
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            debug_log("ROUTE_QUERY - Last message is not a user message, routing to response")
            return {"next": "respond"}
        
        # Get the query text
        query = last_message.content
            
        # Apply state transition logic based on the new query
        state = StateTransitions.transition_from_task(state, query)
        
        # Determine the intent using the centralized logic
        intent = StateTransitions.determine_intent(query)
        
        debug_log(f"ROUTE_QUERY - Intent detected: {intent} for query: {query}")
        
        # CRITICAL FIX: Explicitly set the current_task in the state based on detected intent
        if intent == "news":
            state["current_task"] = "news"
            debug_log("ROUTE_QUERY - Explicitly setting current_task to 'news'")
        elif intent == "podcast":
            state["current_task"] = "podcast" 
            debug_log("ROUTE_QUERY - Explicitly setting current_task to 'podcast'")
        elif intent == "food_order":
            state["current_task"] = "food_order"
            debug_log("ROUTE_QUERY - Explicitly setting current_task to 'food_order'")
        else:
            state["current_task"] = "general"
            debug_log("ROUTE_QUERY - Explicitly setting current_task to 'general'")
            
        # Map from intent to routing destination
        intent_routing = {
            "food_order": "food_ordering",
            "news": "news_agent",
            "podcast": "podcast_agent",
            "general": "prepare_context"
        }
        
        # Get the appropriate routing destination
        next_node = intent_routing.get(intent, "prepare_context")
        
        debug_log(f"ROUTE_QUERY - Detected {intent} intent, routing to {next_node}")
        debug_log(f"ROUTE_QUERY - Current state: current_task={state.get('current_task')}")
            
        return {"next": next_node}
    
    except Exception as e:
        import traceback
        error_msg = f"Error in route_query: {str(e)}\n{traceback.format_exc()}"
        try:
            debug_log(error_msg)
        except:
            print(error_msg)
        
        # Default to prepare_context node on error
        return {"next": "prepare_context"}

def prepare_context(state: InfoAssistantState) -> InfoAssistantState:
    """
    Prepare context for general requests by analyzing conversation history
    and potentially pre-fetching relevant information.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with enhanced context
    """
    from utils.direct_response import debug_log
    
    try:
        debug_log("PREPARE_CONTEXT - Entering context preparation node")
        
        # Process general requests using any appropriate tools
        return process_general_request(state)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in prepare_context: {str(e)}\n{traceback.format_exc()}"
        try:
            debug_log(error_msg)
        except:
            print(error_msg)
        
        # Use direct response method if tools fail
        from utils.direct_response import get_direct_answer
        
        messages = state.get("messages", [])
        if not messages:
            return state
            
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return state
            
        # Get a direct answer as fallback
        response = get_direct_answer(last_message.content)
        
        return {
            **state,
            "messages": messages + [AIMessage(content=response)],
            "current_task": "general",
            "last_tool_used": None
        }