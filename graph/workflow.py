"""
Modified workflow.py with a central main agent that controls the conversation
"""
from typing import Dict, List, Any, Tuple, Annotated, TypedDict, Sequence, Union, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import os
from config import DEFAULT_MODEL, TEMPERATURE, OPENAI_API_KEY

# Import all your tools
from tools.podcast_tools import podcast_tools
from tools.news_tools import get_recent_news
from utils.formatting import format_response, format_podcast_response, format_news_response

# Import debug utilities
from utils.direct_response import debug_log, verbose_print

# Import the state type
from graph.state import InfoAssistantState

# Define configuration schema for LangGraph Cloud compatibility
class ConfigSchema(TypedDict):
    model_name: str
    system_prompt: Optional[str]

def create_info_assistant():
    """
    Create an information assistant with a main agent that dispatches to tool nodes
    and handles the final response.
    """
    # 1. Create a state graph with config schema for Cloud compatibility
    workflow = StateGraph(InfoAssistantState, config_schema=ConfigSchema)
    
    # 2. Define node functions
    
    # 2.1 Main agent node that handles conversation and dispatches to tools when needed
    def main_agent(state: InfoAssistantState, config: ConfigSchema) -> Dict[str, Any]:
        """
        Main conversation agent that determines when to use specialized tools.
        
        This function:
        1. Examines the latest message
        2. Decides whether to use a specialized tool or respond directly
        3. Routes to the appropriate tool node or responds
        """
        # Import debugging utility
        try:
            debug_log("MAIN_AGENT - Entering main_agent node for routing")
        except ImportError:
            verbose_print("MAIN_AGENT - Debug logging not available")
        
        try:
            messages = state["messages"]
            if not messages:
                verbose_print("MAIN_AGENT - No messages, ending flow")
                return {"next": "END"}
                
            last_message = messages[-1]
            
            # If this is a tool result coming back, process it in the respond node
            if state.get("tool_results") and state["tool_results"].get("pending"):
                verbose_print("MAIN_AGENT - Tool results pending, routing to respond node")
                return {"next": "respond"}
                
            # Only process human messages for routing
            if not isinstance(last_message, HumanMessage):
                if isinstance(last_message, dict) and last_message.get("type") == "human":
                    # Convert dict to HumanMessage
                    content = last_message.get("content", "")
                else:
                    verbose_print("MAIN_AGENT - Not a human message, routing to respond node")
                    return {"next": "respond"}
            
            # Extract content safely
            if isinstance(last_message, HumanMessage):
                content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]
            else:
                content = str(last_message)
                
            content = content.lower()
            verbose_print(f"MAIN_AGENT - Processing message: {content}")
            
            # Expanded podcast-related keyword detection
            podcast_keywords = [
                "podcast", "episode", "listen", "audio show", "similar to",
                "recommend", "recommendation", "show me", "suggest", "series",
                "audio", "medieval", "history", "topics", "episodes"
            ]
            
            # Route to podcast tools if query relates to podcasts
            if any(term in content for term in podcast_keywords):
                verbose_print("MAIN_AGENT - Detected podcast-related query, routing to podcast_tools")
                return {"next": "podcast_tools"}
                
            # Route to news tools if query relates to news
            news_keywords = [
                "news", "recent events", "happened recently", "latest on", "update me",
                "current events", "today's headlines", "breaking"
            ]
            if any(term in content for term in news_keywords):
                verbose_print("MAIN_AGENT - Detected news-related query, routing to news_tools")
                return {"next": "news_tools"}
            
            # For ambiguous cases, use LLM-based classification
            if any(term in content for term in ["information", "tell me about", "what is", "how to"]):
                verbose_print("MAIN_AGENT - Ambiguous query, using LLM classifier")
                try:
                    # Use the configured model from config if available
                    model_name = config.get("model_name", DEFAULT_MODEL)
                    llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
                    classification_prompt = """Classify this query into exactly one of these categories:
                    - 'podcast' (if about podcast recommendations, episodes, listening to audio content)
                    - 'news' (if about recent events, updates, current affairs)
                    - 'general' (for general knowledge questions)
                    
                    Query: """ + content + """
                    
                    Classification (respond with only one word, either 'podcast', 'news', or 'general'):"""
                    
                    classification = llm.invoke(classification_prompt)
                    verbose_print(f"MAIN_AGENT - LLM classification result: {classification}")
                    
                    if "podcast" in classification.content.lower():
                        verbose_print("MAIN_AGENT - LLM classified as podcast query, routing to podcast_tools")
                        return {"next": "podcast_tools"}
                    elif "news" in classification.content.lower():
                        verbose_print("MAIN_AGENT - LLM classified as news query, routing to news_tools")
                        return {"next": "news_tools"}
                except Exception as e:
                    verbose_print(f"MAIN_AGENT - Error in LLM classification: {str(e)}")
            
            # Default: handle it directly in the respond node
            verbose_print("MAIN_AGENT - Using default routing to respond node")
            return {"next": "respond"}
            
        except Exception as e:
            verbose_print(f"MAIN_AGENT - Error: {str(e)}")
            import traceback
            verbose_print(traceback.format_exc())
            return {"next": "respond"}
    
    # 2.2 Podcast tools node
    def podcast_tools_node(state: InfoAssistantState, config: ConfigSchema) -> InfoAssistantState:
        """Process podcast-related requests and store results for the main agent."""
        # Import debugging utility
        try:
            debug_log("PODCAST_TOOLS - Entering podcast_tools_node")
        except ImportError:
            verbose_print("PODCAST_TOOLS - Debug logging not available")
        
        try:
            messages = state["messages"]
            last_message = messages[-1]
            
            # Handle different message formats
            if isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]
                # Create a HumanMessage object for the agent executor
                last_message = HumanMessage(content=content)
            elif isinstance(last_message, HumanMessage):
                content = last_message.content
            else:
                content = str(last_message)
                last_message = HumanMessage(content=content)
                
            verbose_print(f"PODCAST_TOOLS - Processing message: {content}")
            
            podcast_history = state.get("podcast_history", [])
            context = state.get("context", {})
            
            # Create specialized podcast agent
            # Use the configured model from config if available
            model_name = config.get("model_name", DEFAULT_MODEL)
            llm = ChatOpenAI(model=model_name, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
            verbose_print("PODCAST_TOOLS - Created LLM instance")
            
            # Create prompt template for the podcast agent
            system_message = config.get("system_prompt", """You are a podcast recommendation specialist. Your goal is to help the user find the perfect podcasts based on their interests and queries. You have access to several specialized tools for podcast discovery.
            
Think step by step about which podcast tool would best serve the user's request.
Be thorough but concise in your tool usage.""")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder("messages"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            verbose_print("PODCAST_TOOLS - Creating agent with podcast tools")
            try:
                agent = create_tool_calling_agent(llm, podcast_tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=podcast_tools)
                
                # Execute the agent with just the last message
                verbose_print("PODCAST_TOOLS - Invoking agent executor")
                result = agent_executor.invoke({"messages": [last_message]})
                verbose_print(f"PODCAST_TOOLS - Agent returned result with {len(result['messages'])} messages")
                
                # Extract tool results
                tool_outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
                verbose_print(f"PODCAST_TOOLS - Found {len(tool_outputs)} tool message outputs")
                
                tool_results = []
                
                # Process tool outputs
                for tool_msg in tool_outputs:
                    verbose_print(f"PODCAST_TOOLS - Processing tool message: {type(tool_msg).__name__}")
                    if hasattr(tool_msg, "content"):
                        content = tool_msg.content
                        verbose_print(f"PODCAST_TOOLS - Tool message content type: {type(content).__name__}")
                        if isinstance(content, str):
                            try:
                                content = json.loads(content)
                                verbose_print("PODCAST_TOOLS - Successfully parsed content as JSON")
                            except Exception as e:
                                verbose_print(f"PODCAST_TOOLS - Failed to parse as JSON: {e}")
                                content = {"text_result": content}
                                
                        tool_results.append(content)
                        podcast_history.append(content)
                
                # Get the AI reasoning about the results, but don't add it to messages yet
                ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                verbose_print(f"PODCAST_TOOLS - Found {len(ai_messages)} AI messages")
                
                final_ai_message = ai_messages[-1] if ai_messages else None
                verbose_print(f"PODCAST_TOOLS - Final AI message: {final_ai_message.content if final_ai_message else 'None'}")
                
                # Store everything needed for the main agent to craft a response
                verbose_print(f"PODCAST_TOOLS - Finished processing with {len(tool_results)} results")
                return {
                    **state,
                    "podcast_history": podcast_history,
                    "current_task": "podcast",
                    "last_tool_used": "podcast_tool",
                    "tool_results": {
                        "type": "podcast",
                        "data": tool_results,
                        "ai_reasoning": final_ai_message.content if final_ai_message else None,
                        "pending": True
                    }
                }
            except Exception as e:
                import traceback
                verbose_print(f"PODCAST_TOOLS - Error: {e}")
                verbose_print(traceback.format_exc())
                
                # Return a state that indicates an error occurred
                return {
                    **state,
                    "podcast_history": podcast_history,
                    "current_task": "podcast",
                    "last_tool_used": "podcast_tool",
                    "tool_results": {
                        "type": "podcast",
                        "error": str(e),
                        "pending": True
                    }
                }
        except Exception as e:
            verbose_print(f"PODCAST_TOOLS - Critical error: {str(e)}")
            import traceback
            verbose_print(traceback.format_exc())
            
            return {
                **state,
                "current_task": "podcast",
                "last_tool_used": "podcast_tool",
                "tool_results": {
                    "type": "podcast",
                    "error": str(e),
                    "pending": True
                }
            }
    
    # 2.3 News tools node
    def news_tools_node(state: InfoAssistantState, config: ConfigSchema) -> InfoAssistantState:
        """Process news-related requests and store results for the main agent."""
        # Import debugging utility
        try:
            debug_log("NEWS_TOOLS - Entering news_tools_node")
        except ImportError:
            verbose_print("NEWS_TOOLS - Debug logging not available")
        
        try:
            messages = state["messages"]
            last_message = messages[-1]
            news_history = state.get("news_history", [])
            context = state.get("context", {})
            
            # Handle different message formats
            if isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]
            elif isinstance(last_message, HumanMessage) and hasattr(last_message, "content"):
                content = last_message.content
            else:
                content = str(last_message)
                
            verbose_print(f"NEWS_TOOLS - Processing message: {content}")
            
            # Extract the topic from user message
            # Use the configured model from config if available
            model_name = config.get("model_name", DEFAULT_MODEL)
            llm = ChatOpenAI(model=model_name, temperature=0, api_key=OPENAI_API_KEY)
            extraction_prompt = """Extract the main topic and time period from this query:
            
Query: """ + content + """
            
Format your response as JSON:
{
    "topic": "the main subject the user wants news about",
    "days_back": number of days to look back (default 7 if not specified, max 30)
}"""
            
            extraction_result = llm.invoke(extraction_prompt)
            verbose_print(f"NEWS_TOOLS - Extraction result: {extraction_result}")
            
            # Parse the extraction result
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'({.*})', extraction_result.content.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    extracted = json.loads(json_match.group(1))
                    topic = extracted.get("topic", "")
                    days_back = min(int(extracted.get("days_back", 7)), 30)
                    verbose_print(f"NEWS_TOOLS - Extracted topic: {topic}, days_back: {days_back}")
                except Exception as e:
                    verbose_print(f"NEWS_TOOLS - JSON parsing error: {e}")
                    topic = content
                    days_back = 7
            else:
                verbose_print("NEWS_TOOLS - No JSON found in extraction result, using defaults")
                topic = content
                days_back = 7
            
            # Use news tool with invoke() instead of calling directly
            try:
                verbose_print(f"NEWS_TOOLS - Invoking get_recent_news with topic={topic}, days_back={days_back}")
                from tools.news_tools import get_recent_news
                
                # Use the invoke method instead of calling directly
                news_result = get_recent_news.invoke({"topic": topic, "days_back": days_back})
                verbose_print(f"NEWS_TOOLS - Got news results with {len(news_result.get('articles', []))} articles")
                
                # Update news history
                news_history.append(news_result)
                
                # Store the results for the main agent
                return {
                    **state,
                    "news_history": news_history,
                    "current_task": "news",
                    "last_tool_used": "news_tool",
                    "tool_results": {
                        "type": "news",
                        "data": news_result,
                        "topic": topic,
                        "days_back": days_back,
                        "pending": True
                    }
                }
            except Exception as e:
                # Store error for main agent to handle
                verbose_print(f"NEWS_TOOLS - Error: {e}")
                import traceback
                verbose_print(traceback.format_exc())
                return {
                    **state,
                    "news_history": news_history,
                    "current_task": "news",
                    "last_tool_used": "news_tool",
                    "tool_results": {
                        "type": "news",
                        "error": str(e),
                        "topic": topic,
                        "pending": True
                    }
                }
        except Exception as e:
            verbose_print(f"NEWS_TOOLS - Critical error: {str(e)}")
            import traceback
            verbose_print(traceback.format_exc())
            
            return {
                **state,
                "current_task": "news",
                "last_tool_used": "news_tool",
                "tool_results": {
                    "type": "news",
                    "error": str(e),
                    "pending": True
                }
            }
    
    # 2.4 Response handler node - Fixed state management
    def respond_node(state: InfoAssistantState, config: ConfigSchema) -> Dict[str, Any]:
        """Process all accumulated information and craft a final response."""
        try:
            messages = state["messages"]
            tool_results = state.get("tool_results", {})
            last_tool_used = state.get("last_tool_used", "")
            current_task = state.get("current_task", "")
            
            # Import debugging utilities
            try:
                debug_log("WORKFLOW - Entering respond_node")
                debug_log(f"WORKFLOW - Messages count: {len(messages)}")
                debug_log(f"WORKFLOW - Current task: {current_task}, Last tool used: {last_tool_used}")
                debug_log(f"WORKFLOW - Tool results present: {bool(tool_results)}")
                if tool_results:
                    debug_log(f"WORKFLOW - Tool results type: {tool_results.get('type')}")
                    debug_log(f"WORKFLOW - Tool results pending: {tool_results.get('pending')}")
            except ImportError:
                verbose_print("WORKFLOW - Debug logging not available")
            
            # Check message type and handle accordingly
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    last_message_content = last_message["content"]
                elif isinstance(last_message, (HumanMessage, AIMessage, ToolMessage)) and hasattr(last_message, "content"):
                    last_message_content = last_message.content
                else:
                    last_message_content = str(last_message)
                verbose_print(f"WORKFLOW - Last message: {last_message_content}")
            else:
                verbose_print("WORKFLOW - No messages found")
                last_message_content = ""
            
            # Check for news data in news_history even if tool_results isn't marked correctly
            news_history = state.get("news_history", [])
            podcast_history = state.get("podcast_history", [])
            
            response_content = ""
            
            # CLOUD COMPATIBILITY FIX: Check for recent tool results in history even if pending flag is wrong
            if last_tool_used == "news_tool" or current_task == "news" or (tool_results and tool_results.get("type") == "news"):
                if news_history and len(news_history) > 0:
                    # Use the most recent news result
                    news_data = news_history[-1]
                    topic = news_data.get("topic", "")
                    
                    # Import the formatting function
                    from utils.formatting import format_news_response
                    
                    # Format news response
                    response_content = format_news_response(news_data, topic, 7)
                    verbose_print(f"WORKFLOW - Generated NEWS response from history: {response_content[:50]}...")
                
            # Check for podcast data in podcast_history
            elif last_tool_used == "podcast_tool" or current_task == "podcast" or (tool_results and tool_results.get("type") == "podcast"):
                if podcast_history and len(podcast_history) > 0:
                    # Import the formatting function
                    from utils.formatting import format_podcast_response
                    
                    # Format podcast response
                    response_content = format_podcast_response({"recommendations": podcast_history})
                    verbose_print(f"WORKFLOW - Generated PODCAST response from history: {response_content[:50]}...")
            
            # If we still don't have a response but have pending tool results, try to use them directly
            if not response_content and tool_results and tool_results.get("pending"):
                result_type = tool_results.get("type")
                
                if result_type == "podcast":
                    # Format podcast results
                    podcast_data = tool_results.get("data", [])
                    # Use the formatted utility function for podcasts
                    from utils.formatting import format_podcast_response
                    response_content = format_podcast_response({"recommendations": podcast_data})
                    
                elif result_type == "news":
                    # Format news results
                    news_data = tool_results.get("data", {})
                    topic = tool_results.get("topic", "")
                    days_back = tool_results.get("days_back", 7)
                    error = tool_results.get("error")
                    
                    if error:
                        response_content = f"I encountered an issue while searching for news about '{topic}': {error}. Would you like to try a different search?"
                    else:
                        # Format a nice response using the utility function
                        from utils.formatting import format_news_response
                        response_content = format_news_response(news_data, topic, days_back)
                
                # Clear the pending flag
                tool_results["pending"] = False
                verbose_print(f"WORKFLOW - Generated response from tool results: {response_content[:50]}...")
            
            # Handle direct responses (no tool was used, or all previous attempts failed)
            if not response_content:
                # Get the user's question safely handling different message types
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        user_question = last_message["content"]
                    elif isinstance(last_message, (HumanMessage, AIMessage, ToolMessage)) and hasattr(last_message, "content"):
                        user_question = last_message.content
                    else:
                        user_question = str(last_message)
                else:
                    user_question = ""
                    
                verbose_print(f"WORKFLOW - User question: {user_question}")
                
                # Import the direct response utility
                try:
                    from utils.direct_response import get_direct_answer
                    
                    # Use the configured model from config if available
                    model_name = config.get("model_name", DEFAULT_MODEL)
                    # Get a direct answer without going through LangGraph
                    response_content = get_direct_answer(user_question, model_name=model_name)
                except Exception as e:
                    verbose_print(f"WORKFLOW - Error using direct_response: {e}")
                    # Fallback if direct_response fails
                    model_name = config.get("model_name", DEFAULT_MODEL)
                    llm = ChatOpenAI(model=model_name, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
                    result = llm.invoke([{"role": "user", "content": user_question}])
                    response_content = result.content
                
                verbose_print(f"WORKFLOW - Response from utility: {response_content[:50]}...")
            
            # Create new AI message with the response
            ai_message = AIMessage(content=response_content)
            verbose_print(f"WORKFLOW - Created AI message with content: {ai_message.content[:50]}...")
            
            # Create a NEW messages list instead of modifying the existing one
            new_messages = list(messages) + [ai_message]
            verbose_print(f"WORKFLOW - New messages list created with {len(new_messages)} messages")
            verbose_print(f"WORKFLOW - Last message in new messages list: {new_messages[-1].content[:50]}...")
            
            # Create a completely fresh state object - preserve the current_task
            new_state = {
                "messages": new_messages,
                "podcast_history": state.get("podcast_history", []),
                "news_history": state.get("news_history", []),
                "current_task": current_task,  # Keep the original task
                "last_tool_used": state.get("last_tool_used"),
                "context": state.get("context", {}),
                "tool_results": tool_results
            }
            
            verbose_print(f"WORKFLOW - New state created with {len(new_state['messages'])} messages")
            verbose_print(f"WORKFLOW - Last message in new state: {new_state['messages'][-1].content[:50]}...")
            
            # Return updated state directly (simpler and more reliable)
            return new_state
            
        except Exception as e:
            verbose_print(f"WORKFLOW - Critical error: {e}")
            import traceback
            verbose_print(traceback.format_exc())
            
            # Create an emergency response
            emergency_message = AIMessage(content="I'm sorry, I encountered an issue processing your request. Could you try again?")
            
            # Create emergency state with the response
            emergency_state = {
                "messages": list(state.get("messages", [])) + [emergency_message],
                "podcast_history": state.get("podcast_history", []),
                "news_history": state.get("news_history", []),
                "current_task": state.get("current_task", "general"),
                "last_tool_used": state.get("last_tool_used", ""),
                "context": state.get("context", {}),
                "tool_results": {}
            }
            
            return emergency_state
    
    # 3. Set up the graph with the nodes
    
    # 3.1 Add all nodes
    workflow.add_node("main_agent", main_agent)
    workflow.add_node("podcast_tools", podcast_tools_node)
    workflow.add_node("news_tools", news_tools_node)
    workflow.add_node("respond", respond_node)
    
    # 3.2 Define edges - conditional routing from main router to tool nodes
    workflow.set_entry_point("main_agent")
    workflow.add_conditional_edges(
        "main_agent",
        lambda x: x.get("next", "respond"),  # Extract the "next" key with a fallback to respond
        {
            "podcast_tools": "podcast_tools",
            "news_tools": "news_tools",
            "respond": "respond",
            "END": END
        }
    )
    
    # 3.3 Connect tool nodes back to respond
    workflow.add_edge("podcast_tools", "respond")
    workflow.add_edge("news_tools", "respond")
    
    # 4. Compile and return the graph
    return workflow.compile()