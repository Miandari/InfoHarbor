"""
Modified workflow.py with a central main agent that controls the conversation
"""
from typing import Dict, List, Any, Tuple, Annotated, TypedDict, Sequence, Union, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import os  # Added missing os import
# Import all your tools
from tools.podcast_tools import podcast_tools
from tools.news_tools import get_recent_news
from utils.formatting import format_response, format_podcast_response, format_news_response

# Define the state for the flow
class InfoAssistantState(TypedDict):
    messages: Sequence[Union[HumanMessage, AIMessage, ToolMessage]]
    podcast_history: List[Dict[str, Any]]
    news_history: List[Dict[str, Any]]
    current_task: Literal["podcast", "news", "general"]
    last_tool_used: str
    context: Dict[str, Any]
    tool_results: Dict[str, Any]  # New field to store tool results

def create_info_assistant():
    """
    Create an information assistant with a main agent that dispatches to tool nodes
    and handles the final response.
    """
    # 1. Create a state graph
    workflow = StateGraph(InfoAssistantState)
    
    # 2. Define node functions
    
    # 2.1 Main agent node that handles conversation and dispatches to tools when needed
    def main_agent(state: InfoAssistantState) -> Dict[str, Any]:
        """
        Main conversation agent that determines when to use specialized tools.
        
        This function:
        1. Examines the latest message
        2. Decides whether to use a specialized tool or respond directly
        3. Routes to the appropriate tool node or responds
        """
        # Import debugging utility
        from utils.direct_response import debug_log
        debug_log("MAIN_AGENT - Entering main_agent node for routing")
        
        messages = state["messages"]
        if not messages:
            debug_log("MAIN_AGENT - No messages, ending flow")
            return {"next": "END"}
            
        last_message = messages[-1]
        
        # If this is a tool result coming back, process it in the respond node
        if state.get("tool_results") and state["tool_results"].get("pending"):
            debug_log("MAIN_AGENT - Tool results pending, routing to respond node")
            return {"next": "respond"}
            
        # Only process human messages for routing
        if not isinstance(last_message, HumanMessage):
            debug_log("MAIN_AGENT - Not a human message, routing to respond node")
            return {"next": "respond"}
            
        content = last_message.content.lower()
        debug_log(f"MAIN_AGENT - Processing message: {content}")
        
        # Expanded podcast-related keyword detection
        podcast_keywords = [
            "podcast", "episode", "listen", "audio show", "similar to",
            "recommend", "recommendation", "show me", "suggest", "series",
            "audio", "medieval", "history", "topics", "episodes"
        ]
        
        # Route to podcast tools if query relates to podcasts
        if any(term in content for term in podcast_keywords):
            debug_log("MAIN_AGENT - Detected podcast-related query, routing to podcast_tools")
            return {"next": "podcast_tools"}
            
        # Route to news tools if query relates to news
        news_keywords = [
            "news", "recent events", "happened recently", "latest on", "update me",
            "current events", "today's headlines", "breaking"
        ]
        if any(term in content for term in news_keywords):
            debug_log("MAIN_AGENT - Detected news-related query, routing to news_tools")
            return {"next": "news_tools"}
        
        # For ambiguous cases, use LLM-based classification
        if any(term in content for term in ["information", "tell me about", "what is", "how to"]):
            debug_log("MAIN_AGENT - Ambiguous query, using LLM classifier")
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            classification_prompt = f"""Classify this query into exactly one of these categories:
            - 'podcast' (if about podcast recommendations, episodes, listening to audio content)
            - 'news' (if about recent events, updates, current affairs)
            - 'general' (for general knowledge questions)
            
            Query: {content}
            
            Classification (respond with only one word, either 'podcast', 'news', or 'general'):"""
            
            classification = llm.invoke(classification_prompt)
            debug_log(f"MAIN_AGENT - LLM classification result: {classification}")
            
            if "podcast" in classification.lower():
                debug_log("MAIN_AGENT - LLM classified as podcast query, routing to podcast_tools")
                return {"next": "podcast_tools"}
            elif "news" in classification.lower():
                debug_log("MAIN_AGENT - LLM classified as news query, routing to news_tools")
                return {"next": "news_tools"}
        
        # Default: handle it directly in the respond node
        debug_log("MAIN_AGENT - Using default routing to respond node")
        return {"next": "respond"}
    
    # 2.2 Podcast tools node
    def podcast_tools_node(state: InfoAssistantState) -> InfoAssistantState:
        """Process podcast-related requests and store results for the main agent."""
        # Import debugging utility
        from utils.direct_response import debug_log
        debug_log("PODCAST_TOOLS - Entering podcast_tools_node")
        
        messages = state["messages"]
        last_message = messages[-1]
        debug_log(f"PODCAST_TOOLS - Processing message: {last_message.content}")
        
        podcast_history = state.get("podcast_history", [])
        context = state.get("context", {})
        
        # Create specialized podcast agent
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        debug_log("PODCAST_TOOLS - Created LLM instance")
        
        # Create prompt template for the podcast agent
        system_message = """You are a podcast recommendation specialist. Your goal is to help the user find the perfect podcasts 
        based on their interests and queries. You have access to several specialized tools for podcast discovery.
        
        Think step by step about which podcast tool would best serve the user's request. 
        Be thorough but concise in your tool usage."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("messages"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        debug_log("PODCAST_TOOLS - Creating agent with podcast tools")
        try:
            agent = create_tool_calling_agent(llm, podcast_tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=podcast_tools)
            
            # Execute the agent with just the last message
            debug_log("PODCAST_TOOLS - Invoking agent executor")
            result = agent_executor.invoke({"messages": [last_message]})
            debug_log(f"PODCAST_TOOLS - Agent returned result with {len(result['messages'])} messages")
            
            # Extract tool results
            tool_outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
            debug_log(f"PODCAST_TOOLS - Found {len(tool_outputs)} tool message outputs")
            
            tool_results = []
            
            # Process tool outputs
            for tool_msg in tool_outputs:
                debug_log(f"PODCAST_TOOLS - Processing tool message: {type(tool_msg).__name__}")
                if hasattr(tool_msg, "content"):
                    content = tool_msg.content
                    debug_log(f"PODCAST_TOOLS - Tool message content type: {type(content).__name__}")
                    if isinstance(content, str):
                        try:
                            import json
                            content = json.loads(content)
                            debug_log("PODCAST_TOOLS - Successfully parsed content as JSON")
                        except Exception as e:
                            debug_log(f"PODCAST_TOOLS - Failed to parse as JSON: {e}")
                            content = {"text_result": content}
                            
                    tool_results.append(content)
                    podcast_history.append(content)
            
            # Get the AI reasoning about the results, but don't add it to messages yet
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            debug_log(f"PODCAST_TOOLS - Found {len(ai_messages)} AI messages")
            
            final_ai_message = ai_messages[-1] if ai_messages else None
            debug_log(f"PODCAST_TOOLS - Final AI message: {final_ai_message.content if final_ai_message else 'None'}")
            
            # Store everything needed for the main agent to craft a response
            debug_log(f"PODCAST_TOOLS - Finished processing with {len(tool_results)} results")
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
            debug_log(f"PODCAST_TOOLS - Error: {e}")
            debug_log(traceback.format_exc())
            
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
    
    # 2.3 News tools node
    def news_tools_node(state: InfoAssistantState) -> InfoAssistantState:
        """Process news-related requests and store results for the main agent."""
        # Import debugging utility
        from utils.direct_response import debug_log
        debug_log("NEWS_TOOLS - Entering news_tools_node")
        
        messages = state["messages"]
        last_message = messages[-1]
        news_history = state.get("news_history", [])
        context = state.get("context", {})
        content = last_message.content
        debug_log(f"NEWS_TOOLS - Processing message: {content}")
        
        # Extract the topic from user message
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        extraction_prompt = f"""Extract the main topic and time period from this query:
        
        Query: {content}
        
        Format your response as JSON:
        {{
            "topic": "the main subject the user wants news about",
            "days_back": number of days to look back (default 7 if not specified, max 30)
        }}
        """
        
        extraction_result = llm.invoke(extraction_prompt)
        debug_log(f"NEWS_TOOLS - Extraction result: {extraction_result}")
        
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
                debug_log(f"NEWS_TOOLS - Extracted topic: {topic}, days_back: {days_back}")
            except Exception as e:
                debug_log(f"NEWS_TOOLS - JSON parsing error: {e}")
                topic = content
                days_back = 7
        else:
            debug_log("NEWS_TOOLS - No JSON found in extraction result, using defaults")
            topic = content
            days_back = 7
        
        # Use news tool with invoke() instead of calling directly
        try:
            debug_log(f"NEWS_TOOLS - Invoking get_recent_news with topic={topic}, days_back={days_back}")
            from tools.news_tools import get_recent_news
            
            # Use the invoke method instead of calling directly
            news_result = get_recent_news.invoke({"topic": topic, "days_back": days_back})
            debug_log(f"NEWS_TOOLS - Got news results with {len(news_result.get('articles', []))} articles")
            
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
            debug_log(f"NEWS_TOOLS - Error: {e}")
            import traceback
            debug_log(traceback.format_exc())
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
    
    # 2.4 Response handler node - Fixed state management
    def respond_node(state: InfoAssistantState) -> Dict[str, Any]:
        """Process all accumulated information and craft a final response."""
        messages = state["messages"]
        tool_results = state.get("tool_results", {})
        
        # Import debugging utilities
        from utils.direct_response import debug_log
        debug_log("WORKFLOW - Entering respond_node")
        debug_log(f"WORKFLOW - Messages count: {len(messages)}")
        debug_log(f"WORKFLOW - Last message: {messages[-1].content if messages else 'None'}")
        
        # If we have pending tool results, craft a response based on them
        if tool_results and tool_results.get("pending"):
            result_type = tool_results.get("type")
            
            if result_type == "podcast":
                # Format podcast results
                podcast_data = tool_results.get("data", [])
                # Use the formatted utility function for podcasts
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
                    response_content = format_news_response(news_data, topic, days_back)
                
            else:
                # Generic response for other types
                response_content = "I found some information that might help answer your query. Is there anything specific you'd like to know more about?"
            
            # Clear the pending flag
            tool_results["pending"] = False
        
        # Handle direct responses (no tool was used)
        else:
            # Get the user's question
            user_question = messages[-1].content if messages else ""
            debug_log(f"WORKFLOW - User question: {user_question}")
            
            # Import the direct response utility
            from utils.direct_response import get_direct_answer
            
            # Get a direct answer without going through LangGraph
            response_content = get_direct_answer(user_question)
            debug_log(f"WORKFLOW - Response from utility: {response_content}")
        
        # Create new AI message with the response
        ai_message = AIMessage(content=response_content)
        debug_log(f"WORKFLOW - Created AI message with content: {ai_message.content}")
        
        # Create a NEW messages list instead of modifying the existing one
        # This ensures LangGraph properly recognizes the state change
        new_messages = list(messages) + [ai_message]
        debug_log(f"WORKFLOW - New messages list created with {len(new_messages)} messages")
        debug_log(f"WORKFLOW - Last message in new messages list: {new_messages[-1].content}")
        
        # Create a completely fresh state object
        new_state = {
            "messages": new_messages,
            "podcast_history": state.get("podcast_history", []),
            "news_history": state.get("news_history", []),
            "current_task": "general" if not tool_results or not tool_results.get("pending") else state.get("current_task"),
            "last_tool_used": state.get("last_tool_used"),
            "context": state.get("context", {}),
            "tool_results": tool_results
        }
        
        debug_log(f"WORKFLOW - New state created with {len(new_state['messages'])} messages")
        debug_log(f"WORKFLOW - Last message in new state: {new_state['messages'][-1].content}")
        
        # Return a dictionary with explicit next node and the updated state
        return {"next": "END", "state": new_state}
    
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
        lambda x: x["next"],  # Extract the "next" key from the dictionary returned by main_agent
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
    
    # 3.4 Add conditional edges from respond node to handle state["next"]
    workflow.add_conditional_edges(
        "respond",
        lambda x: x["next"],  # Extract the next key from respond output
        {
            "END": END
        }
    )
    
    # 4. Compile and return the graph
    return workflow.compile()