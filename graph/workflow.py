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
    def main_agent(state: InfoAssistantState) -> Literal["podcast_tools", "news_tools", "respond", "END"]:
        """
        Main conversation agent that determines when to use specialized tools.
        
        This function:
        1. Examines the latest message
        2. Decides whether to use a specialized tool or respond directly
        3. Routes to the appropriate tool node or responds
        """
        messages = state["messages"]
        if not messages:
            return "END"
            
        last_message = messages[-1]
        
        # If this is a tool result coming back, process it in the respond node
        if state.get("tool_results") and state["tool_results"].get("pending"):
            return "respond"
            
        # Only process human messages for routing
        if not isinstance(last_message, HumanMessage):
            return "respond"
            
        content = last_message.content.lower()
        
        # Route to podcast tools if query relates to podcasts
        if any(term in content for term in ["podcast", "episode", "listen", "audio show", "similar to"]):
            return "podcast_tools"
        # Route to news tools if query relates to news
        elif any(term in content for term in ["news", "recent events", "happened recently", "latest on", "update me"]):
            return "news_tools"
        
        # If not specific, use a more general LLM-based classifier for borderline cases
        if any(term in content for term in ["recommend", "what happened", "information about", "tell me about"]):
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            classification = llm.invoke(f"""Classify this query as either 'podcast', 'news', or 'general':
            
            Query: {content}
            
            Classification (respond with only one word, either 'podcast', 'news', or 'general'):""")
            
            if "podcast" in classification.lower():
                return "podcast_tools"
            elif "news" in classification.lower():
                return "news_tools"
        
        # Default: handle it directly in the respond node
        return "respond"
    
    # 2.2 Podcast tools node
    def podcast_tools_node(state: InfoAssistantState) -> InfoAssistantState:
        """Process podcast-related requests and store results for the main agent."""
        messages = state["messages"]
        last_message = messages[-1]
        podcast_history = state.get("podcast_history", [])
        context = state.get("context", {})
        
        # Create specialized podcast agent
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        
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
        
        agent = create_tool_calling_agent(llm, podcast_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=podcast_tools)
        
        # Execute the agent with just the last message
        result = agent_executor.invoke({"messages": [last_message]})
        
        # Extract tool results
        tool_outputs = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        tool_results = []
        
        # Process tool outputs
        for tool_msg in tool_outputs:
            if hasattr(tool_msg, "content"):
                content = tool_msg.content
                if isinstance(content, str):
                    try:
                        import json
                        content = json.loads(content)
                    except:
                        content = {"text_result": content}
                        
                tool_results.append(content)
                podcast_history.append(content)
        
        # Get the AI reasoning about the results, but don't add it to messages yet
        # The main agent will handle the final response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        final_ai_message = ai_messages[-1] if ai_messages else None
        
        # Store everything needed for the main agent to craft a response
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
    
    # 2.3 News tools node
    def news_tools_node(state: InfoAssistantState) -> InfoAssistantState:
        """Process news-related requests and store results for the main agent."""
        messages = state["messages"]
        last_message = messages[-1]
        news_history = state.get("news_history", [])
        context = state.get("context", {})
        content = last_message.content
        
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
            except:
                topic = content
                days_back = 7
        else:
            topic = content
            days_back = 7
        
        # Use news tool
        try:
            news_result = get_recent_news(topic=topic, days_back=days_back)
            
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
    
    # 2.4 Response handler node
    def respond_node(state: InfoAssistantState) -> InfoAssistantState:
        """Process all accumulated information and craft a final response."""
        messages = state["messages"]
        context = state.get("context", {})
        tool_results = state.get("tool_results", {})
        
        # Create the main LLM for responses
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # If we have pending tool results, craft a response based on them
        if tool_results and tool_results.get("pending"):
            result_type = tool_results.get("type")
            
            if result_type == "podcast":
                # Format podcast results
                podcast_data = tool_results.get("data", [])
                ai_reasoning = tool_results.get("ai_reasoning", "")
                
                # Create a prompt for generating a coherent response
                prompt = f"""You are an intelligent assistant helping the user find podcasts.
                
                The user asked: {messages[-1].content if messages else "about podcasts"}
                
                Here are the podcast search results:
                {json.dumps(podcast_data, indent=2)}
                
                Here's some reasoning about the results:
                {ai_reasoning}
                
                Craft a helpful, conversational response that:
                1. Summarizes the podcast recommendations clearly
                2. Highlights key features of each recommended podcast
                3. Suggests what the user might enjoy about them
                4. Asks if they'd like more details on any specific podcast
                
                Response:"""
                
                response_content = llm.invoke(prompt)
                
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
            
            # Return updated state with the new AI message
            return {
                **state,
                "messages": messages + [AIMessage(content=response_content)],
                "tool_results": tool_results
            }
        
        # Handle direct responses (no tool was used)
        else:
            # Use a general system prompt
            system_prompt = """You are an intelligent, helpful assistant with expertise in both podcasts and news.
            
            You can help users find podcast recommendations or get updates on recent news events.
            Be conversational, helpful, and concise in your responses.
            
            If the user asks about podcasts or news, let them know you can provide more specific information if they ask.
            """
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("messages"),
            ])
            
            # Generate a response
            response = llm.invoke(prompt.format(messages=messages[-1:]))
            
            # Return updated state with the new message
            return {
                **state,
                "messages": messages + [AIMessage(content=response.content)],
                "current_task": "general"
            }
    
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
        main_agent,
        {
            "podcast_tools": "podcast_tools",
            "news_tools": "news_tools",
            "respond": "respond",
            "END": END
        }
    )
    
    # 3.3 Connect tool nodes back to the main agent
    workflow.add_edge("podcast_tools", "respond")
    workflow.add_edge("news_tools", "respond")
    
    # 3.4 Connect respond node back to main agent for the next query
    workflow.add_edge("respond", "main_agent")
    
    # 4. Compile and return the graph
    return workflow.compile()