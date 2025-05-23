"""
Main LangGraph workflow for the elderly assistant agent
"""
import time
import uuid
from typing import Dict, Any, List
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.state import AgentState
from src.agent.nodes import AgentNode
from src.processors.preprocessor import PreprocessorNode
from src.processors.postprocessor import PostprocessorNode
from src.tools.base import tool_registry
from src.tools.podcast_tool import PodcastDiscoveryTool
from src.tools.news_tool import NewsDiscoveryTool
from src.config.settings import settings


# Register tools
tool_registry.register(PodcastDiscoveryTool())
tool_registry.register(NewsDiscoveryTool())


def create_agent_graph():
    """Create the main agent workflow graph"""
    
    # Initialize nodes
    preprocessor = PreprocessorNode()
    agent_node = AgentNode()
    postprocessor = PostprocessorNode()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("preprocessor", preprocessor.process)
    workflow.add_node("agent", agent_node.process)
    workflow.add_node("postprocessor", postprocessor.process)
    
    # Define edges
    workflow.add_edge("preprocessor", "agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        route_agent_response,
        {
            "postprocessor": "postprocessor",
            "end": END
        }
    )
    
    # Conditional edge from postprocessor
    workflow.add_conditional_edges(
        "postprocessor",
        route_postprocessor_response,
        {
            "agent": "agent",  # Retry if quality issues
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("preprocessor")
    
    # Compile the graph
    return workflow.compile()


def route_agent_response(state: AgentState) -> str:
    """Determine where to route after agent processing"""
    
    # Check if there was an error
    if state.get("error"):
        return "end"
    
    # Check if agent processing is done
    if state.get("agent_done"):
        return "postprocessor"
    
    return "end"


def route_postprocessor_response(state: AgentState) -> str:
    """Determine where to route after postprocessing"""
    
    # Check if revision is needed
    if state.get("needs_revision") and state.get("retry_count", 0) < 2:
        # Reset for retry
        state["needs_revision"] = False
        state["agent_done"] = False
        return "agent"
    
    return "end"


class ElderlyAssistantAgent:
    """Main agent class that wraps the LangGraph workflow"""
    
    def __init__(self):
        self.graph = create_agent_graph()
        self.logger = logging.getLogger("agent")
        
    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        platform: str = "api",
        elder_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response
        
        Args:
            message: The user's message
            user_id: Unique user identifier
            conversation_id: Optional conversation ID for continuity
            platform: Platform type (api, web, voice, etc.)
            elder_mode: Whether to use elder-friendly features
            
        Returns:
            Dict containing response and metadata
        """
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Create initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "conversation_id": conversation_id,
            "user_id": user_id,
            "memory_context": None,
            "memory_updates": [],
            "enhanced_context": None,
            "system_prompt": "",
            "tools_used": [],
            "tool_results": {},
            "start_time": time.time(),
            "preprocessing_done": False,
            "agent_done": False,
            "postprocessing_done": False,
            "elder_mode": elder_mode,
            "requires_clarification": False,
            "health_check_needed": False,
            "platform": platform,
            "response_format": "text",
            "final_response": None,
            "metadata": {},
            "error": None,
            "retry_count": 0
        }
        
        try:
            # Run the graph
            self.logger.info(f"Processing message for user {user_id}")
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract final response
            if final_state.get("final_response"):
                return final_state["final_response"]
            else:
                # Fallback if postprocessing didn't complete
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        return {
                            "response": msg.content,
                            "conversation_id": conversation_id,
                            "created_at": datetime.now().isoformat(),
                            "metadata": {
                                "latency_ms": int((time.time() - initial_state["start_time"]) * 1000),
                                "user_id": user_id,
                                "error": "Postprocessing incomplete"
                            }
                        }
                        
                # No AI response found
                raise Exception("No response generated")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I'm having trouble processing your request. Please try again.",
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "latency_ms": int((time.time() - initial_state["start_time"]) * 1000),
                    "user_id": user_id,
                    "error": str(e)
                }
            }
    
    async def chat_stream(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        platform: str = "api",
        elder_mode: bool = True
    ):
        """
        Stream a chat response
        
        Yields:
            Dict containing streaming events
        """
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Yield start event
        yield {
            "type": "start",
            "conversation_id": conversation_id
        }
        
        # Create initial state (same as chat method)
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "conversation_id": conversation_id,
            "user_id": user_id,
            "memory_context": None,
            "memory_updates": [],
            "enhanced_context": None,
            "system_prompt": "",
            "tools_used": [],
            "tool_results": {},
            "start_time": time.time(),
            "preprocessing_done": False,
            "agent_done": False,
            "postprocessing_done": False,
            "elder_mode": elder_mode,
            "requires_clarification": False,
            "health_check_needed": False,
            "platform": platform,
            "response_format": "text",
            "final_response": None,
            "metadata": {},
            "error": None,
            "retry_count": 0
        }
        
        try:
            # Stream events from the graph
            async for event in self.graph.astream(initial_state):
                # Process different event types
                if "agent" in event:
                    # Tool usage events
                    state = event["agent"]
                    if state.get("tools_used"):
                        for tool in state["tools_used"]:
                            yield {
                                "type": "tool",
                                "tool": tool,
                                "status": "completed"
                            }
                
                # Check for the final response
                if "postprocessor" in event:
                    state = event["postprocessor"]
                    if state.get("final_response"):
                        response = state["final_response"]["response"]
                        
                        # Stream the response in chunks
                        words = response.split()
                        current_chunk = ""
                        
                        for i, word in enumerate(words):
                            current_chunk += word + " "
                            
                            # Send chunk every 5 words or at the end
                            if (i + 1) % 5 == 0 or i == len(words) - 1:
                                yield {
                                    "type": "chunk",
                                    "chunk": current_chunk,
                                    "done": False
                                }
                                current_chunk = ""
                        
                        # Send memory update events
                        for update in state.get("memory_updates", []):
                            yield {
                                "type": "memory",
                                "action": "stored",
                                "fact": update.get("content", "")[:100]
                            }
            
            # Send end event
            yield {
                "type": "end",
                "done": True,
                "latency_ms": int((time.time() - initial_state["start_time"]) * 1000),
                "tokens_used": 0  # Would calculate in production
            }
            
        except Exception as e:
            self.logger.error(f"Error in streaming: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "done": True
            }