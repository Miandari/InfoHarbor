"""
Agent node implementation for the LangGraph workflow
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function

from src.agent.state import AgentState
from src.tools.base import tool_registry
from src.config.settings import settings


class AgentNode:
    """Main agent node that processes messages and uses tools"""
    
    def __init__(self):
        self.logger = logging.getLogger("agent.node")
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
            streaming=True
        )
        self._setup_tools()
        
    def _setup_tools(self):
        """Setup tools for the agent"""
        self.tools = tool_registry.to_langchain_tools()
        self.tool_executor = ToolExecutor(self.tools)
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind(
            functions=[convert_to_openai_function(t) for t in self.tools]
        )
        
    async def process(self, state: AgentState) -> AgentState:
        """Process the state and generate a response"""
        self.logger.info("Agent processing started")
        
        try:
            # Get messages from state
            messages = state["messages"]
            
            # Create prompt with proper message placeholders
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            # Get system prompt from state (set by preprocessor)
            system_prompt = messages[0].content if messages and messages[0].type == "system" else ""
            
            # Remove system message from messages list for processing
            if messages and messages[0].type == "system":
                messages = messages[1:]
            
            # Create the agent chain
            agent_chain = (
                {
                    "system_prompt": lambda x: system_prompt,
                    "messages": lambda x: messages
                }
                | prompt
                | self.llm_with_tools
                | OpenAIFunctionsAgentOutputParser()
            )
            
            # Run the agent
            result = await agent_chain.ainvoke({})
            
            # Handle the result
            if hasattr(result, 'tool'):
                # Agent wants to use a tool
                tool_name = result.tool
                tool_input = result.tool_input
                
                self.logger.info(f"Agent using tool: {tool_name}")
                
                # Execute the tool
                tool_result = await self._execute_tool(tool_name, tool_input, state)
                
                # Add tool usage to state
                state["tools_used"].append(tool_name)
                state["tool_results"][tool_name] = tool_result
                
                # Create tool message
                tool_message = ToolMessage(
                    content=str(tool_result.get("data", {}).get("formatted_response", tool_result)),
                    tool_call_id=result.tool_call_id if hasattr(result, 'tool_call_id') else tool_name
                )
                
                # Add tool message to messages
                state["messages"].append(tool_message)
                
                # Call agent again with tool result
                return await self.process(state)
                
            else:
                # Agent provided a final answer
                response_content = result.return_values.get("output", str(result))
                
                # Handle special cases for elderly users
                if state.get("elder_mode") and state.get("health_check_needed"):
                    response_content = self._add_health_check_response(response_content)
                
                # Create AI message
                ai_message = AIMessage(content=response_content)
                state["messages"].append(ai_message)
                
                # Mark agent processing as done
                state["agent_done"] = True
                
                self.logger.info("Agent processing completed")
                
        except Exception as e:
            self.logger.error(f"Error in agent processing: {e}")
            
            # Create error response
            error_response = self._create_error_response(e, state.get("elder_mode", True))
            ai_message = AIMessage(content=error_response)
            state["messages"].append(ai_message)
            state["error"] = str(e)
            state["agent_done"] = True
            
        return state
    
    async def _execute_tool(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        
        # Get the tool
        tool = tool_registry.get(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found",
                "data": None
            }
        
        # Add context from state
        context = {
            "user_preferences": self._extract_preferences_from_memory(state),
            "platform": state.get("platform", "api"),
            "elder_mode": state.get("elder_mode", True)
        }
        
        # Merge with any existing context
        if "context" in tool_input:
            context.update(tool_input["context"])
            
        tool_input["context"] = context
        
        # Execute the tool
        result = await tool.execute(
            query=tool_input.get("query", ""),
            context=context
        )
        
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "metadata": result.metadata
        }
    
    def _extract_preferences_from_memory(self, state: AgentState) -> List[str]:
        """Extract user preferences from memory context"""
        preferences = []
        
        memory_context = state.get("memory_context", {})
        if memory_context and "profile" in memory_context:
            profile = memory_context["profile"]
            
            # Extract preferences
            if profile.preferences:
                for pref in profile.preferences:
                    preferences.extend(pref.likes)
                    
            # Extract hobbies
            if profile.personal_info and profile.personal_info.hobbies:
                preferences.extend(profile.personal_info.hobbies)
                
        return preferences
    
    def _add_health_check_response(self, response: str) -> str:
        """Add health check follow-up to response"""
        health_followup = (
            "\n\nI noticed you mentioned something that might be health-related. "
            "If you're experiencing any discomfort or have health concerns, "
            "please don't hesitate to contact your doctor or healthcare provider. "
            "Is there anything specific about your health that you'd like to discuss?"
        )
        
        return response + health_followup
    
    def _create_error_response(self, error: Exception, elder_mode: bool) -> str:
        """Create a user-friendly error response"""
        if elder_mode:
            return (
                "I apologize, but I'm having a bit of trouble understanding or "
                "helping with that right now. Could you please try asking in a "
                "different way? Or perhaps we could talk about something else?"
            )
        else:
            return (
                f"I encountered an error while processing your request: {str(error)}. "
                "Please try again or rephrase your question."
            )