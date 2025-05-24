"""
Agent node implementation for the LangGraph workflow
"""
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from src.agent.state import AgentState
from src.tools.base import tool_registry
from src.settings import settings


class AgentNode:
    """Main agent node that processes messages and uses tools"""
    
    def __init__(self):
        self.logger = logging.getLogger("agent.node")
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
            streaming=True
        )
        self._setup_tools()
        
    def _setup_tools(self):
        """Setup tools for the agent"""
        self.tools = tool_registry.to_langchain_tools()
        self.tool_node = ToolNode(self.tools)
        
        # Create prompt for the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
    async def process(self, state: AgentState) -> AgentState:
        """Process the state and generate a response"""
        self.logger.info("Agent processing started")
        
        try:
            # Get messages from state
            messages = state["messages"]
            
            # Extract human message and system prompt
            human_message = ""
            system_prompt = ""
            chat_history = []
            
            for msg in messages:
                if msg.type == "system":
                    system_prompt = msg.content
                elif msg.type == "human":
                    human_message = msg.content
                else:
                    chat_history.append(msg)
            
            # Prepare input for agent
            agent_input = {
                "input": human_message,
                "system_prompt": system_prompt,
                "chat_history": chat_history,
                "user_preferences": self._extract_preferences_from_memory(state),
                "elder_mode": state.get("elder_mode", True)
            }
            
            # Run the agent
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Extract the output
            output = result.get("output", "I apologize, but I couldn't process your request.")
            
            # Handle special cases for elderly users
            if state.get("elder_mode") and state.get("health_check_needed"):
                output = self._add_health_check_response(output)
            
            # Create AI message
            ai_message = AIMessage(content=output)
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