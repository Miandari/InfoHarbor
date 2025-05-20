"""
Centralized state transition logic for the information assistant.

This module implements best practices for LangGraph state management by:
1. Centralizing all state transitions in one place
2. Using explicit transition functions with clear documentation
3. Supporting proper state versioning and history
"""

import operator
from typing import Dict, Any, List, Optional, Union, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from graph.state import InfoAssistantState

class StateTransitions:
    """
    Centralized state transition manager for the information assistant.
    
    This class provides static methods for handling all state transitions,
    ensuring consistent behavior across the application.
    """
    
    @staticmethod
    def transition_from_task(state: InfoAssistantState, new_query: str) -> InfoAssistantState:
        """
        Manage transitions between different tasks based on the current state and new user query.
        
        Args:
            state: Current state of the conversation
            new_query: The user's new query text
            
        Returns:
            Updated state with appropriate transitions applied
        """
        current_task = state.get("current_task")
        new_intent = StateTransitions.determine_intent(new_query)
        
        # Log the transition attempt for debugging
        try:
            from utils.direct_response import debug_log
            debug_log(f"STATE TRANSITION - From '{current_task}' to intent '{new_intent}' for query: {new_query[:50]}...")
        except:
            pass
            
        # Special handling for completed food orders
        if current_task == "food_order" and state.get("food_order_state") == "completed":
            # Always reset food order specific state after completion
            state = {
                **state,
                "food_order_state": None,
            }
            
        # CRITICAL FIX: Always reset when switching between different tools
        # This ensures we don't have state from one tool affecting another
        if current_task and new_intent and current_task != new_intent:
            # We're switching tools (e.g., from food to news), reset ALL tool-specific state
            try:
                from utils.direct_response import debug_log
                debug_log(f"STATE TRANSITION - Detected tool switch from '{current_task}' to '{new_intent}' - RESETTING ALL TOOL STATE")
            except:
                pass
                
            return {
                **state,
                "current_task": new_intent,  # IMPORTANT: Set to the new intent immediately
                "food_order_state": None,
                "tool_results": {},
                "pending_tools": [],
                "next_actions": [],
                "working_memory": {},
                "reflection": None,
                "last_tool_used": None  # Reset last tool used
            }
        
        # Default: return state with current_task set to the new intent
        # This ensures the intent is always updated
        return {**state, "current_task": new_intent}
    
    @staticmethod
    def handle_food_order_completion(state: InfoAssistantState, query: str, result: Any) -> InfoAssistantState:
        """
        Handle the completion of a food order, updating state appropriately.
        
        Args:
            state: Current state
            query: The user's order details
            result: Result from the food order processing
            
        Returns:
            Updated state with food order marked as completed
        """
        # Get existing food order history or initialize new list
        food_order_history = state.get("food_order_history", [])
        
        # Add new order to history
        food_order_history.append({"order_text": query})
        
        # Format a response
        response = "Thank you! I've sent your food order via Telegram. Here's what I sent:\n\n" + \
                  f"{query}\n\n" + \
                  "Your order has been submitted. Is there anything else you need help with?"
        
        # Update and return the state
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=query), AIMessage(content=response)],
            "current_task": None,  # Reset task to None after completion
            "last_tool_used": "food_order_tool",
            "food_order_state": None,  # Reset food order state
            "food_order_history": food_order_history,
            "tool_results": {
                "type": "food_order",
                "data": result,
                "pending": False
            },
            # Reset ReAct-related state to ensure clean transitions
            "next_actions": [],
            "pending_tools": [],
            "reflection": None,
            "working_memory": {}
        }
    
    @staticmethod
    def determine_intent(query: str) -> str:
        """
        Determine the user's intent from their message.
        This is the centralized intent detection function used across the application.
        
        Args:
            query: The user's message text
            
        Returns:
            String intent identifier: "food_order", "podcast", "news", or "general"
        """
        query_lower = query.lower()
        
        # Check for food ordering intents - EXPANDED to catch more phrases
        food_ordering_terms = [
            "order food", "order a pizza", "food delivery", "order pizza", 
            "get food", "place an order", "delivery", "takeout", "hungry",
            "want to order", "want food", "like to order", "need food",
            "food now", "order", "want to eat", "want pizza", "want to get food",
            "want dinner", "want lunch", "want breakfast",
            # Simple direct statements
            "i want to order food", "i want food", "order food now",
            "i'm hungry", "i am hungry", "get me food"
        ]
        
        # Use more advanced pattern matching for food intents
        if any(term in query_lower for term in food_ordering_terms):
            return "food_order"
        
        # More specific full phrase matching
        food_phrases = ["i want to order", "get me", "order some", "i need"]
        food_items = ["pizza", "burger", "food", "dinner", "lunch"]
        
        for phrase in food_phrases:
            if phrase in query_lower:
                for item in food_items:
                    if item in query_lower:
                        return "food_order"
    
        # Check for podcast-related intents
        podcast_terms = [
            "podcast", "episode", "listen", "audio show", "similar to", "recommend", 
            "series", "show me podcast", "find podcast"
        ]
        if any(term in query_lower for term in podcast_terms):
            return "podcast"
        
        # Check for news-related intents
        news_terms = [
            "news", "recent events", "happened recently", "latest", "update me", "headlines",
            "current events", "in the news", "world news", "breaking news", "important news",
            "most important", "what's happening", "what is happening", "tell me about"
        ]
        time_indicators = [
            "today", "this week", "this month", "recently", "last week", "last month", 
            "last day", "yesterday", "past few days"
        ]
        
        # Check for explicit news terms
        if any(term in query_lower for term in news_terms):
            return "news"
        
        # Check for time indicators combined with information seeking
        if any(term in query_lower for term in time_indicators) and (
            "about" in query_lower or 
            "what" in query_lower or 
            "tell me" in query_lower or
            "important" in query_lower or
            "happened" in query_lower or
            "going on" in query_lower
        ):
            return "news"
        
        # Default to general query
        return "general"
    
    @staticmethod
    def add_to_history(state: InfoAssistantState, history_key: str, item: Any) -> InfoAssistantState:
        """
        Add an item to a history list in the state.
        
        Args:
            state: Current state
            history_key: The key for the history list (e.g., "podcast_history")
            item: The item to add to history
            
        Returns:
            Updated state with item added to the specified history
        """
        existing_history = state.get(history_key, [])
        return {
            **state,
            history_key: existing_history + [item]
        }
    
    @staticmethod
    def create_clean_state(previous_state: Optional[InfoAssistantState] = None) -> InfoAssistantState:
        """
        Create a clean state, optionally based on a previous state.
        
        Args:
            previous_state: Optional previous state to inherit conversation history from
            
        Returns:
            A clean state with reinitialized fields
        """
        if previous_state:
            # Keep conversation history and user memory but reset task-specific state
            return {
                **previous_state,
                "current_task": None,
                "food_order_state": None,
                "tool_results": {},
                "next_actions": [],
                "working_memory": {},
                "pending_tools": [],
                "reflection": None
            }
        else:
            # Create completely new state with a default user ID
            default_user_id = f"user_{uuid.uuid4().hex[:8]}"
            
            return {
                "messages": [],
                "podcast_history": [],
                "news_history": [],
                "food_order_history": [],
                "food_order_state": None,
                "current_task": None,
                "last_tool_used": None,
                "user_id": default_user_id,  # Add default user ID
                "user_memory": {},           # Initialize empty user memory
                "memory_updates": [],        # Initialize empty memory updates
                "context": {},
                "tool_results": {},
                "reasoning": [],
                "next_actions": [],
                "working_memory": {},
                "pending_tools": [],
                "reflection": None,
                "state_version": 1
            }
    
    @staticmethod
    def identify_user(state: InfoAssistantState, user_id: str) -> InfoAssistantState:
        """
        Associate a specific user ID with the current conversation state.
        
        Args:
            state: Current state
            user_id: User identifier
            
        Returns:
            Updated state with user ID
        """
        return {
            **state,
            "user_id": user_id
        }
    
    @staticmethod
    def add_memory_item(state: InfoAssistantState, memory_item: Dict[str, Any]) -> InfoAssistantState:
        """
        Add a memory item to the memory updates list.
        
        Args:
            state: Current state
            memory_item: Memory item with type, content, etc.
            
        Returns:
            Updated state with memory item added to updates
        """
        memory_updates = state.get("memory_updates", [])
        return {
            **state,
            "memory_updates": memory_updates + [memory_item]
        }
    
    @staticmethod
    def add_memory_context_to_reasoning(state: InfoAssistantState) -> InfoAssistantState:
        """
        Add memory context to the agent's reasoning chain.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with memory context in reasoning
        """
        # Get memory context from state
        memory_context = state.get("context", {}).get("memory_context", "")
        if not memory_context:
            return state
            
        # Add memory context to reasoning
        reasoning = state.get("reasoning", [])
        reasoning.append(f"RELEVANT USER MEMORY:\n{memory_context}")
        
        return {
            **state,
            "reasoning": reasoning
        }
        
    @staticmethod
    def handle_error_state(state: InfoAssistantState, error_message: str, 
                          tool_type: Optional[str] = None) -> InfoAssistantState:
        """
        Create a standardized error state.
        
        Args:
            state: Current state
            error_message: The error message to store
            tool_type: Optional tool type that caused the error
            
        Returns:
            Updated state with error information and clean pending_tools
        """
        # Keep track of the error in working memory
        working_memory = state.get("working_memory", {})
        working_memory["last_error"] = error_message
        
        # Add tool-specific error if provided
        if tool_type:
            working_memory[f"{tool_type}_error"] = error_message
        
        # Format a friendly error message for the user
        user_friendly_error = "I'm sorry, but I encountered a technical issue while processing your request. "
        
        if tool_type == "food_order":
            user_friendly_error += "There was a problem with the food ordering system. Let's try again with more details."
        elif tool_type == "podcast":
            user_friendly_error += "I couldn't retrieve the podcast information you requested. Would you like to try a different search?"
        elif tool_type == "news":
            user_friendly_error += "I couldn't access the latest news at the moment. Please try again in a moment."
        elif tool_type == "memory":
            # Don't expose memory errors to the user, just continue the conversation
            user_friendly_error = "I'd be happy to continue our conversation. What else can I help you with?"
        else:
            user_friendly_error += "Let me try a different approach to help you."
        
        # Add the error response to messages
        messages = state.get("messages", [])
        messages.append(AIMessage(content=user_friendly_error))
        
        return {
            **state,
            "messages": messages,
            "tool_results": {
                "type": tool_type or "general",
                "error": error_message,
                "pending": False
            },
            "working_memory": working_memory,
            "pending_tools": [],  # Always clear pending tools on error
            "next_actions": []    # Clear any planned next actions
        }
        
    @staticmethod
    def verify_tool_completion(state: InfoAssistantState, tool_name: str) -> bool:
        """
        Verify that a tool has been properly removed from pending_tools upon completion.
        
        Args:
            state: Current state
            tool_name: The name of the tool to verify
            
        Returns:
            Boolean indicating whether the tool is properly completed (not in pending_tools)
        """
        pending_tools = state.get("pending_tools", [])
        tool_base_name = tool_name.replace("_tools", "")
        
        return tool_base_name not in pending_tools