"""
Memory-related nodes for the LangGraph workflow.

This module provides nodes that:
1. Load user memories at the start of a conversation
2. Update memory with new information from user interactions
3. Handle memory reflection and summarization
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import uuid
import os
import json
from datetime import datetime

from memory.memory_manager import MemoryManager
from graph.state import InfoAssistantState

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get (or initialize) the memory manager singleton."""
    global _memory_manager
    if _memory_manager is None:
        storage_path = os.getenv("MEMORY_STORAGE_PATH", "./memory_storage")
        _memory_manager = MemoryManager(storage_path=storage_path)
    return _memory_manager

def memory_retrieval_node(state: InfoAssistantState) -> InfoAssistantState:
    """
    Retrieve relevant user memories and add them to the conversation state.
    
    This node runs at the start of a conversation to provide memory context.
    
    Args:
        state: Current conversation state
        
    Returns:
        State enriched with user memory context
    """
    try:
        # Get user ID from state or generate a placeholder for testing
        user_id = state.get("user_id", "default_user")
        
        # Get the memory manager
        memory_manager = get_memory_manager()
        
        # Use memory manager's debug_print instead of direct prints
        memory_manager.debug_print(f"Memory retrieval for user_id: {user_id}")
        
        # Get conversation context from messages
        messages = state.get("messages", [])
        memory_manager.debug_print(f"Using {len(messages)} messages for memory context")
        
        # Retrieve memories relevant to the conversation
        try:
            memories = memory_manager.retrieve_memories(
                user_id=user_id,
                conversation_context=messages[-5:] if len(messages) > 5 else messages,
                limit=10
            )
            
            memory_manager.debug_print(f"Retrieved memories type: {type(memories)}")
            memory_manager.debug_print(f"Retrieved memories content preview: {str(memories)[:100]}")
            
            # Check memory structure to verify it's properly formed
            if isinstance(memories, dict):
                memory_manager.debug_print(f"Memory structure keys: {list(memories.keys())}")
                
                # Check if identity was retrieved
                if "identity" in memories:
                    memory_manager.debug_print(f"Identity info retrieved: {memories['identity']}")
                else:
                    memory_manager.debug_print(f"No identity information in memories")
            else:
                memory_manager.debug_print(f"Unexpected memories format - not a dict: {type(memories)}")
                
        except Exception as retrieval_error:
            import traceback
            memory_manager.debug_print(f"Memory retrieval specific error: {retrieval_error}")
            memory_manager.debug_print(f"Memory retrieval error traceback: {traceback.format_exc()}")
            # Create empty memory structure as fallback
            memories = {
                "identity": {},
                "preferences": {"likes": [], "dislikes": []},
                "important_facts": [],
                "conversation_history": [],
            }
        
        # Format memories as context string
        try:
            memory_context = memory_manager.format_memory_for_context(memories)
            memory_manager.debug_print(f"Memory context generated, length: {len(memory_context)}")
        except Exception as format_error:
            import traceback
            memory_manager.debug_print(f"Memory formatting error: {format_error}")
            memory_manager.debug_print(f"Memory formatting traceback: {traceback.format_exc()}")
            memory_context = "Error retrieving memory content."
        
        # Update state with memory information
        return {
            **state,
            "user_memory": memories,
            "context": {
                **(state.get("context", {})),
                "memory_context": memory_context
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in memory retrieval: {e}")
        print(f"Memory retrieval traceback: {traceback.format_exc()}")
        # Return unmodified state on error
        return state

def memory_extraction_node(state: InfoAssistantState) -> InfoAssistantState:
    """
    Extract potential memory items from the conversation.
    
    This node runs after user interactions to identify new information
    to remember about the user.
    
    Args:
        state: Current conversation state
        
    Returns:
        State with extracted memory items
    """
    try:
        # Get user ID from state or use default
        user_id = state.get("user_id", "default_user")
        
        # Get the memory manager
        memory_manager = get_memory_manager()
        
        # Get conversation messages
        messages = state.get("messages", [])
        if len(messages) < 2:  # Need at least user + assistant message
            return state
        
        # Extract memory items from recent conversation
        # Use last 10 messages maximum to avoid token limits
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        extraction_result = memory_manager.extract_memory_items(
            user_id=user_id,
            messages=recent_messages
        )
        
        # Get extracted items
        extracted_items = extraction_result.get("extracted_items", [])
        
        # Update state with memory extraction results
        return {
            **state,
            "memory_updates": state.get("memory_updates", []) + extracted_items
        }
    except Exception as e:
        print(f"Error in memory extraction: {e}")
        # Return unmodified state on error
        return state

def memory_update_node(state: InfoAssistantState) -> InfoAssistantState:
    """
    Update user memory with extracted items and conversation summary.
    
    This node runs at the end of a conversation to persist memory updates.
    
    Args:
        state: Current conversation state
        
    Returns:
        State with updated memory
    """
    try:
        # Get user ID from state or use default
        user_id = state.get("user_id", "default_user")
        
        # Get the memory manager
        memory_manager = get_memory_manager()
        
        # Get memory updates
        memory_updates = state.get("memory_updates", [])
        
        # Update memory with extracted items
        if memory_updates:
            memory_manager.update_memory(
                user_id=user_id,
                extracted_items=memory_updates
            )
        
        # Store conversation summary
        messages = state.get("messages", [])
        if len(messages) >= 2:  # Need at least user + assistant message
            memory_manager.store_conversation_summary(
                user_id=user_id,
                messages=messages
            )
        
        # Clear memory updates from state after persisting
        return {
            **state,
            "memory_updates": []
        }
    except Exception as e:
        print(f"Error in memory update: {e}")
        # Return unmodified state on error
        return {**state, "memory_updates": []}

def add_memory_context_to_prompt(state: InfoAssistantState) -> InfoAssistantState:
    """
    Prepare memory context for use in LLM prompts.
    
    This node formats memory information to be included in prompts.
    
    Args:
        state: Current conversation state
        
    Returns:
        State with memory context ready for prompting
    """
    try:
        # Get memory context from state
        memory_context = state.get("context", {}).get("memory_context", "")
        
        # If we have memory context, add it to the state's reasoning field
        # This will be used by the agent to include memory in its reasoning
        if memory_context:
            reasoning = state.get("reasoning", [])
            
            if not reasoning:
                # Initial reasoning with memory context
                reasoning.append(f"MEMORY CONTEXT:\n{memory_context}\n\nI should use this memory context to personalize my responses.")
            else:
                # Add memory context as a new reasoning step
                reasoning.append(f"RELEVANT USER MEMORY:\n{memory_context}")
            
            return {
                **state,
                "reasoning": reasoning
            }
        
        # No change if no memory context
        return state
        
    except Exception as e:
        print(f"Error adding memory context: {e}")
        # Return unmodified state on error
        return state