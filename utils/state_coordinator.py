"""
Utility functions for state coordination in the LangGraph workflow.

This module provides utilities to manage state references and prevent
duplication of messages between nodes.
"""

from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import copy
import hashlib
import uuid

def get_state_coordinator():
    """Get the singleton state coordinator."""
    return StateCoordinator()

class StateCoordinator:
    """
    Coordinates state between nodes to prevent message duplication.
    
    This class maintains a registry of node processing and ensures
    state transitions are clean between nodes.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateCoordinator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the state coordinator."""
        self.message_registry = {}
        self.node_registry = {}
        self.node_transitions = {}  # Track node transition counts
        self.transition_count = 0
    
    def register_message(self, message):
        """Register a message in the coordinator to track duplicates."""
        # Create a fingerprint for this message
        if hasattr(message, "content"):
            content = message.content
            msg_type = message.__class__.__name__
        else:
            content = str(message)
            msg_type = "unknown"
            
        fingerprint = f"{msg_type}:{content}"
        message_hash = hashlib.md5(fingerprint.encode()).hexdigest()
        
        # Register in the global registry
        if message_hash not in self.message_registry:
            self.message_registry[message_hash] = True
            return True
        return False
    
    def register_node_transition(self, from_node, to_node, state):
        """Register a node transition and sanitize state between nodes."""
        self.transition_count += 1
        
        # Deep copy the messages to prevent reference issues
        if "messages" in state:
            # Create a registry of existing messages in this state
            seen_contents = {}
            unique_messages = []
            
            for msg in state["messages"]:
                # Extract content
                if hasattr(msg, "content"):
                    content = msg.content
                else:
                    content = str(msg)
                    
                # Only include if not seen in this state
                if content not in seen_contents:
                    seen_contents[content] = True
                    unique_messages.append(copy.deepcopy(msg))
            
            # Replace with unique messages
            state = {**state, "messages": unique_messages}
        
        # Track the transition
        transition_id = f"{from_node}->{to_node}:{self.transition_count}"
        self.node_registry[transition_id] = len(state.get("messages", []))
        
        return state
    
    def purge_registry(self):
        """Reset the registry (useful for testing)."""
        self.message_registry = {}
        self.node_registry = {}
        self.transition_count = 0
    
    def wrap_node_for_coordination(self, node_func, node_name=None):
        """Wrap a node with our state coordination system."""
        if node_name is None:
            # Try to get the function name
            node_name = getattr(node_func, "__name__", "unknown_node")
            
        @wraps(node_func)
        def coordinated_node(state):
            # For diagnostic purposes, track the before state
            input_message_count = len(state.get("messages", []))
            
            # Sanitize input state
            sanitized_state = self._sanitize_state(state)
            incoming_sanitized_count = len(sanitized_state.get("messages", []))
            
            # If we cleaned up duplicates on input, log it
            if input_message_count != incoming_sanitized_count:
                print(f"[COORD] {node_name} input: {input_message_count} → {incoming_sanitized_count} messages")
            
            # Execute the original node
            result = node_func(sanitized_state)
            
            # Check for non-dict result
            if not isinstance(result, dict):
                print(f"[COORD] {node_name} returned non-dict result: {type(result)}")
                return result
                
            # Sanitize output state
            result_message_count = len(result.get("messages", []))
            sanitized_result = self._sanitize_state(result)
            outgoing_sanitized_count = len(sanitized_result.get("messages", []))
            
            # If we cleaned up duplicates on output, log it
            if result_message_count != outgoing_sanitized_count:
                print(f"[COORD] {node_name} output: {result_message_count} → {outgoing_sanitized_count} messages")
            
            # Register this node transition to track it
            self.node_transitions[node_name] = self.node_transitions.get(node_name, 0) + 1
            
            return sanitized_result
        
        return coordinated_node
    
    def _sanitize_state(self, state):
        """Sanitize state to prevent duplication."""
        if not isinstance(state, dict):
            print(f"[STATE_COORD] Warning: Non-dict state received: {type(state)}")
            from graph.transitions import StateTransitions
            return StateTransitions.create_clean_state()
            
        if "messages" not in state:
            return state
            
        # Create a registry of existing messages in this state
        seen_contents = {}
        unique_messages = []
        
        for msg in state["messages"]:
            # Extract content
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            else:
                content = str(msg)
                
            # Skip empty messages
            if not content or not content.strip():
                continue
                
            # Only include if not seen in this state
            if content not in seen_contents:
                seen_contents[content] = True
                # Use deep copy to prevent reference issues
                if hasattr(msg, "__dict__"):
                    unique_messages.append(copy.deepcopy(msg))
                else:
                    unique_messages.append(msg)
        
        # Replace with unique messages
        return {**state, "messages": unique_messages}