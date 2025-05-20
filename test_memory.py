#!/usr/bin/env python3
"""
Test script to demonstrate the memory system functionality.
This script runs through a multi-turn conversation to show how the
assistant remembers user information across turns.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from main import run_info_assistant
from utils.direct_response import set_debug_mode

def memory_demo(user_id=None, debug=False):
    """
    Runs a demonstration of the memory functionality.
    
    Args:
        user_id: Optional user ID to continue an existing conversation
        debug: Whether to enable debug mode
    """
    # Explicitly set debug mode based on flag
    set_debug_mode(debug)
    if debug:
        print("Debug mode enabled. Details will be logged to debug_log.txt")
    else:
        # Make sure debug mode is explicitly turned off
        set_debug_mode(False)

    # Start with a clean state
    state = None
    
    # Generate user ID if not provided
    if not user_id:
        import uuid
        user_id = f"demo_user_{uuid.uuid4().hex[:6]}"
    
    print(f"=== Memory System Demonstration ===")
    print(f"User ID: {user_id}")
    print("This demo will show how the assistant remembers information across turns.\n")
    
    # Demo conversations - designed to demonstrate memory capabilities
    demo_conversations = [
        # Introduction and name extraction
        "Hi, my name is Sarah. I'd like to know more about your memory capabilities.",
        
        # Preference extraction
        "I really love Italian food, especially pasta. But I don't like spicy food much.",
        
        # Facts about user
        "I'm originally from Boston and I work as a software engineer. I've been in this field for about 7 years.",
        
        # Test if name is remembered
        "Do you remember my name?",
        
        # Test if preferences are remembered
        "What kind of food do I like?",
        
        # Test multiple memory types
        "Could you summarize what you know about me?",
        
        # Add more preferences
        "I also enjoy hiking on weekends and listening to jazz music.",
        
        # Check updated preferences
        "What are my hobbies?",
        
        # Add factual information
        "I recently adopted a golden retriever puppy named Max.",
        
        # Final memory test
        "Tell me everything you remember about me."
    ]
    
    # Run through the demo conversations
    for i, query in enumerate(demo_conversations):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {query}")
        
        # Process query with memory - pass the debug flag to run_info_assistant
        response, updated_state = run_info_assistant(query, state, user_id=user_id, debug=debug)
        state = updated_state
        
        print(f"Assistant: {response}")
        
        # Brief pause between turns
        if i < len(demo_conversations) - 1:
            input("\nPress Enter to continue to next turn...")
    
    print("\n=== Demo Complete ===")
    print(f"The assistant has built a memory profile for user: {user_id}")
    print("You can continue this conversation in interactive mode:")
    print(f"python main.py --interactive  # Then enter user ID: {user_id}")
    print("\nOr run a single query with this user's memory:")
    print(f"python main.py --user {user_id} \"Tell me more about podcasts\"")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Memory system demonstration')
    parser.add_argument('--user', type=str, help='User ID to use for the demo')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Run the demo
    memory_demo(user_id=args.user, debug=args.debug)