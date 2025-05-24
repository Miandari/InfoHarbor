#!/usr/bin/env python3
"""
Local testing script for the Elderly Assistant Agent
Run this to test the agent interactively in your terminal
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.graph import ElderlyAssistantAgent
from src.settings import settings
import logging

# Configure logging for console
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

async def main():
    """Interactive chat session"""
    print("🏠 Elder Care Assistant - Local Testing")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'help' to see available commands")
    print()
    
    # Check for OpenAI API key
    if not settings.openai_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        print("or create a .env file with OPENAI_API_KEY=your-key-here")
        return
    
    # Initialize agent
    print("🤖 Initializing agent...")
    agent = ElderlyAssistantAgent()
    
    # Test user ID
    user_id = "test_user_123"
    conversation_id = None
    
    print("✅ Agent ready! You can start chatting.")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Take care!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            if not user_input:
                continue
            
            # Process message
            print("🤔 Thinking...")
            
            response = await agent.chat(
                message=user_input,
                user_id=user_id,
                conversation_id=conversation_id,
                platform="terminal",
                elder_mode=True
            )
            
            # Update conversation ID
            conversation_id = response.get("conversation_id")
            
            # Display response
            print(f"\n🏠 Assistant: {response['response']}")
            
            # Show metadata if requested
            metadata = response.get("metadata", {})
            if metadata.get("tools_used"):
                print(f"🔧 Tools used: {', '.join(metadata['tools_used'])}")
            
            latency = metadata.get("latency_ms", 0)
            print(f"⏱️  Response time: {latency}ms")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Take care!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def print_help():
    """Print help information"""
    print()
    print("📋 Available Commands:")
    print("  help     - Show this help message")
    print("  clear    - Clear the screen")
    print("  quit/exit- End the session")
    print()
    print("💡 Try asking about:")
    print("  • Health reminders or medication schedules")
    print("  • Podcast recommendations")
    print("  • Current news")
    print("  • General conversation")
    print()


if __name__ == "__main__":
    asyncio.run(main())