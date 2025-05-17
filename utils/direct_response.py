"""
Direct response utility that bypasses LangGraph for general knowledge questions.
This module provides a simple, direct way to get responses from OpenAI.
"""

import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
import time
import sys

# Global debug flag - default to False
DEBUG_MODE = False

def set_debug_mode(enabled=True):
    """Enable or disable debug logging"""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    if enabled:
        # Create a new log file or append to existing one with a separator
        with open('/Users/mk/Work/Elder_Care/LangGraph-tool-testing/debug_log.txt', 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n[{timestamp}] ========== DEBUG MODE ENABLED ==========\n")

# Add a logging function for debugging
def debug_log(message):
    """Write debug message to a log file if debug mode is enabled"""
    global DEBUG_MODE
    if not DEBUG_MODE:
        return  # Skip logging if debug mode is disabled
        
    with open('/Users/mk/Work/Elder_Care/LangGraph-tool-testing/debug_log.txt', 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def get_direct_answer(question: str) -> str:
    """
    Get a direct answer to a question using the OpenAI API.
    
    Args:
        question: The question to answer
        
    Returns:
        The answer as a string
    """
    try:
        debug_log(f"Received question: {question}")
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            debug_log("ERROR: No API key found")
            return "ERROR: OpenAI API key not found in environment variables."
        
        debug_log("API key found, creating client")
        
        # Create client
        client = OpenAI(api_key=api_key)
        
        # Make a simple, direct API call
        debug_log("Making API call with question")
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Answer questions directly and informatively. Never repeat the question back to the user."
                },
                {
                    "role": "user", 
                    "content": question
                }
            ]
        )
        
        # Get the response text
        response = completion.choices[0].message.content
        debug_log(f"API returned response: {response}")
        
        # Safety check: Don't return if it's just repeating the question
        if response.lower().strip() == question.lower().strip():
            debug_log("WARNING: Response matched question, using fallback")
            fallback = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "IMPORTANT: The user asked a question. Answer it directly WITHOUT repeating the question."
                    },
                    {
                        "role": "user",
                        "content": f"Question that needs a direct answer: {question}"
                    }
                ]
            )
            response = fallback.choices[0].message.content
            debug_log(f"Fallback response: {response}")
        
        # Add log right before returning
        debug_log(f"FINAL RESPONSE: {response}")
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        debug_log(f"ERROR in get_direct_answer: {e}\n{error_trace}")
        return f"I encountered a technical issue while processing your question. Error: {str(e)}"