"""
Utility for direct response handling outside the LangGraph workflow
"""
import os
import time
from typing import Optional
from langchain_openai import ChatOpenAI

# Global debug flag
DEBUG_MODE = False
VERBOSE_MODE = False  # New flag to control print statements

def set_debug_mode(enabled=True):
    """Enable or disable debug logging"""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    if enabled:
        # Create a new log file or append to existing one with a separator
        with open('/Users/mk/Work/Elder_Care/LangGraph-tool-testing/debug_log.txt', 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n[{timestamp}] ========== DEBUG MODE ENABLED ==========\n")

def set_verbose_mode(enabled=True):
    """Enable or disable verbose console output"""
    global VERBOSE_MODE
    VERBOSE_MODE = enabled

def verbose_print(message):
    """Print message only if verbose mode is enabled"""
    global VERBOSE_MODE
    if VERBOSE_MODE:
        print(message)

def debug_log(message):
    """Write debug message to a log file if debug mode is enabled"""
    global DEBUG_MODE
    if not DEBUG_MODE:
        return  # Skip logging if debug mode is disabled
        
    with open('/Users/mk/Work/Elder_Care/LangGraph-tool-testing/debug_log.txt', 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def get_direct_response(message: str) -> Optional[str]:
    """
    Check if a message should get a direct response without going through the LangGraph workflow.
    This is used for handling simple queries that don't need the full workflow.
    
    Args:
        message: The user's message
        
    Returns:
        A direct response string, or None if the message should be processed by the workflow
    """
    # Convert to lowercase for easier matching
    message_lower = message.lower().strip()
    
    # Handle direct commands and simple questions
    if message_lower in ["hello", "hi", "hey"]:
        return "Hello! How can I assist you today?"
    
    elif message_lower in ["bye", "goodbye", "exit"]:
        return "Goodbye! Feel free to come back if you need any assistance."
    
    elif message_lower in ["thank you", "thanks"]:
        return "You're welcome! Is there anything else you need help with?"
    
    elif message_lower == "help":
        return ("I can help you with several things:\n"
                "- Get news on specific topics\n"
                "- Find podcast recommendations\n"
                "- Order food\n"
                "- Answer general questions\n\n"
                "Just let me know what you're interested in!")
                
    # No direct response needed, let the workflow handle it
    return None

def get_direct_answer(question: str, model_name: Optional[str] = None) -> str:
    """
    Get a direct answer to a question using the OpenAI API.
    This is a fallback for when we need a direct response outside the LangGraph workflow.
    
    Args:
        question: The user's question
        model_name: Optional model name to use instead of default
        
    Returns:
        The answer from the model
    """
    debug_log(f"Received question: {question}")
    
    # Check if this is a food ordering request
    question_lower = question.lower()
    food_order_keywords = ["order food", "order a pizza", "food delivery", "order pizza", "get food", 
                          "place an order", "hungry", "deliver food", "takeout", "food order",
                          "pizza", "want to order", "want pizza"]
                          
    is_food_order = any(keyword in question_lower for keyword in food_order_keywords) or \
                    ("want" in question_lower and "pizza" in question_lower)
                    
    if is_food_order:
        debug_log("Food ordering intent detected in direct_response, returning structured prompt")
        return "I'd be happy to help you order food! Please provide your complete order details including:\n\n" + \
               "1. Restaurant name\n" + \
               "2. Items you'd like to order (including quantities)\n" + \
               "3. Delivery or pickup preference\n" + \
               "4. Delivery address (if applicable)\n" + \
               "5. Any special instructions"
    
    # Check if we have an API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        debug_log("No API key found, returning default response")
        return "I'm unable to process your request at this time. Please make sure the API keys are configured properly."
    
    debug_log("API key found, creating client")
    
    try:
        # Use provided model or default to the one in config
        if not model_name:
            from config import DEFAULT_MODEL
            model_name = DEFAULT_MODEL
        
        # Create a client
        client = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key
        )
        
        debug_log("Making API call with question")
        response = client.invoke(
            [{"role": "user", "content": question}]
        )
        
        debug_log(f"API returned response: {response.content}")
        return response.content
        
    except Exception as e:
        debug_log(f"Error in get_direct_answer: {str(e)}")
        import traceback
        debug_log(traceback.format_exc())
        
        # Return a generic response in case of error
        return "I'm sorry, I encountered an error processing your request. Please try again later."