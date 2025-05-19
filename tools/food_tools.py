"""
Food ordering tools for the LangGraph-based assistant.
This module provides tools to handle food ordering requests and process them.
"""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

# Import the base structures from the podcast tools for consistency
from tools.podcast_tools import BaseArgs, ToolResponse

# Define the input schema for food ordering
class FoodOrderInput(BaseModel):
    order_text: str = Field(..., description="The full text of the food order from the user")

# Define the arguments schema for food ordering that extends BaseArgs
class FoodOrderArgs(BaseArgs):
    """Input schema for food ordering."""
    order_text: str = Field(..., description="The full text of the food order from the user")

# Food ordering tool class
class FoodOrderingTool:
    """Tool for processing food orders."""
    
    def __init__(self):
        """Initialize the food ordering tool."""
        # You might initialize telegram client or other API clients here
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def run(self, args: FoodOrderArgs) -> ToolResponse:
        """
        Process a food order.
        
        Args:
            args: Arguments for the food ordering tool
        
        Returns:
            ToolResponse with order confirmation
        """
        try:
            # Format the order details for Telegram
            message = self._format_telegram_message(args.order_text)
            
            # Send the message to Telegram
            sent = self._send_telegram_message(message)
            if not sent:
                raise Exception("Failed to send message to Telegram")
            
            # Return successful response
            return ToolResponse(
                ok=True,
                content={
                    "message": "Order sent successfully",
                    "telegram_message": message
                },
                error=None
            )
        except Exception as e:
            # Return error response
            return ToolResponse(
                ok=False,
                content=None,
                error=f"Error processing food order: {str(e)}"
            )
    
    def _format_telegram_message(self, order_text: str) -> str:
        """Format the order details as a Telegram message."""
        message = f"🍕 NEW FOOD ORDER 🍕\n\n"
        message += f"Order Details:\n{order_text}\n\n"
        message += f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return message
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send a message to Telegram."""
        import requests
        
        if not self.telegram_bot_token or not self.telegram_chat_id:
            print("WARNING: Telegram credentials not found in environment variables")
            return False
        
        # Set up the API endpoint
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        
        # Prepare the payload
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"  # Supports basic formatting
        }
        
        try:
            # Send the request
            response = requests.post(url, json=payload)
            response_data = response.json()
            
            # Log success or failure
            if response.status_code == 200 and response_data.get("ok"):
                print(f"Successfully sent message to Telegram chat ID: {self.telegram_chat_id}")
                return True
            else:
                error_msg = response_data.get("description", "Unknown error")
                print(f"Failed to send Telegram message: {error_msg}")
                return False
                
        except Exception as e:
            print(f"Exception while sending Telegram message: {str(e)}")
            return False

# Create the food ordering tool
food_ordering_tool = FoodOrderingTool()

@tool
def process_food_order(order_text: str) -> Dict[str, Any]:
    """Process a food order and send it via Telegram."""
    args = FoodOrderArgs(
        trace_id="food-order-request",
        order_text=order_text
    )
    
    result = food_ordering_tool.run(args)
    
    if result.ok:
        return result.content
    else:
        raise ValueError(result.error)

# Export the food ordering tool
food_tools = [process_food_order]