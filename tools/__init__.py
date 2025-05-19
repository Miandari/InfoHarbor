"""
Tools module initialization.
"""

from tools.podcast_tools import podcast_tools
from tools.news_tools import get_recent_news
from tools.food_tools import food_tools

# Export all tools
all_tools = podcast_tools + [get_recent_news] + food_tools