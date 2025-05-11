"""
Configuration settings for the Information Assistant application.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LISTENNOTES_API_KEY = os.getenv("LISTENNOTES_API_KEY")

# LLM Configuration
DEFAULT_MODEL = "gpt-4o"
TEMPERATURE = 0

# Tool Configuration
MAX_PODCAST_RESULTS = 10
MAX_NEWS_RESULTS = 5
DEFAULT_NEWS_DAYS_BACK = 7

# List of trusted news domains
TRUSTED_NEWS_DOMAINS = [
    "nytimes.com", "cnn.com", "bbc.com", "reuters.com", "apnews.com", 
    "wsj.com", "theguardian.com", "nbcnews.com", "cbsnews.com"
]

# List of trusted podcast domains 
PODCAST_DOMAINS = [
    "listennotes.com",
    "podcastreview.org", 
    "podchaser.com",
    "chartable.com", 
    "goodpods.com",
    "player.fm",
    "podcastaddict.com"
]