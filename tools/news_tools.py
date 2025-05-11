"""
News search tools using Tavily API, wrapped for use with LangChain and LangGraph.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import TAVILY_API_KEY, TRUSTED_NEWS_DOMAINS, MAX_NEWS_RESULTS, DEFAULT_NEWS_DAYS_BACK

class NewsInput(BaseModel):
    topic: str = Field(..., description="Topic to get news updates about")
    days_back: int = Field(DEFAULT_NEWS_DAYS_BACK, description="How many days back to search for news (1-30)")
    max_results: int = Field(MAX_NEWS_RESULTS, description="Number of news articles to return (1-10)")

@tool(args_schema=NewsInput)
def get_recent_news(topic: str, days_back: int = DEFAULT_NEWS_DAYS_BACK, max_results: int = MAX_NEWS_RESULTS) -> dict:
    """Get recent news articles about a specific topic from the past few days."""
    # Initialize Tavily client
    tavily_api_key = TAVILY_API_KEY
    if not tavily_api_key:
        raise ValueError("Tavily API key is required")
    
    # Validate input parameters
    days_back = min(max(days_back, 1), 30)
    max_results = min(max(max_results, 1), 10)
    
    client = TavilyClient(api_key=tavily_api_key)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for search
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # Construct query with date range
    query = f"{topic} news after:{start_date_str} before:{end_date_str}"
    
    # Execute search
    search_results = client.search(
        query=query,
        search_depth="advanced",
        include_domains=TRUSTED_NEWS_DOMAINS,
        max_results=max_results,
        include_answer=True
    )
    
    # Format results
    articles = []
    if "results" in search_results:
        for result in search_results["results"]:
            # Extract date if possible
            date = "Unknown date"
            if result.get("published_date"):
                date = result["published_date"]
            
            articles.append({
                "title": result.get("title", "No title"),
                "url": result.get("url", ""),
                "source": result.get("source", "Unknown source"),
                "date": date,
                "snippet": result.get("content", "")[:200] + "..." if result.get("content") else "No content available"
            })
    
    # Add Tavily's generated answer if available
    summary = search_results.get("answer", "No summary available")
    
    return {
        "topic": topic,
        "time_range": f"{start_date_str} to {end_date_str}",
        "articles": articles,
        "article_count": len(articles),
        "summary": summary,
        "source": "Tavily Search API"
    }