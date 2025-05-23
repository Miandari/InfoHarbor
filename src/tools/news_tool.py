"""
News discovery tool following the consistent pattern
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.tools.base import BaseTool, ToolResult


class NewsDiscoveryTool(BaseTool):
    """Tool for fetching and summarizing news based on user interests"""
    
    @property
    def name(self) -> str:
        return "news_discovery"
    
    @property
    def description(self) -> str:
        return "Search for and summarize current news articles based on topics or interests"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The news topic or search query"
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["world", "politics", "technology", "health", "science", "sports", "entertainment", "business", "local"]
                            },
                            "description": "News categories to search within"
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["today", "week", "month"],
                            "description": "How recent the news should be"
                        },
                        "complexity": {
                            "type": "string",
                            "enum": ["simple", "detailed"],
                            "description": "Level of detail in summaries"
                        },
                        "avoid_topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Topics to avoid (e.g., distressing content)"
                        }
                    }
                }
            },
            "required": ["query"]
        }
    
    async def _execute_impl(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Search for news based on query"""
        
        # Extract context parameters
        categories = context.get("categories", ["world", "politics", "technology"]) if context else []
        time_range = context.get("time_range", "week") if context else "week"
        complexity = context.get("complexity", "simple") if context else "simple"
        avoid_topics = context.get("avoid_topics", []) if context else []
        
        # For MVP, return mock news data
        # In production, integrate with news APIs like NewsAPI or RSS feeds
        mock_articles = self._get_mock_news()
        
        # Filter by query
        query_lower = query.lower()
        filtered = [
            article for article in mock_articles
            if query_lower in article["title"].lower()
            or query_lower in article["summary"].lower()
            or any(query_lower in cat.lower() for cat in article["categories"])
        ]
        
        # Filter by categories if specified
        if categories:
            filtered = [
                article for article in filtered
                if any(cat in article["categories"] for cat in categories)
            ]
        
        # Filter by time range
        filtered = self._filter_by_time(filtered, time_range)
        
        # Filter out avoided topics
        if avoid_topics:
            filtered = [
                article for article in filtered
                if not any(
                    topic.lower() in article["title"].lower()
                    or topic.lower() in article["summary"].lower()
                    for topic in avoid_topics
                )
            ]
        
        # Sort by date (newest first)
        filtered.sort(key=lambda x: x["published_date"], reverse=True)
        
        # Take top 5 results
        results = filtered[:5]
        
        if results:
            # Format for elderly users
            formatted_response = self._format_news_for_elderly(
                results, 
                complexity=complexity
            )
            
            return ToolResult(
                success=True,
                data={
                    "articles": results,
                    "formatted_response": formatted_response,
                    "total_found": len(filtered),
                    "search_query": query
                },
                metadata={
                    "source": "mock_data",  # In production: "newsapi"
                    "filtered_count": len(filtered),
                    "time_range": time_range,
                    "categories": categories
                }
            )
        else:
            return ToolResult(
                success=True,
                data={
                    "articles": [],
                    "formatted_response": f"I couldn't find any recent news about '{query}'. Would you like me to search for something else?",
                    "total_found": 0,
                    "search_query": query
                }
            )
    
    def _get_mock_news(self) -> List[Dict[str, Any]]:
        """Get mock news data for MVP"""
        base_date = datetime.now()
        
        return [
            {
                "title": "New Breakthrough in Alzheimer's Research Shows Promise",
                "summary": "Scientists have discovered a new approach to treating Alzheimer's disease that shows early promise in clinical trials.",
                "categories": ["health", "science"],
                "source": "Health News Daily",
                "author": "Dr. Sarah Johnson",
                "published_date": (base_date - timedelta(hours=5)).isoformat(),
                "url": "https://example.com/alzheimers-breakthrough",
                "reading_time": 3,
                "sentiment": "positive"
            },
            {
                "title": "Local Community Center Announces New Senior Programs",
                "summary": "The downtown community center is launching new programs specifically designed for seniors, including gentle exercise classes and social gatherings.",
                "categories": ["local", "community"],
                "source": "Local Times",
                "author": "Mary Smith",
                "published_date": (base_date - timedelta(days=1)).isoformat(),
                "url": "https://example.com/senior-programs",
                "reading_time": 2,
                "sentiment": "positive"
            },
            {
                "title": "Technology Companies Develop Easier Smartphones for Seniors",
                "summary": "Major tech companies are introducing smartphones with larger buttons, clearer displays, and simplified interfaces designed for older adults.",
                "categories": ["technology", "business"],
                "source": "Tech Today",
                "author": "John Davis",
                "published_date": (base_date - timedelta(days=2)).isoformat(),
                "url": "https://example.com/senior-smartphones",
                "reading_time": 4,
                "sentiment": "positive"
            },
            {
                "title": "Weather Update: Mild Week Ahead with Possible Rain Thursday",
                "summary": "Meteorologists predict comfortable temperatures this week with a chance of light rain on Thursday afternoon.",
                "categories": ["weather", "local"],
                "source": "Weather Central",
                "author": "Bob Wilson",
                "published_date": base_date.isoformat(),
                "url": "https://example.com/weather-update",
                "reading_time": 1,
                "sentiment": "neutral"
            },
            {
                "title": "Medicare Announces New Coverage Options for 2025",
                "summary": "Medicare has announced expanded coverage options for 2025, including better prescription drug coverage and telehealth services.",
                "categories": ["health", "politics"],
                "source": "Government News",
                "author": "Lisa Brown",
                "published_date": (base_date - timedelta(days=3)).isoformat(),
                "url": "https://example.com/medicare-updates",
                "reading_time": 5,
                "sentiment": "positive"
            },
            {
                "title": "Study Shows Benefits of Daily Walking for Heart Health",
                "summary": "A new study confirms that walking just 30 minutes a day can significantly improve heart health and reduce the risk of cardiovascular disease.",
                "categories": ["health", "science"],
                "source": "Medical Journal",
                "author": "Dr. Robert Lee",
                "published_date": (base_date - timedelta(days=4)).isoformat(),
                "url": "https://example.com/walking-benefits",
                "reading_time": 3,
                "sentiment": "positive"
            }
        ]
    
    def _filter_by_time(
        self, 
        articles: List[Dict[str, Any]], 
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Filter articles by time range"""
        now = datetime.now()
        
        if time_range == "today":
            cutoff = now - timedelta(days=1)
        elif time_range == "week":
            cutoff = now - timedelta(days=7)
        else:  # month
            cutoff = now - timedelta(days=30)
        
        return [
            article for article in articles
            if datetime.fromisoformat(article["published_date"]) > cutoff
        ]
    
    def _format_news_for_elderly(
        self, 
        articles: List[Dict[str, Any]], 
        complexity: str = "simple"
    ) -> str:
        """Format news articles in an elderly-friendly way"""
        if not articles:
            return "I couldn't find any news articles on that topic."
        
        response_parts = [
            f"Here are the latest news articles I found for you:\n"
        ]
        
        for i, article in enumerate(articles, 1):
            # Calculate how long ago it was published
            published = datetime.fromisoformat(article["published_date"])
            time_ago = self._get_friendly_time_ago(published)
            
            response_parts.append(f"\n{i}. **{article['title']}**")
            
            if complexity == "simple":
                # Simple summary for easier reading
                response_parts.append(f"   - {article['summary']}")
                response_parts.append(f"   - From: {article['source']} ({time_ago})")
            else:
                # More detailed information
                response_parts.append(f"   - Summary: {article['summary']}")
                response_parts.append(f"   - Source: {article['source']}")
                response_parts.append(f"   - Author: {article['author']}")
                response_parts.append(f"   - Published: {time_ago}")
                response_parts.append(f"   - Reading time: About {article['reading_time']} minutes")
            
            # Add helpful context
            if article.get("sentiment") == "positive":
                response_parts.append(f"   - 😊 This is good news!")
        
        response_parts.append("\n\nWould you like me to read any of these articles in more detail?")
        
        return "\n".join(response_parts)
    
    def _get_friendly_time_ago(self, published_date: datetime) -> str:
        """Convert datetime to friendly 'time ago' string"""
        now = datetime.now()
        diff = now - published_date
        
        if diff.days == 0:
            if diff.seconds < 3600:
                return "less than an hour ago"
            elif diff.seconds < 7200:
                return "about an hour ago"
            else:
                hours = diff.seconds // 3600
                return f"about {hours} hours ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        elif diff.days < 14:
            return "last week"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} weeks ago"
        else:
            return "last month"