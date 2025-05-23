"""
Podcast discovery tool following the consistent pattern
"""
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from src.tools.base import BaseTool, ToolResult


class PodcastDiscoveryTool(BaseTool):
    """Tool for discovering and recommending podcasts"""
    
    @property
    def name(self) -> str:
        return "podcast_discovery"
    
    @property
    def description(self) -> str:
        return "Search for and recommend podcasts based on user interests and preferences"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The podcast search query or topic of interest"
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "user_preferences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "User's known podcast preferences"
                        },
                        "exclude_topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Topics to avoid"
                        },
                        "episode_length": {
                            "type": "string",
                            "enum": ["short", "medium", "long", "any"],
                            "description": "Preferred episode length"
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
        """Search for podcasts based on query"""
        
        # For MVP, return mock data
        # In production, integrate with podcast APIs like Spotify or Apple Podcasts
        
        # Extract preferences from context
        user_prefs = context.get("user_preferences", []) if context else []
        exclude_topics = context.get("exclude_topics", []) if context else []
        episode_length = context.get("episode_length", "any") if context else "any"
        
        # Mock podcast database
        mock_podcasts = self._get_mock_podcasts()
        
        # Filter based on query
        query_lower = query.lower()
        filtered = [
            p for p in mock_podcasts
            if query_lower in p["title"].lower() 
            or query_lower in p["description"].lower()
            or any(query_lower in cat.lower() for cat in p["categories"])
        ]
        
        # Filter out excluded topics
        if exclude_topics:
            filtered = [
                p for p in filtered
                if not any(
                    topic.lower() in p["title"].lower() 
                    or topic.lower() in p["description"].lower()
                    for topic in exclude_topics
                )
            ]
        
        # Sort by relevance (mock scoring)
        for podcast in filtered:
            score = 0
            # Boost score for matching user preferences
            for pref in user_prefs:
                if pref.lower() in podcast["title"].lower() or \
                   pref.lower() in podcast["description"].lower():
                    score += 2
            
            # Boost score for exact query match
            if query_lower in podcast["title"].lower():
                score += 3
                
            # Consider episode length preference
            if episode_length != "any" and podcast.get("avg_length") == episode_length:
                score += 1
                
            podcast["relevance_score"] = score
        
        # Sort by relevance
        filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take top 5 results
        results = filtered[:5]
        
        if results:
            # Format results for elderly users
            formatted_results = self._format_results_for_elderly(results)
            
            return ToolResult(
                success=True,
                data={
                    "podcasts": results,
                    "formatted_response": formatted_results,
                    "total_found": len(filtered),
                    "search_query": query
                },
                metadata={
                    "source": "mock_data",  # In production: "spotify_api"
                    "filtered_count": len(filtered),
                    "preferences_applied": bool(user_prefs)
                }
            )
        else:
            return ToolResult(
                success=True,
                data={
                    "podcasts": [],
                    "formatted_response": f"I couldn't find any podcasts about '{query}'. Would you like me to search for something similar?",
                    "total_found": 0,
                    "search_query": query
                }
            )
    
    def _get_mock_podcasts(self) -> List[Dict[str, Any]]:
        """Get mock podcast data for MVP"""
        return [
            {
                "title": "The History of Rome",
                "host": "Mike Duncan",
                "description": "A weekly podcast tracing the history of the Roman Empire, beginning with Aeneas's arrival in Italy and ending with the exile of Romulus Augustulus.",
                "categories": ["History", "Education"],
                "avg_length": "medium",
                "episode_count": 179,
                "latest_episode": "Episode 179 - The End",
                "latest_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "rating": 4.8,
                "subscriber_count": 50000
            },
            {
                "title": "Stuff You Should Know",
                "host": "Josh Clark and Chuck Bryant",
                "description": "If you've ever wanted to know about champagne, satanism, the Stonewall Uprising, chaos theory, LSD, El Nino, true crime and Rosa Parks, then look no further.",
                "categories": ["Education", "Comedy", "Society & Culture"],
                "avg_length": "long",
                "episode_count": 1500,
                "latest_episode": "How Retirement Works",
                "latest_date": (datetime.now() - timedelta(days=2)).isoformat(),
                "rating": 4.7,
                "subscriber_count": 100000
            },
            {
                "title": "Science Vs",
                "host": "Wendy Zukerman",
                "description": "We pit facts against popular myths to find out what's real and what's not.",
                "categories": ["Science", "Health"],
                "avg_length": "medium",
                "episode_count": 200,
                "latest_episode": "Memory: Can We Trust It?",
                "latest_date": (datetime.now() - timedelta(days=7)).isoformat(),
                "rating": 4.6,
                "subscriber_count": 75000
            },
            {
                "title": "The Daily",
                "host": "The New York Times",
                "description": "This is what the news should sound like. The biggest stories of our time, told by the best journalists in the world.",
                "categories": ["News", "Politics"],
                "avg_length": "short",
                "episode_count": 1800,
                "latest_episode": "Today's Top Stories",
                "latest_date": datetime.now().isoformat(),
                "rating": 4.5,
                "subscriber_count": 200000
            },
            {
                "title": "Fresh Air",
                "host": "Terry Gross",
                "description": "Fresh Air from NPR features interviews with prominent cultural and entertainment figures, as well as distinguished experts on current affairs and news.",
                "categories": ["Arts", "Culture", "News"],
                "avg_length": "long",
                "episode_count": 5000,
                "latest_episode": "Interview with Award-Winning Author",
                "latest_date": (datetime.now() - timedelta(days=1)).isoformat(),
                "rating": 4.7,
                "subscriber_count": 150000
            }
        ]
    
    def _format_results_for_elderly(self, podcasts: List[Dict[str, Any]]) -> str:
        """Format podcast results in an elderly-friendly way"""
        if not podcasts:
            return "I couldn't find any podcasts matching your request."
        
        response_parts = [
            f"I found {len(podcasts)} podcasts that might interest you:\n"
        ]
        
        for i, podcast in enumerate(podcasts, 1):
            # Format episode length in friendly terms
            length_map = {
                "short": "about 20-30 minutes",
                "medium": "about 30-45 minutes",
                "long": "about 45-60 minutes or more"
            }
            
            length_desc = length_map.get(podcast.get("avg_length", "medium"), "varied lengths")
            
            # Format the podcast info
            response_parts.append(f"\n{i}. **{podcast['title']}**")
            response_parts.append(f"   - Hosted by: {podcast['host']}")
            response_parts.append(f"   - About: {podcast['description'][:100]}...")
            response_parts.append(f"   - Episodes are {length_desc}")
            response_parts.append(f"   - Latest episode: \"{podcast['latest_episode']}\"")
            
            # Add helpful context for elderly users
            if podcast.get("rating", 0) >= 4.5:
                response_parts.append(f"   - This is a very popular and well-reviewed podcast!")
                
        response_parts.append("\n\nWould you like me to help you listen to any of these podcasts?")
        
        return "\n".join(response_parts)