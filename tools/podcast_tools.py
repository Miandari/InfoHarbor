"""
Podcast recommendation system with LangChain/LangGraph integration.
This implementation uses only the Tavily Search API without ListenNotes API.
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, Type, Union
from dotenv import load_dotenv

# Import Pydantic for schemas - Updated to Pydantic v2
from pydantic import BaseModel, Field

# Import LangChain tool decorators
from langchain_core.tools import tool

# Try to import Tavily - users should install this
try:
    from tavily import TavilyClient
except ImportError:
    print("Warning: Tavily not installed. Install with: pip install tavily-python")
    # Create a mock client for type checking
    class TavilyClient:
        def __init__(self, api_key=None): pass
        def search(self, **kwargs): return {"results": [], "answer": ""}

# Load environment variables
load_dotenv()

# Get API keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#-------------------------------------------------------------------------
# PART 1: CORE IMPLEMENTATION
#-------------------------------------------------------------------------

# Define common base argument schema
class BaseArgs(BaseModel):
    """A shared metadata every tool can optionally consume"""
    trace_id: str = Field(..., description="Used for distributed tracing")
    
    class Config:
        populate_by_name = True  # snake_case = camelCase
        extra = "forbid"  # no undeclared params

# Define the standard tool response schema
class ToolResponse(BaseModel):
    """Standard response format for all tools"""
    ok: bool = Field(..., description="True if the call succeeded")
    content: Any = Field(None, description="Primary payload")
    error: Optional[str] = Field(None, description="Human-readable error")

class PodcastRecommendationAgent:
    """
    An agent that provides podcast recommendations using only the Tavily Search API.
    """
    
    def __init__(
        self, 
        tavily_api_key: Optional[str] = None,
        use_env_vars: bool = True
    ):
        """
        Initialize the agent with API keys.
        
        Args:
            tavily_api_key: Your Tavily API key
            use_env_vars: Whether to use environment variables for API keys
        """
        # Get API keys from environment variables if not provided
        if use_env_vars:
            tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
            
        if not tavily_api_key:
            raise ValueError("Tavily API key is required")
            
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Preferred domains for podcast search
        self.podcast_domains = [
            "podcastreview.org", 
            "podchaser.com",
            "chartable.com", 
            "goodpods.com",
            "player.fm",
            "podcastaddict.com",
            "spotify.com",
            "apple.com/podcasts",
            "npr.org/podcasts",
            "pca.st",
            "iheart.com",
            "podbean.com",
            "radiotopia.fm",
            "gimletmedia.com"
        ]
    
    def recommend_podcasts(
        self, 
        query: str, 
        max_results: int = 5,
        search_depth: str = "advanced",
        include_episode_data: bool = True,
        topic_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get podcast recommendations based on user query.
        
        Args:
            query: User's query for podcast recommendations
            max_results: Maximum number of recommendations to return
            search_depth: Tavily search depth ('basic', 'advanced', or 'comprehensive')
            include_episode_data: Whether to include episode data
            topic_focus: Optional topic to focus recommendations on
            
        Returns:
            Dictionary containing recommendations and metadata
        """
        # Enhance the query for better podcast-specific results
        search_query = self._optimize_query(query, topic_focus)
        
        # Step 1: Search with Tavily
        search_results = self._get_tavily_results(search_query, search_depth)
        
        # Step 2: Extract podcast information from search results
        potential_podcasts = self._extract_podcast_info(search_results)
        
        # Step 3: Get episode data if requested
        if include_episode_data:
            enriched_podcasts = []
            for podcast in potential_podcasts[:max_results]:
                # Do a focused search for this podcast's episodes
                podcast_with_episodes = self._get_podcast_episodes(podcast["name"], 3)
                if podcast_with_episodes:
                    # Merge the podcast data with episode data
                    podcast.update(podcast_with_episodes)
                enriched_podcasts.append(podcast)
            
            recommendations = enriched_podcasts
        else:
            recommendations = potential_podcasts[:max_results]
            
        # Format the final response
        response = {
            "original_query": query,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "source": "Tavily Search API"
        }
            
        return response
    
    def get_podcast_by_name(self, podcast_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific podcast by name.
        
        Args:
            podcast_name: The name of the podcast to look up
            
        Returns:
            Dictionary with podcast details
        """
        try:
            # Do a focused search for this podcast
            search_query = f'"{podcast_name}" podcast details host information episodes'
            search_results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                include_domains=self.podcast_domains,
                max_results=5
            )
            
            # Extract podcast info
            podcast_info = self._extract_single_podcast_info(search_results, podcast_name)
            
            if not podcast_info:
                return {"error": "Podcast not found", "source": "Tavily Search API"}
            
            # Get episode data with a separate search
            episodes_data = self._get_podcast_episodes(podcast_name, 5)
            if episodes_data and "episodes" in episodes_data:
                podcast_info["episodes"] = episodes_data["episodes"]
            
            return {
                "podcast": podcast_info,
                "source": "Tavily Search API"
            }
            
        except Exception as e:
            return {"error": str(e), "source": "Tavily Search API"}
    
    def get_topic_recommendations(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Get podcast recommendations for a specific topic.
        
        Args:
            topic: The topic to get recommendations for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with topic-specific recommendations
        """
        return self.recommend_podcasts(
            f"best podcasts about {topic}",
            max_results=max_results,
            topic_focus=topic
        )
    
    def get_similar_to(self, podcast_name: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Get podcasts similar to a specific podcast.
        
        Args:
            podcast_name: The name of the podcast to find similar ones to
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with similar podcast recommendations
        """
        return self.recommend_podcasts(
            f"podcasts similar to {podcast_name} recommendations",
            max_results=max_results
        )
        
    def _optimize_query(self, query: str, topic_focus: Optional[str] = None) -> str:
        """
        Optimize the user query for better podcast-specific results.
        
        Args:
            query: Original user query
            topic_focus: Optional topic to focus on
            
        Returns:
            Optimized search query
        """
        # Check if the query already mentions podcasts
        if "podcast" not in query.lower():
            query = f"podcast {query}"
            
        # Add topic focus if provided
        if topic_focus:
            query = f"{query} about {topic_focus}"
            
        # Add recommendation intent if not present
        if not any(word in query.lower() for word in ["recommend", "best", "top", "popular"]):
            query = f"best {query} recommendations"
            
        return query
        
    def _get_tavily_results(self, query: str, search_depth: str) -> Dict[str, Any]:
        """
        Get search results from Tavily.
        
        Args:
            query: The search query
            search_depth: Depth of search ('basic', 'advanced', or 'comprehensive')
            
        Returns:
            Raw Tavily search results
        """
        try:
            # Perform the search with podcast-specific domains
            results = self.tavily_client.search(
                query=query,
                search_depth=search_depth,
                include_domains=self.podcast_domains,
                max_results=10,
                include_answer=True,
                include_raw_content=True
            )
            return results
        except Exception as e:
            print(f"Error with Tavily search: {e}")
            return {"results": [], "error": str(e)}
            
    def _extract_podcast_info(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract podcast information from Tavily search results.
        
        Args:
            search_results: Raw Tavily search results
            
        Returns:
            List of potential podcasts with extracted information
        """
        podcasts = []
        seen_titles = set()
        
        if not search_results or "results" not in search_results:
            return podcasts
            
        # Look for podcast names in the search results
        for result in search_results.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Extract potential podcast names using patterns
            potential_podcasts = self._extract_podcast_names(title, content)
            
            for podcast_name in potential_podcasts:
                # Skip if we've already added this podcast
                if podcast_name.lower() in seen_titles:
                    continue
                    
                # Create a podcast entry
                podcast_entry = {
                    "name": podcast_name,
                    "description": self._extract_description(content, podcast_name),
                    "source_url": url,
                    "confidence": self._calculate_confidence(result, podcast_name),
                    "publisher": self._extract_publisher(content, podcast_name),
                    "website": url if "podcast" in url.lower() else "",
                    "episodes": []  # Will be populated later if requested
                }
                
                podcasts.append(podcast_entry)
                seen_titles.add(podcast_name.lower())
                
        # If we have an answer from Tavily, process that too
        if "answer" in search_results and search_results["answer"]:
            answer_podcasts = self._extract_podcast_names_from_answer(search_results["answer"])
            
            for podcast_name in answer_podcasts:
                if podcast_name.lower() not in seen_titles:
                    podcast_entry = {
                        "name": podcast_name,
                        "description": self._extract_description(search_results["answer"], podcast_name),
                        "source_url": None,
                        "confidence": 0.9,  # High confidence for answer-derived podcasts
                        "publisher": self._extract_publisher(search_results["answer"], podcast_name),
                        "episodes": []  # Will be populated later if requested
                    }
                    
                    podcasts.append(podcast_entry)
                    seen_titles.add(podcast_name.lower())
        
        # Sort by confidence
        podcasts.sort(key=lambda x: x["confidence"], reverse=True)
        return podcasts
    
    def _extract_single_podcast_info(self, search_results: Dict[str, Any], podcast_name: str) -> Dict[str, Any]:
        """
        Extract information about a specific podcast from search results.
        
        Args:
            search_results: Tavily search results
            podcast_name: Name of the podcast to extract info for
            
        Returns:
            Dictionary with podcast information or None if not found
        """
        if not search_results or "results" not in search_results:
            return None
            
        # Combine all content to extract comprehensive information
        all_content = ""
        best_url = ""
        best_score = 0
        
        for result in search_results.get("results", []):
            content = result.get("content", "")
            all_content += " " + content
            
            # Find the best URL (likely to be the podcast's homepage)
            score = result.get("score", 0)
            if podcast_name.lower() in result.get("title", "").lower() and score > best_score:
                best_url = result.get("url", "")
                best_score = score
        
        # Also check the answer
        if "answer" in search_results and search_results["answer"]:
            all_content += " " + search_results["answer"]
            
        # Extract description, publisher, and host information
        description = self._extract_description(all_content, podcast_name)
        publisher = self._extract_publisher(all_content, podcast_name)
        host = self._extract_host(all_content, podcast_name)
        
        # Create a comprehensive podcast entry
        return {
            "title": podcast_name,
            "description": description,
            "publisher": publisher,
            "host": host,
            "website": best_url,
            "source": "Tavily Search API",
            "episodes": []  # Will be populated separately
        }
    
    def _get_podcast_episodes(self, podcast_name: str, count: int = 3) -> Dict[str, Any]:
        """
        Get episodes for a specific podcast using a separate Tavily search.
        
        Args:
            podcast_name: Name of the podcast
            count: Number of episodes to retrieve
            
        Returns:
            Dictionary with episode information
        """
        try:
            # Do a focused search for recent episodes
            search_query = f'"{podcast_name}" podcast recent episodes'
            results = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                include_domains=self.podcast_domains,
                max_results=5,
                include_answer=True
            )
            
            # Extract episodes from results
            episodes = self._extract_episodes_from_results(results, podcast_name, count)
            
            return {
                "episodes": episodes
            }
            
        except Exception as e:
            print(f"Error fetching episodes: {e}")
            return {"episodes": []}
    
    def _extract_episodes_from_results(self, results: Dict[str, Any], podcast_name: str, count: int) -> List[Dict[str, Any]]:
        """
        Extract episode information from search results.
        
        Args:
            results: Tavily search results
            podcast_name: Name of the podcast
            count: Maximum number of episodes to extract
            
        Returns:
            List of episode information dictionaries
        """
        episodes = []
        seen_titles = set()
        
        # Extract from Tavily answer first (it's often the best source)
        if "answer" in results and results["answer"]:
            answer_episodes = self._extract_episodes_from_text(results["answer"], podcast_name)
            for episode in answer_episodes:
                if episode["title"].lower() not in seen_titles:
                    episodes.append(episode)
                    seen_titles.add(episode["title"].lower())
        
        # Extract from search results
        for result in results.get("results", []):
            content = result.get("content", "")
            url = result.get("url", "")
            
            # Skip if this doesn't seem episode-related
            if not any(term in content.lower() for term in ["episode", "show notes", "listen"]):
                continue
                
            # Extract episodes from this result
            result_episodes = self._extract_episodes_from_text(content, podcast_name)
            
            for episode in result_episodes:
                if episode["title"].lower() not in seen_titles:
                    # Add episode URL if this result seems to be an episode page
                    if "episode" in url.lower():
                        episode["url"] = url
                    episodes.append(episode)
                    seen_titles.add(episode["title"].lower())
                    
                    # Stop if we have enough episodes
                    if len(episodes) >= count:
                        break
            
            if len(episodes) >= count:
                break
                
        return episodes[:count]
    
    def _extract_episodes_from_text(self, text: str, podcast_name: str) -> List[Dict[str, Any]]:
        """
        Extract episode information from text content.
        
        Args:
            text: Text content to extract from
            podcast_name: Name of the podcast
            
        Returns:
            List of episode information dictionaries
        """
        episodes = []
        
        # Pattern for finding episodes in numbered lists
        list_pattern = re.compile(r'(?:Episode\s+#?(\d+)[:\s-]+)?([^.!?]+)[.!?]')
        matches = list_pattern.findall(text)
        
        for match in matches:
            episode_num, title_text = match
            
            # Clean up title text
            title = title_text.strip()
            
            # If title is too long or too short, it's probably not a real episode title
            if len(title) < 5 or len(title) > 150:
                continue
                
            # Skip if this doesn't look like an episode title
            if not any(word in title.lower() for word in ["episode", podcast_name.lower(), "show", "talks", "discusses", "interview"]):
                # But if it has an episode number, it's probably an episode title
                if not episode_num:
                    continue
            
            # Create episode entry
            episode = {
                "title": title,
                "episode_number": episode_num if episode_num else None,
                "description": self._extract_episode_description(text, title),
                "audio_length_sec": None,  # Not available without audio metadata
                "url": "",  # Will be populated if we find a specific episode URL
                "thumbnail": ""  # Not available from text search
            }
            
            episodes.append(episode)
            
        # Try another pattern if we didn't find any episodes
        if not episodes:
            # Look for "Episode X:" or similar patterns
            episode_pattern = re.compile(r'Episode\s+#?(\d+)[:\s-]+([^.!?]+)[.!?]')
            matches = episode_pattern.findall(text)
            
            for match in matches:
                episode_num, title = match
                
                # Create episode entry
                episode = {
                    "title": title.strip(),
                    "episode_number": episode_num,
                    "description": self._extract_episode_description(text, title),
                    "audio_length_sec": None,
                    "url": "",
                    "thumbnail": ""
                }
                
                episodes.append(episode)
                
        return episodes
            
    def _extract_podcast_names(self, title: str, content: str) -> List[str]:
        """
        Extract podcast names from text using various patterns.
        
        Args:
            title: Title of the search result
            content: Content of the search result
            
        Returns:
            List of potential podcast names
        """
        podcast_names = []
        
        # Pattern for titles like "X podcast"
        title_pattern = re.compile(r'([^.,:;()"\']+?)\s+podcast', re.IGNORECASE)
        
        # Pattern for podcast names in lists or with quotes
        list_pattern = re.compile(r'(?:\d+\.\s+|\*\s+)(?:"([^"]+)"|\'([^\']+)\'|([^,.;:]+))', re.IGNORECASE)
        
        # Pattern for "hosted by" or "with host"
        host_pattern = re.compile(r'([^,.;:]+)(?:\s+is\s+hosted\s+by|\s+with\s+host)', re.IGNORECASE)
        
        # Extract from title
        title_matches = title_pattern.findall(title)
        podcast_names.extend([match.strip() for match in title_matches if len(match.strip()) > 3])
        
        # Extract from content
        title_matches = title_pattern.findall(content)
        podcast_names.extend([match.strip() for match in title_matches if len(match.strip()) > 3])
        
        # Extract list items
        list_matches = list_pattern.findall(content)
        for match_tuple in list_matches:
            for match in match_tuple:
                if match and len(match.strip()) > 3:
                    podcast_names.append(match.strip())
        
        # Extract host pattern
        host_matches = host_pattern.findall(content)
        podcast_names.extend([match.strip() for match in host_matches if len(match.strip()) > 3])
        
        # Remove duplicates while preserving order
        unique_names = []
        seen = set()
        for name in podcast_names:
            if name.lower() not in seen:
                unique_names.append(name)
                seen.add(name.lower())
                
        return unique_names
    
    def _extract_podcast_names_from_answer(self, answer_text: str) -> List[str]:
        """
        Extract podcast names from Tavily's answer text.
        
        Args:
            answer_text: The answer text from Tavily
            
        Returns:
            List of podcast names
        """
        # Look for numbered or bulleted lists
        list_pattern = re.compile(r'(?:\d+\.\s+|\*\s+)(?:"([^"]+)"|\'([^\']+)\'|([^,.;:]+))', re.IGNORECASE)
        matches = list_pattern.findall(answer_text)
        
        podcasts = []
        for match_tuple in matches:
            for match in match_tuple:
                if match and len(match.strip()) > 3:
                    podcasts.append(match.strip())
        
        # Look for podcast names in running text
        text_pattern = re.compile(r'(?:"([^"]+)"|\'([^\']+)\')(?:\s+podcast)', re.IGNORECASE)
        text_matches = text_pattern.findall(answer_text)
        
        for match_tuple in text_matches:
            for match in match_tuple:
                if match and len(match.strip()) > 3:
                    podcasts.append(match.strip())
                    
        return podcasts
            
    def _extract_description(self, content: str, podcast_name: str) -> str:
        """
        Extract a description for a podcast from the content.
        
        Args:
            content: Content to extract from
            podcast_name: Name of the podcast
            
        Returns:
            Extracted description
        """
        # Try to find sentences containing the podcast name
        sentences = re.split(r'(?<=[.!?])\s+', content)
        relevant_sentences = []
        
        for sentence in sentences:
            if podcast_name.lower() in sentence.lower():
                relevant_sentences.append(sentence)
                
        # If we found relevant sentences, join them
        if relevant_sentences:
            description = ' '.join(relevant_sentences[:2])  # Limit to 2 sentences
            # Truncate if too long
            if len(description) > 300:
                description = description[:297] + "..."
            return description
        
        # Otherwise return a generic description
        return f"A podcast that matched your search criteria."
    
    def _extract_episode_description(self, content: str, episode_title: str) -> str:
        """
        Extract a description for an episode from content.
        
        Args:
            content: Content to extract from
            episode_title: Title of the episode
            
        Returns:
            Extracted episode description
        """
        # Find sentences after the episode title
        pattern = re.compile(re.escape(episode_title) + r'[.!?]\s+([^.!?]+)[.!?]')
        matches = pattern.findall(content)
        
        if matches:
            return matches[0].strip()
        
        # Alternative: find sentences containing words from the episode title
        words = [word for word in episode_title.lower().split() if len(word) > 3]
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        for sentence in sentences:
            # Check if this sentence contains multiple words from the title
            if sum(1 for word in words if word in sentence.lower()) >= 2:
                return sentence.strip()
                
        return "No description available."
    
    def _extract_publisher(self, content: str, podcast_name: str) -> str:
        """
        Extract the publisher of a podcast from content.
        
        Args:
            content: Content to extract from
            podcast_name: Name of the podcast
            
        Returns:
            Extracted publisher name or empty string
        """
        # Look for patterns like "X is produced by Y" or "from Y"
        produced_pattern = re.compile(
            r'(?:' + re.escape(podcast_name) + r'|it|this podcast)(?:\s+is)?\s+(?:produced|published|distributed|created)\s+by\s+([^.,;:!?]+)',
            re.IGNORECASE
        )
        
        from_pattern = re.compile(
            r'(?:' + re.escape(podcast_name) + r'|this podcast)(?:\s+is)?\s+from\s+([^.,;:!?]+)',
            re.IGNORECASE
        )
        
        network_pattern = re.compile(
            r'(?:' + re.escape(podcast_name) + r'|it|this podcast)(?:\s+is)?\s+(?:a\s+podcast\s+from|part\s+of)\s+(?:the\s+)?([^.,;:!?]+)(?:\s+network|podcast\s+network)?',
            re.IGNORECASE
        )
        
        # Try each pattern
        for pattern in [produced_pattern, from_pattern, network_pattern]:
            matches = pattern.findall(content)
            if matches:
                publisher = matches[0].strip()
                # Remove "the" from beginning
                if publisher.lower().startswith("the "):
                    publisher = publisher[4:]
                return publisher
                
        return ""
    
    def _extract_host(self, content: str, podcast_name: str) -> str:
        """
        Extract the host of a podcast from content.
        
        Args:
            content: Content to extract from
            podcast_name: Name of the podcast
            
        Returns:
            Extracted host name or empty string
        """
        # Look for patterns like "hosted by X" or "with host X"
        host_pattern = re.compile(
            r'(?:' + re.escape(podcast_name) + r'|it|this podcast)(?:\s+is)?\s+hosted\s+by\s+([^.,;:!?]+)',
            re.IGNORECASE
        )
        
        with_host_pattern = re.compile(
            r'(?:' + re.escape(podcast_name) + r'|it|this podcast)(?:\s+is)?\s+with\s+(?:host|presenter)\s+([^.,;:!?]+)',
            re.IGNORECASE
        )
        
        # Try each pattern
        for pattern in [host_pattern, with_host_pattern]:
            matches = pattern.findall(content)
            if matches:
                return matches[0].strip()
                
        return ""
            
    def _calculate_confidence(self, result: Dict[str, Any], podcast_name: str) -> float:
        """
        Calculate confidence score for a podcast recommendation.
        
        Args:
            result: Search result
            podcast_name: Name of the podcast
            
        Returns:
            Confidence score between 0 and 1
        """
        base_score = result.get("score", 0)
        
        # Boost score if podcast name appears in title
        title_boost = 0.2 if podcast_name.lower() in result.get("title", "").lower() else 0
        
        # Boost score if result is from a podcast-specific domain
        domain_boost = 0.1
        for domain in self.podcast_domains:
            if domain in result.get("url", ""):
                domain_boost = 0.3
                break
                
        # Calculate final score, capped at 1.0
        confidence = min(1.0, base_score + title_boost + domain_boost)
        return confidence


# Define Pydantic schemas for tool arguments that extend BaseArgs
class PodcastRecommendationArgs(BaseArgs):
    """Input schema for podcast recommendations."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Number of results (between 1-20)")
    include_episodes: bool = Field(True, description="Include episode information")

class PodcastByNameArgs(BaseArgs):
    """Input schema for getting details about a specific podcast."""
    podcast_name: str = Field(..., description="The name of the podcast to look up")

class SimilarPodcastsArgs(BaseArgs):
    """Input schema for finding podcasts similar to a given one."""
    podcast_name: str = Field(..., description="The podcast name to find similar ones to")
    max_results: int = Field(10, description="Number of results (between 1-20)")

class TopicRecommendationArgs(BaseArgs):
    """Input schema for getting podcast recommendations on a specific topic."""
    topic: str = Field(..., description="The topic to get recommendations for")
    max_results: int = Field(10, description="Number of results (between 1-20)")


# Define tool classes using the schemas
class PodcastRecommendationTool:
    """Tool for recommending podcasts based on user queries."""
    
    def __init__(self, agent: PodcastRecommendationAgent = None):
        """Initialize the tool with a podcast recommendation agent."""
        self.agent = agent or PodcastRecommendationAgent(use_env_vars=True)
    
    def run(self, args: PodcastRecommendationArgs) -> ToolResponse:
        """
        Get podcast recommendations based on user query.
        
        Args:
            args: Arguments for the podcast recommendation tool
        
        Returns:
            ToolResponse with podcast recommendations
        """
        try:
            # Validate max_results
            max_results = min(max(args.max_results, 1), 20)
            
            results = self.agent.recommend_podcasts(
                query=args.query,
                max_results=max_results,
                include_episode_data=args.include_episodes
            )
            
            # Return successful response
            return ToolResponse(
                ok=True,
                content=results,
                error=None
            )
        except Exception as e:
            # Return error response
            return ToolResponse(
                ok=False,
                content=None,
                error=f"Error getting podcast recommendations: {str(e)}"
            )


class PodcastByNameTool:
    """Tool for getting detailed information about a specific podcast by name."""
    
    def __init__(self, agent: PodcastRecommendationAgent = None):
        """Initialize the tool with a podcast recommendation agent."""
        self.agent = agent or PodcastRecommendationAgent(use_env_vars=True)
    
    def run(self, args: PodcastByNameArgs) -> ToolResponse:
        """
        Get detailed information about a specific podcast by name.
        
        Args:
            args: Arguments for the podcast lookup tool
        
        Returns:
            ToolResponse with podcast details
        """
        try:
            result = self.agent.get_podcast_by_name(args.podcast_name)
            
            # Check if there was an error from the API
            if "error" in result:
                return ToolResponse(
                    ok=False,
                    content=None,
                    error=f"Error retrieving podcast: {result['error']}"
                )
            
            # Return successful response
            return ToolResponse(
                ok=True,
                content=result,
                error=None
            )
        except Exception as e:
            # Return error response
            return ToolResponse(
                ok=False,
                content=None,
                error=f"Error getting podcast details: {str(e)}"
            )


class SimilarPodcastsTool:
    """Tool for finding podcasts similar to a specified podcast."""
    
    def __init__(self, agent: PodcastRecommendationAgent = None):
        """Initialize the tool with a podcast recommendation agent."""
        self.agent = agent or PodcastRecommendationAgent(use_env_vars=True)
    
    def run(self, args: SimilarPodcastsArgs) -> ToolResponse:
        """
        Find podcasts similar to a specified podcast.
        
        Args:
            args: Arguments for the similar podcasts tool
        
        Returns:
            ToolResponse with similar podcast recommendations
        """
        try:
            # Validate max_results
            max_results = min(max(args.max_results, 1), 20)
            
            results = self.agent.get_similar_to(
                podcast_name=args.podcast_name,
                max_results=max_results
            )
            
            # Return successful response
            return ToolResponse(
                ok=True,
                content=results,
                error=None
            )
        except Exception as e:
            # Return error response
            return ToolResponse(
                ok=False,
                content=None,
                error=f"Error finding similar podcasts: {str(e)}"
            )


class TopicPodcastsTool:
    """Tool for finding podcasts on a specific topic."""
    
    def __init__(self, agent: PodcastRecommendationAgent = None):
        """Initialize the tool with a podcast recommendation agent."""
        self.agent = agent or PodcastRecommendationAgent(use_env_vars=True)
    
    def run(self, args: TopicRecommendationArgs) -> ToolResponse:
        """
        Get podcast recommendations for a specific topic.
        
        Args:
            args: Arguments for the topic recommendations tool
        
        Returns:
            ToolResponse with topic-specific podcast recommendations
        """
        try:
            # Validate max_results
            max_results = min(max(args.max_results, 1), 20)
            
            results = self.agent.get_topic_recommendations(
                topic=args.topic,
                max_results=max_results
            )
            
            # Return successful response
            return ToolResponse(
                ok=True,
                content=results,
                error=None
            )
        except Exception as e:
            # Return error response
            return ToolResponse(
                ok=False,
                content=None,
                error=f"Error getting topic recommendations: {str(e)}"
            )


# Helper function to create all tool instances
def create_podcast_tools():
    """
    Create the podcast recommendation tools and return them as a dictionary.
    
    Returns:
        Dictionary mapping tool names to tool instances
    """
    # Initialize the podcast recommendation agent (singleton pattern)
    podcast_agent = get_podcast_agent()
    
    # Create the tools
    tools = {
        "podcast_recommendation": PodcastRecommendationTool(agent=podcast_agent),
        "podcast_by_name": PodcastByNameTool(agent=podcast_agent),
        "similar_podcasts": SimilarPodcastsTool(agent=podcast_agent),
        "topic_podcasts": TopicPodcastsTool(agent=podcast_agent)
    }
    
    return tools

#-------------------------------------------------------------------------
# PART 2: LANGCHAIN TOOL INTEGRATION 
#-------------------------------------------------------------------------

# Create a singleton instance of the podcast agent
def get_podcast_agent():
    """Get or create the singleton podcast agent instance."""
    if not hasattr(get_podcast_agent, "_instance"):
        get_podcast_agent._instance = PodcastRecommendationAgent(
            tavily_api_key=None,  # Will use env var
            use_env_vars=True
        )
    return get_podcast_agent._instance

# Define structured tool inputs for LangChain
class PodcastRecommendationInput(BaseModel):
    query: str = Field(..., description="Search query for podcast recommendations")
    max_results: int = Field(10, description="Number of results to return (1-20)")
    include_episodes: bool = Field(True, description="Whether to include episode information")

@tool(args_schema=PodcastRecommendationInput)
def get_podcast_recommendations(query: str, max_results: int = 10, include_episodes: bool = True) -> dict:
    """Get podcast recommendations based on user query."""
    # Initialize your tools
    podcast_tools = create_podcast_tools()
    
    # Make sure max_results is within acceptable bounds
    max_results = min(max(max_results, 1), 20)
    
    # Create the arguments for your existing tool
    args = PodcastRecommendationArgs(
        trace_id="langgraph-execution",
        query=query,
        max_results=max_results,
        include_episodes=include_episodes
    )
    
    # Execute your tool and get the result
    result = podcast_tools["podcast_recommendation"].run(args)
    
    if result.ok:
        return result.content
    else:
        raise ValueError(result.error)

@tool
def get_podcast_details(podcast_name: str) -> dict:
    """Get detailed information about a specific podcast by name."""
    podcast_tools = create_podcast_tools()
    args = PodcastByNameArgs(trace_id="langgraph-execution", podcast_name=podcast_name)
    result = podcast_tools["podcast_by_name"].run(args)
    
    if result.ok:
        return result.content
    else:
        raise ValueError(result.error)

@tool
def get_similar_podcasts(podcast_name: str, max_results: int = 10) -> dict:
    """Find podcasts similar to a specified podcast."""
    podcast_tools = create_podcast_tools()
    # Make sure max_results is within acceptable bounds
    max_results = min(max(max_results, 1), 20)
    
    args = SimilarPodcastsArgs(trace_id="langgraph-execution", podcast_name=podcast_name, max_results=max_results)
    result = podcast_tools["similar_podcasts"].run(args)
    
    if result.ok:
        return result.content
    else:
        raise ValueError(result.error)

@tool
def get_topic_podcasts(topic: str, max_results: int = 10) -> dict:
    """Find podcasts on a specific topic."""
    podcast_tools = create_podcast_tools()
    # Make sure max_results is within acceptable bounds
    max_results = min(max(max_results, 1), 20)
    
    args = TopicRecommendationArgs(trace_id="langgraph-execution", topic=topic, max_results=max_results)
    result = podcast_tools["topic_podcasts"].run(args)
    
    if result.ok:
        return result.content
    else:
        raise ValueError(result.error)

# Export all tools for LangGraph 
podcast_tools = [
    get_podcast_recommendations,
    get_podcast_details,
    get_similar_podcasts,
    get_topic_podcasts
]

# Example of using the tools
if __name__ == "__main__":
    import json
    
    # Create the tools
    tools = create_podcast_tools()
    
    # Example 1: Get podcast recommendations
    print("Using traditional tool:")
    recommendation_args = PodcastRecommendationArgs(
        trace_id="demo-123",
        query="educational history podcasts that focus on ancient civilizations",
        max_results=3,
        include_episodes=True
    )
    
    recommendation_result = tools["podcast_recommendation"].run(recommendation_args)
    print("PODCAST RECOMMENDATIONS RESULT:")
    print(f"Success: {recommendation_result.ok}")
    if recommendation_result.ok:
        print(f"Found {recommendation_result.content['recommendation_count']} recommendations")
        for i, podcast in enumerate(recommendation_result.content.get('recommendations', []), 1):
            print(f"{i}. {podcast.get('name', 'Unknown')}")
    else:
        print(f"Error: {recommendation_result.error}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using the LangChain tool wrappers
    print("Using LangChain tool:")
    result = get_podcast_recommendations(
        query="history podcasts about ancient Rome",
        max_results=2,
        include_episodes=True
    )
    
    print("LANGCHAIN TOOL RESULT:")
    print(f"Found {result['recommendation_count']} recommendations")
    for i, podcast in enumerate(result.get('recommendations', []), 1):
        print(f"{i}. {podcast.get('name', 'Unknown')}")