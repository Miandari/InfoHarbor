"""
Memory manager for user-specific long-term memory using langmem.

This module provides functionality to:
1. Initialize and manage user memory collections
2. Store and retrieve memories in a persistent manner
3. Extract and update memory items from conversations
4. Provide relevant memories based on conversation context
"""

import os
import json
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

try:
    from langmem import create_manage_memory_tool, create_search_memory_tool
    from langgraph.store.memory import InMemoryStore
except ImportError:
    raise ImportError(
        "langmem package not found. Please install it with: pip install langmem"
    )

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import DEFAULT_MODEL, OPENAI_API_KEY
from utils.direct_response import debug_log, verbose_print

class UserMemorySchema:
    """Schema definition for user memory items."""
    
    # Core identity fields
    USER_ID = "user_id"
    NAME = "name"
    
    # Preference fields
    LIKES = "likes"
    DISLIKES = "dislikes"
    
    # Personal context fields
    IMPORTANT_FACTS = "important_facts"
    CONVERSATION_HISTORY = "conversation_history"
    
    # Metadata fields
    LAST_UPDATED = "last_updated"
    CREATED_AT = "created_at"
    CONFIDENCE = "confidence"
    MEMORY_TYPE = "memory_type"  # One of: "identity", "preference", "fact", "conversation"


class MemoryManager:
    """
    Manages user-specific long-term memory using langmem.
    
    This class handles:
    - Memory initialization and persistence
    - Memory retrieval based on relevance
    - Memory extraction from conversations
    - Memory updates with versioning
    """
    
    def __init__(
        self, 
        storage_path: str = "./memory_storage",
        namespace: str = "user_memories",
        debug: bool = False
    ):
        """
        Initialize the memory manager with proper configuration.
        
        Args:
            storage_path: Path to store memory data
            namespace: Namespace for memory collections
            debug: Whether to enable debug output
        """
        self.storage_path = storage_path
        self.namespace = namespace
        self.debug = debug
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Initialize LLM for memory operations
        self.llm = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=0.2,  # Low temperature for more consistent extraction
            api_key=OPENAI_API_KEY
        )
        
        # Initialize memory store using InMemoryStore
        # Based on latest langmem documentation, without persist_dir which is unsupported
        self.memory_store = InMemoryStore(
            index={
                "dims": 1536,  # OpenAI embedding dimensions
                "embed": "openai:text-embedding-3-small"
            }
        )
        
        # Create memory management tools with proper namespace format
        # Using tuple format as shown in documentation
        self.manage_memory_tool = create_manage_memory_tool(
            namespace=(namespace,),  # Note tuple syntax with trailing comma
            store=self.memory_store
        )
        
        self.search_memory_tool = create_search_memory_tool(
            namespace=(namespace,),  # Note tuple syntax with trailing comma
            store=self.memory_store
        )

    def debug_print(self, message: str) -> None:
        """
        Print debug message if debug mode is enabled.
        Also logs to debug_log.txt for persistent debugging.
        
        Args:
            message: Message to print/log
        """
        if self.debug:
            verbose_print(f"DEBUG: {message}")
        # Only log to debug file if debug is enabled
        if self.debug:
            debug_log(message)

    def initialize_user_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Initialize basic memory for a new user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with basic user memory structure
        """
        try:
            self.debug_print(f"Initializing memory for user: {user_id}")
            verbose_print(f"Initializing memory for user: {user_id}")
            
            # Create basic user identity memory
            user_memory = {
                "identity": {
                    UserMemorySchema.USER_ID: user_id,
                    UserMemorySchema.NAME: "",
                    UserMemorySchema.MEMORY_TYPE: "identity",
                    UserMemorySchema.CREATED_AT: datetime.now().isoformat(),
                    UserMemorySchema.LAST_UPDATED: datetime.now().isoformat(),
                    UserMemorySchema.CONFIDENCE: 1.0
                },
                "preferences": {
                    "likes": [],
                    "dislikes": [],
                    UserMemorySchema.MEMORY_TYPE: "preference",
                    UserMemorySchema.CREATED_AT: datetime.now().isoformat(),
                    UserMemorySchema.LAST_UPDATED: datetime.now().isoformat()
                },
                "important_facts": [],
                "conversation_history": []
            }
            
            # Store the basic memory
            try:
                # Fix: Use the correct tool reference and format
                store_result = self.manage_memory_tool.invoke({
                    "user_id": user_id,
                    "memory": user_memory["identity"],
                    "memory_type": "identity"
                })
                
                # Also store preferences structure
                pref_store_result = self.manage_memory_tool.invoke({
                    "user_id": user_id,
                    "memory": user_memory["preferences"],
                    "memory_type": "preference"
                })
                
                self.debug_print(f"Memory initialization result: {store_result}")
                verbose_print(f"Memory initialization result: {store_result}")
                    
                return user_memory
            except Exception as e:
                self.debug_print(f"Error storing user memory: {e}")
                return user_memory  # Return the memory even if storage fails
        except Exception as e:
            self.debug_print(f"Error initializing user memory: {e}")
            self.debug_print(f"Error traceback: {traceback.format_exc()}")
            return {}

    def retrieve_memories(
        self, 
        user_id: str, 
        conversation_context: Union[str, List[BaseMessage]], 
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for a user based on conversation context.
        
        Args:
            user_id: User identifier
            conversation_context: Current conversation context as text or messages
            limit: Maximum number of memories to retrieve
            
        Returns:
            Dictionary of relevant memory items organized by category
        """
        # Convert conversation context to string if it's a list of messages
        if isinstance(conversation_context, list):
            context_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in conversation_context[-5:]  # Use last 5 messages for context
            ])
        else:
            context_text = conversation_context
        
        # Initialize empty memory structure
        memory_categories = {
            "identity": {
                "name": "",
                "user_id": user_id
            },
            "preferences": {
                "likes": [], 
                "dislikes": []
            },
            "important_facts": [],
            "conversation_history": [],
        }
            
        try:
            # Search for relevant memories using the search tool
            search_query = {"query": context_text, "limit": limit}
            self.debug_print(f"Sending search query: {json.dumps(search_query)}")
            
            # Add debugging to check the actual search tool response
            try:
                memories = self.search_memory_tool.invoke(search_query)
                self.debug_print(f"Memory search result type: {type(memories)}")
                
                if isinstance(memories, str):
                    self.debug_print(f"Memory search string result preview: {memories}")
                    # Try to parse JSON string if that's what we got
                    if memories.strip().startswith('[') or memories.strip().startswith('{'):
                        try:
                            memories = json.loads(memories)
                            self.debug_print("Successfully parsed JSON string response")
                        except json.JSONDecodeError:
                            self.debug_print("Failed to parse memory response as JSON")
                    else:
                        self.debug_print(f"Received string response from memory system, length: {len(memories)}")
                        # Handle empty array case - don't initialize new memory for this
                        if memories.strip() == '[]':
                            self.debug_print("Empty array received, treating as no memories found")
                            memories = []
                
                if isinstance(memories, dict):
                    self.debug_print(f"Memory search dict result preview: {json.dumps(memories)[:200]}")
                elif isinstance(memories, list):
                    self.debug_print(f"Memory search list result length: {len(memories)}")
                    if memories and len(memories) > 0:
                        if isinstance(memories[0], str):
                            self.debug_print(f"First item preview (string): {memories[0][:100]}")
                        elif isinstance(memories[0], dict):
                            self.debug_print(f"First item preview (dict): {json.dumps(memories[0])[:100]}")
            except Exception as e:
                self.debug_print(f"Search memory tool error: {e}")
                self.debug_print(f"Search memory tool error traceback: {traceback.format_exc()}")
                memories = None
            
            # Handle empty or minimal responses
            if not memories or (isinstance(memories, list) and len(memories) == 0):
                self.debug_print("No memories found for the user, checking if user exists")
                # Check if user exists by searching for identity directly
                try:
                    identity_query = {"query": f"user_id: {user_id}", "limit": 1}
                    identity_result = self.search_memory_tool.invoke(identity_query)
                    
                    if identity_result and len(identity_result) > 0:
                        self.debug_print("User exists but no relevant memories found")
                        # User exists but no relevant memories for this context
                        return memory_categories
                    else:
                        self.debug_print("User does not exist, initializing new memory")
                        return self.initialize_user_memory(user_id)
                except Exception as e:
                    self.debug_print(f"Error checking user existence: {e}")
                    return self.initialize_user_memory(user_id)
        
        except Exception as e:
            self.debug_print(f"Error retrieving memories: {e}")
            self.debug_print(f"Error traceback: {traceback.format_exc()}")
            # If retrieval fails, initialize user memory
            identity_memory = self.initialize_user_memory(user_id)
            if identity_memory:
                memory_categories["identity"] = {
                    "name": identity_memory.get(UserMemorySchema.NAME, ""),
                    "user_id": identity_memory.get(UserMemorySchema.USER_ID, "")
                }
        
        return memory_categories

    def extract_memory_items(
        self, 
        user_id: str, 
        messages: List[BaseMessage]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract memory items from a conversation.
        
        Args:
            user_id: User identifier
            messages: List of conversation messages
            
        Returns:
            Dictionary of extracted memory items by category
        """
        if not messages:
            return {"extracted_items": []}
        
        # Convert messages to a summarized text
        conversation_text = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ])
        
        # Use LLM to extract memory items
        extraction_prompt = f"""
        Analyze this conversation and identify personal information about the user that should be remembered. 
        Focus on extracting:
        
        1. Identity information (name)
        2. Preferences (likes and dislikes)
        3. Important personal facts
        
        Conversation:
        {conversation_text}
        
        Extract the information and format it as a JSON with the following structure:
        {{
            "extracted_items": [
                {{
                    "memory_type": "identity|preference|fact",
                    "content": {{
                        // For identity:
                        "name": "User's name if mentioned",
                        
                        // For preferences:
                        "preference_type": "like|dislike",
                        "item": "What the user likes/dislikes",
                        "details": "Any details about the preference",
                        
                        // For facts:
                        "fact": "Important fact about the user",
                        "context": "Context where this fact was mentioned"
                    }},
                    "confidence": 0.1-1.0,  // How confident are you in this extraction?
                    "source_message_index": N  // Which message contained this information (0-indexed)
                }}
            ]
        }}
        
        Only include items that are explicitly mentioned or can be strongly inferred from the conversation.
        If no relevant information is found, return an empty array for "extracted_items".
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            content = response.content
            
            # Extract JSON from LLM response
            try:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Try to find JSON within the text
                    json_match = re.search(r'\{\s*"extracted_items"\s*:', content, re.DOTALL)
                    if json_match:
                        content = content[json_match.start():]
                        
                extracted_data = json.loads(content)
                return extracted_data
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from extraction response: {content}")
                return {"extracted_items": []}
                
        except Exception as e:
            print(f"Error extracting memory items: {e}")
            return {"extracted_items": []}

    def update_memory(
        self, 
        user_id: str, 
        extracted_items: List[Dict[str, Any]]
    ) -> None:
        """
        Update the memory store with new extracted items.
        
        Args:
            user_id: User identifier
            extracted_items: List of extracted memory items
        """
        if not extracted_items:
            return
            
        # Process each extracted item
        for item in extracted_items:
            memory_type = item.get("memory_type", "")
            content = item.get("content", {})
            confidence = item.get("confidence", 0.5)
            
            if memory_type == "identity":
                name = content.get("name", "")
                if name:
                    self._update_user_name(user_id, name, confidence)
                    
            elif memory_type == "preference":
                preference_type = content.get("preference_type", "")
                item_value = content.get("item", "")
                details = content.get("details", "")
                
                if preference_type and item_value:
                    # Store preference as a new memory
                    memory_data = {
                        "preference_type": preference_type,
                        "item": item_value,
                        "details": details,
                        UserMemorySchema.MEMORY_TYPE: "preference",
                        UserMemorySchema.CONFIDENCE: confidence,
                        UserMemorySchema.LAST_UPDATED: datetime.now().isoformat()
                    }
                    
                    try:
                        create_data = {
                            "action": "create",
                            "content": json.dumps(memory_data),
                            "metadata": {
                                "memory_type": "preference",
                                "preference_type": preference_type
                            }
                        }
                        
                        result = self.manage_memory_tool.invoke(create_data)
                    except Exception as e:
                        self.debug_print(f"Error storing preference: {e}")
                    
            elif memory_type == "fact":
                fact = content.get("fact", "")
                context = content.get("context", "")
                
                if fact:
                    # Store fact as a new memory
                    memory_data = {
                        "fact": fact,
                        "context": context,
                        UserMemorySchema.MEMORY_TYPE: "fact",
                        UserMemorySchema.CONFIDENCE: confidence,
                        UserMemorySchema.LAST_UPDATED: datetime.now().isoformat()
                    }
                    
                    try:
                        create_data = {
                            "action": "create",
                            "content": json.dumps(memory_data),
                            "metadata": {"memory_type": "fact"}
                        }
                        
                        result = self.manage_memory_tool.invoke(create_data)
                    except Exception as e:
                        self.debug_print(f"Error storing fact: {e}")

    def store_conversation_summary(
        self, 
        user_id: str, 
        messages: List[BaseMessage]
    ) -> None:
        """
        Create and store a summary of a conversation.
        
        Args:
            user_id: User identifier
            messages: List of conversation messages
        """
        if not messages:
            return
            
        # Convert messages to a summarized text
        conversation_text = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ])
        
        # Use LLM to summarize the conversation
        summary_prompt = f"""
        Create a concise summary of this conversation. Include:
        1. Main topics discussed
        2. Key points or decisions made
        3. Overall sentiment or tone
        
        Format your response as a JSON with the following structure:
        {{
            "summary": "Brief one-paragraph summary",
            "topics": ["topic1", "topic2"],
            "key_points": ["point1", "point2"],
            "sentiment": "positive|neutral|negative"
        }}
        
        Conversation:
        {conversation_text}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            content = response.content
            
            # Extract JSON from LLM response
            try:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                else:
                    # Try to find JSON within the text
                    json_match = re.search(r'\{\s*"summary"\s*:', content, re.DOTALL)
                    if json_match:
                        content = content[json_match.start():]
                        # Find the end of the JSON object
                        brace_count = 0
                        for i, char in enumerate(content):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    content = content[:i+1]
                                    break
                
                summary_data = json.loads(content)
                
                # Store the conversation summary
                memory_data = {
                    "summary": summary_data.get("summary", ""),
                    "topics": summary_data.get("topics", []),
                    "key_points": summary_data.get("key_points", []),
                    "sentiment": summary_data.get("sentiment", "neutral"),
                    UserMemorySchema.MEMORY_TYPE: "conversation",
                    UserMemorySchema.LAST_UPDATED: datetime.now().isoformat()
                }
                
                # Store using manage_memory_tool
                try:
                    create_data = {
                        "action": "create",
                        "content": json.dumps(memory_data),
                        "metadata": {"memory_type": "conversation"}
                    }
                    
                    result = self.manage_memory_tool.invoke(create_data)
                except Exception as e:
                    self.debug_print(f"Error storing conversation summary: {e}")
                
            except json.JSONDecodeError:
                self.debug_print(f"Failed to parse JSON from summary response: {content}")
                
        except Exception as e:
            self.debug_print(f"Error creating conversation summary: {e}")

    def format_memory_for_context(
        self, 
        memories: Dict[str, Any]
    ) -> str:
        """
        Format retrieved memories as context for the LLM.
        
        Args:
            memories: Dictionary of memory categories
            
        Returns:
            Formatted memory context
        """
        context_parts = []
        
        # Format identity information
        if memories.get("identity", {}).get("name"):
            context_parts.append(f"User's name: {memories['identity']['name']}")
        
        # Format preferences
        likes = memories.get("preferences", {}).get("likes", [])
        if likes:
            like_items = [f"- {item['item']}" + (f": {item['details']}" if item['details'] else "") 
                         for item in likes[:3]]  # Limit to top 3
            context_parts.append("User likes:\n" + "\n".join(like_items))
            
        dislikes = memories.get("preferences", {}).get("dislikes", [])
        if dislikes:
            dislike_items = [f"- {item['item']}" + (f": {item['details']}" if item['details'] else "") 
                            for item in dislikes[:3]]  # Limit to top 3
            context_parts.append("User dislikes:\n" + "\n".join(dislike_items))
            
        # Format important facts
        facts = memories.get("important_facts", [])
        if facts:
            fact_items = [f"- {item['fact']}" for item in facts[:5]]  # Limit to top 5
            context_parts.append("Important facts about the user:\n" + "\n".join(fact_items))
            
        # Format recent conversation history
        conversations = memories.get("conversation_history", [])
        if conversations:
            # Sort by timestamp (most recent first)
            sorted_conversations = sorted(
                conversations,
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            if sorted_conversations:
                latest_conversation = sorted_conversations[0]
                context_parts.append(f"Last conversation summary: {latest_conversation.get('summary', '')}")
        
        # Join all parts with line breaks
        if context_parts:
            return "User Information:\n" + "\n\n".join(context_parts)
        else:
            return ""

    def _update_user_name(
        self, 
        user_id: str, 
        name: str, 
        confidence: float
    ) -> None:
        """Update the user's name in memory."""
        try:
            # Search for existing identity memory
            search_query = {"query": "user identity", "limit": 1, "filter": {"memory_type": "identity"}}
            memories = self.search_memory_tool.invoke(search_query)
            
            # Get existing identity or create new one
            if memories and len(memories) > 0:
                # Skip processing if memories contains individual characters
                if isinstance(memories, list) and all(isinstance(m, str) and len(m.strip()) <= 1 for m in memories):
                    print("Received invalid memory data (individual characters), creating new identity memory")
                    self._create_identity_memory(user_id, name, confidence)
                    return
                
                # Parse existing memory
                try:
                    memory = memories[0] if isinstance(memories, list) else memories
                    
                    # Skip empty memory items
                    if not memory or (isinstance(memory, str) and not memory.strip()):
                        self._create_identity_memory(user_id, name, confidence)
                        return
                        
                    # Skip individual bracket characters
                    if isinstance(memory, str) and memory.strip() in ['[', ']', '{', '}', ',']:
                        self._create_identity_memory(user_id, name, confidence)
                        return
                    
                    # Try to parse string representation of memory
                    if isinstance(memory, str):
                        try:
                            existing_memory = json.loads(memory)
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing error in _update_user_name: {e}, creating new memory")
                            self._create_identity_memory(user_id, name, confidence)
                            return
                    else:
                        existing_memory = memory
                        
                    # If existing memory is in content field, extract it
                    if isinstance(existing_memory, dict) and "content" in existing_memory:
                        if isinstance(existing_memory["content"], str):
                            try:
                                content_data = json.loads(existing_memory["content"])
                                existing_memory = content_data
                            except json.JSONDecodeError:
                                # If content is not valid JSON, keep as is
                                pass
                        else:
                            existing_memory = existing_memory["content"]
                        
                    # Get memory ID and current confidence
                    memory_id = None
                    if isinstance(existing_memory, dict):
                        memory_id = existing_memory.get("id")
                        
                        # Try alternative locations for the ID
                        if not memory_id and "metadata" in existing_memory:
                            memory_id = existing_memory["metadata"].get("id")
                    
                    current_confidence = 0.0
                    if isinstance(existing_memory, dict):
                        confidence_data = existing_memory.get(UserMemorySchema.CONFIDENCE)
                        if isinstance(confidence_data, (float, int)):
                            current_confidence = confidence_data
                    
                    # Prepare updated memory data
                    if isinstance(existing_memory, dict):
                        # Update existing memory
                        updated_memory = existing_memory.copy()
                        updated_memory[UserMemorySchema.NAME] = name
                        updated_memory[UserMemorySchema.CONFIDENCE] = confidence
                        updated_memory[UserMemorySchema.LAST_UPDATED] = datetime.now().isoformat()
                        if UserMemorySchema.USER_ID not in updated_memory:
                            updated_memory[UserMemorySchema.USER_ID] = user_id
                        if UserMemorySchema.MEMORY_TYPE not in updated_memory:
                            updated_memory[UserMemorySchema.MEMORY_TYPE] = "identity"
                    else:
                        # Create new memory if existing memory is not a dict
                        self._create_identity_memory(user_id, name, confidence)
                        return
                    
                    # Update the memory if confidence is higher
                    if confidence >= current_confidence:
                        if memory_id:
                            update_data = {
                                "action": "update",
                                "id": memory_id,
                                "content": json.dumps(updated_memory),
                                "metadata": {"memory_type": "identity"}
                            }
                            
                            self.manage_memory_tool.invoke(update_data)
                        else:
                            # If no ID, create new memory
                            create_data = {
                                "action": "create",
                                "content": json.dumps(updated_memory),
                                "metadata": {"memory_type": "identity"}
                            }
                            
                            self.manage_memory_tool.invoke(create_data)
                    
                except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
                    print(f"Error updating user name: {e}")
                    print(f"Error traceback: {traceback.format_exc()}")
                    # Create new memory if parsing fails
                    self._create_identity_memory(user_id, name, confidence)
            else:
                # No existing memory, create new one
                self._create_identity_memory(user_id, name, confidence)
                
        except Exception as e:
            print(f"Error in _update_user_name: {e}")
            # Fallback to creating a new memory
            self._create_identity_memory(user_id, name, confidence)

    def _create_identity_memory(
        self,
        user_id: str,
        name: str,
        confidence: float
    ) -> None:
        """Create a new identity memory."""
        memory_data = {
            UserMemorySchema.USER_ID: user_id,
            UserMemorySchema.NAME: name,
            UserMemorySchema.LIKES: [],
            UserMemorySchema.DISLIKES: [],
            UserMemorySchema.IMPORTANT_FACTS: [],
            UserMemorySchema.CONVERSATION_HISTORY: [],
            UserMemorySchema.MEMORY_TYPE: "identity",
            UserMemorySchema.CONFIDENCE: confidence,
            UserMemorySchema.LAST_UPDATED: datetime.now().isoformat()
        }
        
        try:
            create_data = {
                "action": "create",
                "content": json.dumps(memory_data),
                "metadata": {"memory_type": "identity"}
            }
            
            self.manage_memory_tool.invoke(create_data)
        except Exception as e:
            print(f"Error creating identity memory: {e}")