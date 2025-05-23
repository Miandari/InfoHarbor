"""
Storage interface for Elder Care Assistant memory system
"""
import sqlite3
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.memory.schemas import UserProfile, Memory


class MemoryStore:
    """SQLite-based storage for user profiles and memories"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    preferences TEXT,
                    health_info TEXT,
                    family_info TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT,
                    timestamp TEXT,
                    importance REAL,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            conn.commit()
    
    async def save_user_profile(self, profile: UserProfile) -> None:
        """Save user profile to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, name, preferences, health_info, family_info, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.name,
                json.dumps(profile.preferences),
                json.dumps(profile.health_info),
                json.dumps(profile.family_info),
                profile.created_at.isoformat(),
                profile.updated_at.isoformat()
            ))
            conn.commit()
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "user_id": row[0],
                    "name": row[1],
                    "preferences": json.loads(row[2]) if row[2] else {},
                    "health_info": json.loads(row[3]) if row[3] else {},
                    "family_info": json.loads(row[4]) if row[4] else {},
                    "created_at": datetime.fromisoformat(row[5]),
                    "updated_at": datetime.fromisoformat(row[6])
                }
            return None
    
    async def save_memory(self, memory: Memory) -> None:
        """Save memory to database"""
        if not memory.id:
            memory.id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memories 
                (id, user_id, content, memory_type, timestamp, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.user_id,
                memory.content,
                memory.memory_type,
                memory.timestamp.isoformat(),
                memory.importance,
                json.dumps(memory.metadata)
            ))
            conn.commit()
    
    async def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[Memory]:
        """Search memories by content (simple text search)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM memories 
                WHERE user_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, f"%{query}%", limit))
            
            memories = []
            for row in cursor.fetchall():
                memory = Memory(
                    id=row[0],
                    user_id=row[1],
                    content=row[2],
                    memory_type=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    importance=row[5],
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                memories.append(memory)
            
            return memories
    
    async def get_recent_memories(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get most recent memories for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM memories 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            memories = []
            for row in cursor.fetchall():
                memory = Memory(
                    id=row[0],
                    user_id=row[1],
                    content=row[2],
                    memory_type=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    importance=row[5],
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                memories.append(memory)
            
            return memories