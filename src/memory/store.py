"""
SQLite-based memory storage implementation
"""
import aiosqlite
import json
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from src.memory.schemas import Memory, UserProfile, ConversationSummary, MemoryType
from src.settings import settings


class MemoryStore:
    """SQLite-based storage for user memories and profiles"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.memory_db_path
        self.logger = logging.getLogger("memory.store")
        
    async def initialize(self):
        """Initialize the database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create memories table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # Create user profiles table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    personal_info TEXT,
                    health_info TEXT,
                    preferences TEXT,
                    routines TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create conversation summaries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_topics TEXT,
                    mood TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_summaries_user_id ON conversation_summaries(user_id)")
            
            await db.commit()
    
    async def store_memory(self, memory: Memory) -> str:
        """Store a memory and return its ID"""
        if not memory.id:
            memory.id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO memories 
                (id, user_id, conversation_id, memory_type, content, importance, 
                 created_at, last_accessed, access_count, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.user_id,
                memory.conversation_id,
                memory.memory_type.value,
                memory.content,
                memory.importance,
                memory.created_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                json.dumps(memory.tags),
                json.dumps(memory.metadata)
            ))
            await db.commit()
        
        return memory.id
    
    async def get_memories(
        self, 
        user_id: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Retrieve memories for a user"""
        async with aiosqlite.connect(self.db_path) as db:
            if memory_type:
                cursor = await db.execute("""
                    SELECT * FROM memories 
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                """, (user_id, memory_type.value, limit))
            else:
                cursor = await db.execute("""
                    SELECT * FROM memories 
                    WHERE user_id = ?
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                """, (user_id, limit))
            
            rows = await cursor.fetchall()
            memories = []
            
            for row in rows:
                memory = Memory(
                    id=row[0],
                    user_id=row[1],
                    conversation_id=row[2],
                    memory_type=MemoryType(row[3]),
                    content=row[4],
                    importance=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    last_accessed=datetime.fromisoformat(row[7]),
                    access_count=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    metadata=json.loads(row[10]) if row[10] else {}
                )
                memories.append(memory)
            
            return memories
    
    async def search_memories(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[Memory]:
        """Search memories by content"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM memories 
                WHERE user_id = ? AND content LIKE ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """, (user_id, f"%{query}%", limit))
            
            rows = await cursor.fetchall()
            memories = []
            
            for row in rows:
                memory = Memory(
                    id=row[0],
                    user_id=row[1],
                    conversation_id=row[2],
                    memory_type=MemoryType(row[3]),
                    content=row[4],
                    importance=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    last_accessed=datetime.fromisoformat(row[7]),
                    access_count=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    metadata=json.loads(row[10]) if row[10] else {}
                )
                memories.append(memory)
            
            return memories
    
    async def update_memory_access(self, memory_id: str):
        """Update memory access time and count"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE memories 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), memory_id))
            await db.commit()
    
    async def store_user_profile(self, profile: UserProfile):
        """Store or update user profile"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, personal_info, health_info, preferences, routines, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                profile.personal_info.model_dump_json() if profile.personal_info else None,
                json.dumps([h.model_dump() for h in profile.health_info]),
                json.dumps([p.model_dump() for p in profile.preferences]),
                json.dumps([r.model_dump() for r in profile.routines]),
                datetime.now().isoformat()
            ))
            await db.commit()
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            row = await cursor.fetchone()
            if not row:
                # Return empty profile
                return UserProfile(user_id=user_id)
            
            # Parse the stored JSON data
            from src.memory.schemas import PersonalInfo, HealthInfo, UserPreference, DailyRoutine
            
            personal_info = None
            if row[1]:
                personal_info = PersonalInfo.model_validate_json(row[1])
            
            health_info = []
            if row[2]:
                health_data = json.loads(row[2])
                health_info = [HealthInfo.model_validate(h) for h in health_data]
            
            preferences = []
            if row[3]:
                pref_data = json.loads(row[3])
                preferences = [UserPreference.model_validate(p) for p in pref_data]
            
            routines = []
            if row[4]:
                routine_data = json.loads(row[4])
                routines = [DailyRoutine.model_validate(r) for r in routine_data]
            
            # Get recent conversation summaries
            recent_summaries = await self._get_recent_summaries(user_id)
            
            return UserProfile(
                user_id=user_id,
                personal_info=personal_info,
                health_info=health_info,
                preferences=preferences,
                routines=routines,
                recent_summaries=recent_summaries,
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6])
            )
    
    async def store_conversation_summary(self, summary: ConversationSummary):
        """Store conversation summary"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO conversation_summaries 
                (conversation_id, user_id, summary, key_topics, mood, created_at, message_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                summary.conversation_id,
                summary.user_id,
                summary.summary,
                json.dumps(summary.key_topics),
                summary.mood,
                summary.created_at.isoformat(),
                summary.message_count
            ))
            await db.commit()
    
    async def _get_recent_summaries(self, user_id: str, limit: int = 5) -> List[ConversationSummary]:
        """Get recent conversation summaries for user"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT * FROM conversation_summaries 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = await cursor.fetchall()
            summaries = []
            
            for row in rows:
                summary = ConversationSummary(
                    conversation_id=row[0],
                    user_id=row[1],
                    summary=row[2],
                    key_topics=json.loads(row[3]) if row[3] else [],
                    mood=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    message_count=row[6]
                )
                summaries.append(summary)
            
            return summaries
    
    async def cleanup_old_memories(self, days: Optional[int] = None):
        """Clean up old, low-importance memories"""
        days = days or settings.memory_ttl_days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Delete low-importance memories older than cutoff
            await db.execute("""
                DELETE FROM memories 
                WHERE created_at < ? AND importance < 3.0 AND access_count < 2
            """, (cutoff_date.isoformat(),))
            
            await db.commit()
            
        self.logger.info(f"Cleaned up memories older than {days} days")