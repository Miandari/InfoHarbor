"""
Configuration settings for the Elderly Assistant Agent
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # LLM Settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Memory Settings
    memory_db_path: str = "data/memory.db"
    memory_search_k: int = 5
    memory_ttl_days: int = 365  # How long to keep memories
    
    # Agent Settings
    agent_name: str = "Elderly Care Assistant"
    agent_timeout_seconds: int = 60
    max_conversation_turns: int = 20
    
    # Personality Settings
    personality_prompt_path: str = "config/prompts.yaml"
    elder_mode: bool = True  # Enable elder-friendly features
    
    # Development Settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = "INFO"
    
    # Render Deployment
    render_external_url: Optional[str] = os.getenv("RENDER_EXTERNAL_URL")
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"  # Prefix for env vars
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()