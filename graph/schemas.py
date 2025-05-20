"""
Schema definitions for the information assistant application.

This file contains TypedDict definitions and other schemas used across the application.
Separating these into their own file helps prevent circular imports.
"""

from typing import Dict, Any, List, TypedDict, Optional


class ConfigSchema(TypedDict, total=False):
    """Configuration schema for the information assistant."""
    api_keys: Dict[str, str]
    model_name: str
    temperature: float
    logging_enabled: bool
    max_retries: int
    custom_options: Dict[str, Any]