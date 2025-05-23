"""
Shared utilities and helper functions for Elder Care Assistant
"""
import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path


def sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent security issues"""
    if not text:
        return ""
    
    # Remove potential harmful characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    max_length = 1000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def generate_user_id(name: str, email: str = "") -> str:
    """Generate a consistent user ID from name and email"""
    combined = f"{name.lower()}{email.lower()}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def format_datetime(dt: datetime, format_type: str = "friendly") -> str:
    """Format datetime for elderly-friendly display"""
    if format_type == "friendly":
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            if diff.seconds < 3600:  # Less than 1 hour
                minutes = diff.seconds // 60
                return f"{minutes} minutes ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours} hours ago"
        elif diff.days == 1:
            return "yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%B %d, %Y")
    
    elif format_type == "time_only":
        return dt.strftime("%I:%M %p")  # 2:30 PM
    
    elif format_type == "date_only":
        return dt.strftime("%B %d, %Y")  # January 15, 2025
    
    else:  # default
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_health_data(health_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize health information"""
    validated = {}
    
    # Allowed health fields with their types
    allowed_fields = {
        "medications": list,
        "allergies": list,
        "conditions": list,
        "emergency_contact": str,
        "doctor_name": str,
        "doctor_phone": str,
        "insurance_info": str
    }
    
    for field, expected_type in allowed_fields.items():
        if field in health_info:
            value = health_info[field]
            if isinstance(value, expected_type):
                if expected_type == str:
                    validated[field] = sanitize_user_input(value)
                elif expected_type == list:
                    validated[field] = [sanitize_user_input(str(item)) for item in value]
    
    return validated


def chunk_text(text: str, max_length: int = 100, overlap: int = 20) -> List[str]:
    """Split text into chunks for better processing"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at a sentence boundary
        if end < len(text):
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def extract_keywords(text: str, elderly_focused: bool = True) -> List[str]:
    """Extract relevant keywords from text"""
    # Common elderly care keywords
    elderly_keywords = {
        'health', 'medication', 'doctor', 'appointment', 'family',
        'grandchildren', 'memory', 'exercise', 'social', 'help',
        'reminder', 'pharmacy', 'insurance', 'retirement'
    }
    
    # Extract words and filter
    words = re.findall(r'\b\w+\b', text.lower())
    
    if elderly_focused:
        keywords = [word for word in words if word in elderly_keywords]
    else:
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(keywords))


def create_safe_filename(filename: str) -> str:
    """Create a safe filename by removing potentially harmful characters"""
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(safe_name) > 100:
        name_part, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
        safe_name = name_part[:90] + ('.' + ext if ext else '')
    
    return safe_name


def calculate_urgency_score(text: str, user_context: Dict[str, Any] = None) -> float:
    """Calculate urgency score for elderly care context"""
    urgency_indicators = {
        'emergency': 1.0,
        'urgent': 0.9,
        'pain': 0.8,
        'can\'t breathe': 1.0,
        'chest pain': 0.9,
        'fell': 0.7,
        'dizzy': 0.6,
        'confused': 0.5,
        'medication': 0.4,
        'doctor': 0.3,
        'appointment': 0.2
    }
    
    text_lower = text.lower()
    max_urgency = 0.0
    
    for indicator, score in urgency_indicators.items():
        if indicator in text_lower:
            max_urgency = max(max_urgency, score)
    
    # Adjust based on user context
    if user_context:
        # Increase urgency for users with health conditions
        if user_context.get('health_conditions'):
            max_urgency = min(1.0, max_urgency + 0.1)
        
        # Increase urgency for elderly users (age > 75)
        if user_context.get('age', 0) > 75:
            max_urgency = min(1.0, max_urgency + 0.1)
    
    return max_urgency


def retry_async(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
            
            raise last_exception
        return wrapper
    return decorator


def load_json_safely(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load JSON file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}


def save_json_safely(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Safely save JSON file with error handling"""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception:
        return False


def mask_sensitive_info(text: str) -> str:
    """Mask sensitive information in text for logging/display"""
    # Mask phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX', text)
    
    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email@masked.com', text)
    
    # Mask potential SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', text)
    
    # Mask credit card numbers
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', text)
    
    return text


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if the identifier is allowed to make a call"""
        now = datetime.now()
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.time_window)
        self.calls = {k: v for k, v in self.calls.items() 
                     if any(call_time > cutoff for call_time in v)}
        
        # Check current calls for identifier
        if identifier not in self.calls:
            self.calls[identifier] = []
        
        # Remove old calls for this identifier
        self.calls[identifier] = [call_time for call_time in self.calls[identifier] 
                                 if call_time > cutoff]
        
        # Check if under limit
        if len(self.calls[identifier]) < self.max_calls:
            self.calls[identifier].append(now)
            return True
        
        return False