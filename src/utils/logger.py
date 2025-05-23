"""
Logging configuration for Elder Care Assistant
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup logging configuration for the application"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific loggers for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


class ElderCareLogger:
    """Custom logger for Elder Care Assistant with specific features"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_user_interaction(self, user_id: str, action: str, details: str = ""):
        """Log user interactions with privacy considerations"""
        # Anonymize sensitive information
        safe_details = self._anonymize_sensitive_data(details)
        self.logger.info(
            f"User interaction - User: {user_id[:8]}... Action: {action} Details: {safe_details}"
        )
    
    def log_health_query(self, user_id: str, query_type: str, urgency: str = "low"):
        """Log health-related queries with appropriate urgency"""
        log_level = logging.WARNING if urgency == "high" else logging.INFO
        self.logger.log(
            log_level,
            f"Health query - User: {user_id[:8]}... Type: {query_type} Urgency: {urgency}"
        )
    
    def log_memory_operation(self, user_id: str, operation: str, memory_type: str):
        """Log memory operations"""
        self.logger.info(
            f"Memory operation - User: {user_id[:8]}... Operation: {operation} Type: {memory_type}"
        )
    
    def log_tool_usage(self, user_id: str, tool_name: str, success: bool, execution_time: float = 0):
        """Log tool usage statistics"""
        status = "success" if success else "failed"
        self.logger.info(
            f"Tool usage - User: {user_id[:8]}... Tool: {tool_name} Status: {status} Time: {execution_time:.3f}s"
        )
    
    def _anonymize_sensitive_data(self, text: str) -> str:
        """Remove or anonymize sensitive information from logs"""
        import re
        
        # Remove potential medication names, phone numbers, addresses
        anonymized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        anonymized = re.sub(r'\b\d{1,5}\s\w+\s(st|street|ave|avenue|rd|road)\b', '[ADDRESS]', anonymized, flags=re.IGNORECASE)
        
        # Limit length to prevent log flooding
        if len(anonymized) > 200:
            anonymized = anonymized[:200] + "..."
        
        return anonymized