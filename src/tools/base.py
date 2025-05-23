"""
Base tool class for consistent tool implementation
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging


class ToolResult(BaseModel):
    """Standardized tool result format"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class BaseTool(ABC):
    """Base class for all tools to ensure consistency"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"tool.{self.name}")
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for the tool"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does"""
        pass
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for tool parameters"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query or request for this tool"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context from the conversation"
                }
            },
            "required": ["query"]
        }
    
    @abstractmethod
    async def _execute_impl(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Implementation of the tool's functionality"""
        pass
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute the tool with error handling and logging
        
        Args:
            query: The user's query or request
            context: Additional context from the conversation
            
        Returns:
            ToolResult with success status and data/error
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing {self.name} with query: {query[:100]}...")
            
            # Add timeout protection
            result = await asyncio.wait_for(
                self._execute_impl(query, context),
                timeout=self.config.get("timeout", 30)
            )
            
            # Add execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            result.metadata.update({
                "tool_name": self.name,
                "execution_time_seconds": execution_time,
                "query_length": len(query)
            })
            
            self.logger.info(f"{self.name} completed successfully in {execution_time:.2f}s")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"{self.name} timed out")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool {self.name} timed out after {self.config.get('timeout', 30)} seconds",
                metadata={"tool_name": self.name}
            )
            
        except Exception as e:
            self.logger.error(f"{self.name} failed with error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool {self.name} encountered an error: {str(e)}",
                metadata={"tool_name": self.name, "error_type": type(e).__name__}
            )
    
    def to_langchain_tool(self):
        """Convert to LangChain tool format"""
        from langchain_core.tools import Tool
        
        async def _run(query: str, context: Optional[Dict[str, Any]] = None) -> str:
            result = await self.execute(query, context)
            if result.success:
                return str(result.data)
            else:
                return f"Error: {result.error}"
        
        return Tool(
            name=self.name,
            description=self.description,
            func=lambda q: asyncio.run(_run(q)),
            coroutine=_run
        )


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)
        
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self._tools.keys())
        
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())
        
    def to_langchain_tools(self) -> List:
        """Convert all tools to LangChain format"""
        return [tool.to_langchain_tool() for tool in self._tools.values()]


# Global tool registry
tool_registry = ToolRegistry()