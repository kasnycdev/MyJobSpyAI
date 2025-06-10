"""Tool system for LLM providers.

This module provides a way to define, discover, and use tools with LLM providers.
Tools are functions that can be called by the model to perform specific tasks.
"""

from myjobspyai.services.llm.tools.base import (
    BaseTool,
    Tool,
    ToolCall,
    ToolError,
    ToolValidationError,
    tool,
)
from myjobspyai.services.llm.tools.registry import (
    ToolRegistry,
    clear_tools,
    get_tool,
    get_tool_registry,
    list_tool_schemas,
    list_tools,
    register_tool,
    unregister_tool,
)

# Re-export the most commonly used symbols
__all__ = [
    # Base classes
    'BaseTool',
    'Tool',
    'ToolCall',
    'ToolError',
    'ToolValidationError',
    'ToolRegistry',
    # Decorators
    'tool',
    'register_tool',
    # Registry functions
    'get_tool_registry',
    'get_tool',
    'list_tools',
    'list_tool_schemas',
    'unregister_tool',
    'clear_tools',
]
