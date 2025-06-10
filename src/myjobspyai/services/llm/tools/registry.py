"""Tool registry for LLM providers.

This module provides a registry for managing tools that can be used with LLM providers.
The registry allows tools to be registered, looked up by name, and listed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type, TypeVar

from myjobspyai.services.llm.tools.base import BaseTool, ToolError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseTool)


class ToolRegistry:
    """Registry for managing tools that can be used with LLM providers."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, override: bool = False) -> None:
        """Register a tool with the registry.

        Args:
            tool: The tool to register
            override: If True, allow overriding an existing tool with the same name

        Raises:
            ToolError: If a tool with the same name is already registered and override is False
        """
        if not isinstance(tool, BaseTool):
            raise ToolError(f"Expected a BaseTool instance, got {type(tool).__name__}")

        if not tool.name:
            raise ToolError("Tool must have a name")

        if tool.name in self._tools and not override:
            raise ToolError(
                f"Tool with name '{tool.name}' is already registered. "
                "Use override=True to replace it."
            )

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name.

        Args:
            name: The name of the tool to get

        Returns:
            The tool with the given name

        Raises:
            KeyError: If no tool with the given name exists
        """
        if name not in self._tools:
            available = ", ".join(f"'{t}'" for t in self._tools)
            raise KeyError(
                f"No tool with name '{name}' found. "
                f"Available tools: {available or 'none'}"
            )
        return self._tools[name]

    def get_tool_schema(self, name: str) -> dict:
        """Get the JSON schema for a tool by name.

        Args:
            name: The name of the tool

        Returns:
            The JSON schema for the tool

        Raises:
            KeyError: If no tool with the given name exists
        """
        return self.get_tool(name).get_schema()

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools.

        Returns:
            A list of all registered tools
        """
        return list(self._tools.values())

    def list_tool_schemas(self) -> List[dict]:
        """Get the JSON schemas for all registered tools.

        Returns:
            A list of JSON schemas for all registered tools
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister

        Raises:
            KeyError: If no tool with the given name exists
        """
        if name not in self._tools:
            raise KeyError(f"No tool with name '{name}' found")
        del self._tools[name]
        logger.debug(f"Unregistered tool: {name}")

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        logger.debug("Cleared all tools from registry")

    def __contains__(self, name: str) -> bool:
        """Check if a tool with the given name is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())


# Global tool registry instance
_tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry.

    Returns:
        The global tool registry instance
    """
    return _tool_registry


def register_tool(
    tool: Optional[BaseTool] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    override: bool = False,
    **kwargs: Any,
) -> callable:
    """Register a tool with the global registry.

    This can be used as a decorator or as a function.

    Examples:
        # As a decorator
        @register_tool(description="Adds two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        # As a function
        def multiply(a: int, b: int) -> int:
            return a * b

        register_tool(
            multiply,
            name="multiply",
            description="Multiplies two numbers",
        )

    Args:
        tool: The tool to register (when used as a decorator without arguments)
        name: The name of the tool (defaults to function name)
        description: Description of the tool (defaults to function docstring)
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return the tool's output directly
        override: If True, allow overriding an existing tool with the same name
        **kwargs: Additional arguments to pass to the tool constructor

    Returns:
        The decorated function or the tool instance
    """

    def decorator(func_or_tool):
        if isinstance(func_or_tool, BaseTool):
            # If a tool instance was passed directly
            tool_instance = func_or_tool
        else:
            # If a function was passed, create a tool from it
            tool_instance = BaseTool.from_function(
                func=func_or_tool,
                name=name or func_or_tool.__name__,
                description=description or (func_or_tool.__doc__ or ""),
                args_schema=args_schema,
                return_direct=return_direct,
                **kwargs,
            )

        # Register the tool
        _tool_registry.register(tool_instance, override=override)
        return tool_instance

    # Handle the case where the decorator is used with or without arguments
    if tool is None:
        # @register_tool() or @register_tool(name="foo")
        return decorator
    elif callable(tool) and not isinstance(tool, BaseTool):
        # @register_tool
        return decorator(tool)
    else:
        # register_tool(tool_instance)
        decorator(tool)
        return tool


def get_tool(name: str) -> BaseTool:
    """Get a tool by name from the global registry.

    Args:
        name: The name of the tool to get

    Returns:
        The tool with the given name

    Raises:
        KeyError: If no tool with the given name exists
    """
    return _tool_registry.get_tool(name)


def list_tools() -> List[BaseTool]:
    """List all registered tools in the global registry.

    Returns:
        A list of all registered tools
    """
    return _tool_registry.list_tools()


def list_tool_schemas() -> List[dict]:
    """Get the JSON schemas for all registered tools in the global registry.

    Returns:
        A list of JSON schemas for all registered tools
    """
    return _tool_registry.list_tool_schemas()


def unregister_tool(name: str) -> None:
    """Unregister a tool by name from the global registry.

    Args:
        name: The name of the tool to unregister

    Raises:
        KeyError: If no tool with the given name exists
    """
    _tool_registry.unregister(name)


def clear_tools() -> None:
    """Clear all registered tools from the global registry."""
    _tool_registry.clear()
