"""Base tool interface and registry for LLM providers.

This module provides the core tool interface and registry system for tools that
can be used with LLM providers. Tools are callable objects that can be used by
LLMs to perform specific tasks.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseTool')


class ToolError(Exception):
    """Base exception for tool-related errors."""

    pass


class ToolValidationError(ToolError):
    """Raised when a tool fails validation."""

    pass


@dataclass
class ToolCall:
    """Represents a call to a tool.

    Attributes:
        name: The name of the tool being called
        args: The arguments to pass to the tool
        id: An optional ID for the tool call
    """

    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


class BaseTool(ABC, BaseModel):
    """Base class for tools that can be used with LLM providers.

    Tools are callable objects that can be used by LLMs to perform specific tasks.
    Each tool has a name, description, and schema defining its input parameters.
    """

    name: str = Field(
        ...,
        description="The name of the tool (must be unique within a registry)",
        min_length=1,
        max_length=64,
        regex=r'^[a-zA-Z0-9_-]+$',
    )

    description: str = Field(
        ...,
        description="A clear, concise description of what the tool does and when to use it",
        min_length=1,
        max_length=512,
    )

    args_schema: Type[BaseModel] = Field(
        default_factory=lambda: BaseModel,
        description="Pydantic model for validating tool input arguments",
    )

    return_direct: bool = Field(
        default=False,
        description="If True, the tool's output will be returned directly without further processing",
    )

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        This method must be implemented by subclasses to define the tool's behavior.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the tool execution
        """
        raise NotImplementedError("Tool must implement _run")

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with the given arguments.

        By default, this runs the sync version in a thread pool. Override this
        method to provide an async implementation.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the tool execution
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run, *args, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        This is the main entry point for using the tool. It validates the input
        against the tool's schema before executing the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolValidationError: If the input fails validation
            ToolError: If the tool execution fails
        """
        try:
            # Validate input against schema if provided
            if self.args_schema and self.args_schema is not BaseModel:
                try:
                    validated = (
                        self.args_schema(**kwargs)
                        if kwargs
                        else self.args_schema(*args)
                    )
                    kwargs = validated.dict(exclude_unset=True)
                except Exception as e:
                    raise ToolValidationError(
                        f"Invalid input for tool '{self.name}': {str(e)}"
                    ) from e

            return self._run(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{self.name}': {str(e)}", exc_info=True)
            raise ToolError(f"Tool execution failed: {str(e)}") from e

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool asynchronously with the given arguments.

        This is the async version of __call__.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the tool execution

        Raises:
            ToolValidationError: If the input fails validation
            ToolError: If the tool execution fails
        """
        try:
            # Validate input against schema if provided
            if self.args_schema and self.args_schema is not BaseModel:
                try:
                    validated = (
                        self.args_schema(**kwargs)
                        if kwargs
                        else self.args_schema(*args)
                    )
                    kwargs = validated.dict(exclude_unset=True)
                except Exception as e:
                    raise ToolValidationError(
                        f"Invalid input for tool '{self.name}': {str(e)}"
                    ) from e

            return await self._arun(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool '{self.name}': {str(e)}", exc_info=True)
            raise ToolError(f"Tool execution failed: {str(e)}") from e

    def get_schema(self) -> dict:
        """Get the JSON schema for the tool's input.

        Returns:
            A dictionary containing the tool's schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": (
                self.args_schema.schema() if self.args_schema is not BaseModel else {}
            ),
            "return_direct": self.return_direct,
        }

    @classmethod
    def from_function(
        cls: Type[T],
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        **kwargs: Any,
    ) -> T:
        """Create a tool from a function.

        This is a convenience method for creating a tool from a function.

        Args:
            func: The function to convert to a tool
            name: The name of the tool (defaults to function name)
            description: Description of the tool (defaults to function docstring)
            args_schema: Pydantic model for argument validation
            return_direct: Whether to return the tool's output directly
            **kwargs: Additional arguments to pass to the tool constructor

        Returns:
            A tool instance
        """
        if name is None:
            name = func.__name__
        if description is None:
            description = inspect.getdoc(func) or ""

        # Create args schema from function signature if not provided
        if args_schema is None:
            sig = inspect.signature(func)
            fields = {}

            for param in sig.parameters.values():
                if param.name == 'self':
                    continue

                # Get type annotation, default to Any if not provided
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    param_type = Any

                # Get default value if any
                default = (
                    ... if param.default == inspect.Parameter.empty else param.default
                )

                fields[param.name] = (param_type, Field(default=default))

            if fields:
                args_schema = create_model(
                    f"{name.capitalize()}Args",
                    **fields,
                )
            else:
                args_schema = BaseModel

        # Create a subclass of BaseTool that implements _run and _arun
        class FunctionTool(BaseTool):
            def _run(self, **kwargs: Any) -> Any:
                return func(**kwargs)

            async def _arun(self, **kwargs: Any) -> Any:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                return await super()._arun(**kwargs)

        # Set the name and description for the class
        FunctionTool.__name__ = f"{name.capitalize()}Tool"
        FunctionTool.__doc__ = description

        # Create and return an instance of the tool
        return FunctionTool(
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            **kwargs,
        )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
    **kwargs: Any,
) -> Callable[[Callable], BaseTool]:
    """Decorator to convert a function into a tool.

    Example::

        @tool(description="Adds two numbers")
        def add(a: int, b: int) -> int:
            return a + b

    Args:
        name: The name of the tool (defaults to function name)
        description: Description of the tool (defaults to function docstring)
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return the tool's output directly
        **kwargs: Additional arguments to pass to the tool constructor

    Returns:
        A decorator that converts a function into a tool
    """

    def decorator(func: Callable) -> BaseTool:
        tool_name = name or func.__name__
        tool_description = description or (inspect.getdoc(func) or "")

        return BaseTool.from_function(
            func=func,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            return_direct=return_direct,
            **kwargs,
        )

    return decorator
