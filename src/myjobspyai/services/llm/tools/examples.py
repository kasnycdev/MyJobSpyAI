"""Example tools for demonstration purposes.

This module contains example tools that demonstrate how to create and use tools
with the LLM provider system.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from myjobspyai.services.llm.tools import BaseTool, ToolError, register_tool, tool


# Example 1: Basic tool using the decorator
@tool(description="Adds two numbers together")
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


# Example 2: Tool with custom name and description
@tool(
    name="multiply_numbers",
    description="Multiplies two numbers together",
)
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b


# Example 3: Tool with Pydantic model for input validation
class WeatherInput(BaseModel):
    """Input for the get_weather tool."""

    location: str = Field(
        ..., description="The city and state, e.g., 'San Francisco, CA'"
    )
    unit: str = Field(
        "celsius",
        description="The temperature unit to use ('celsius' or 'fahrenheit')",
        regex="^(celsius|fahrenheit)$",
    )


@tool(args_schema=WeatherInput, description="Get the current weather in a location")
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a location.

    This is a mock implementation that returns a fixed response.
    In a real application, this would call a weather API.

    Args:
        location: The city and state, e.g., 'San Francisco, CA'
        unit: The temperature unit to use ('celsius' or 'fahrenheit')

    Returns:
        A string describing the current weather
    """
    # This is a mock implementation
    return (
        f"The weather in {location} is 22°{unit[0].upper()}, sunny with a light breeze."
    )


# Example 4: Tool that returns directly (no further processing by the LLM)
@tool(return_direct=True, description="Get the current time in a timezone")
def get_time(timezone: str = "UTC") -> str:
    """Get the current time in a timezone.

    This is a mock implementation that returns a fixed response.
    In a real application, this would use the datetime module.

    Args:
        timezone: The timezone to get the time for (e.g., 'America/New_York')

    Returns:
        A string with the current time in the specified timezone
    """
    # This is a mock implementation
    return f"The current time in {timezone} is 10:30 AM"


# Example 5: Tool that raises an error
@tool(description="This tool always raises an error")
def error_tool() -> str:
    """Raise an error.

    This tool is used to demonstrate error handling.

    Raises:
        ToolError: Always raises this error
    """
    raise ToolError("This is a test error from error_tool")


# Example 6: Tool that uses a class-based approach
class CalculatorTool(BaseTool):
    """A calculator tool that can perform basic arithmetic operations."""

    name: str = "calculator"
    description: str = "Performs basic arithmetic operations"

    class ArgsSchema(BaseModel):
        expression: str = Field(
            ..., description="The arithmetic expression to evaluate, e.g., '2 + 2 * 3'"
        )

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, expression: str) -> float:
        """Evaluate an arithmetic expression.

        Args:
            expression: The arithmetic expression to evaluate

        Returns:
            The result of the evaluation

        Raises:
            ToolError: If the expression is invalid
        """
        try:
            # WARNING: Using eval() is dangerous in production code!
            # This is just an example. In a real application, you would want to
            # use a safe expression evaluator or implement the logic yourself.
            return float(eval(expression))  # nosec
        except Exception as e:
            raise ToolError(f"Failed to evaluate expression: {e}") from e


# Register the class-based tool
calculator_tool = CalculatorTool()


# Example 7: Tool that processes a list of items
class ProcessItemsInput(BaseModel):
    """Input for the process_items tool."""

    items: List[str] = Field(..., description="List of items to process")
    operation: str = Field(
        "reverse",
        description="Operation to perform on the items ('reverse', 'sort', 'shuffle')",
        regex="^(reverse|sort|shuffle)$",
    )


@tool(args_schema=ProcessItemsInput, description="Process a list of items")
def process_items(items: List[str], operation: str = "reverse") -> List[str]:
    """Process a list of items with the specified operation.

    Args:
        items: List of items to process
        operation: Operation to perform ('reverse', 'sort', 'shuffle')

    Returns:
        The processed list of items
    """
    if operation == "reverse":
        return items[::-1]
    elif operation == "sort":
        return sorted(items)
    elif operation == "shuffle":
        import random

        shuffled = items.copy()
        random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Example 8: Tool that makes an HTTP request
@tool(description="Fetch data from a URL")
def fetch_url(
    url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None
) -> str:
    """Fetch data from a URL.

    This is a simplified example. In a real application, you would want to add
    error handling, timeouts, retries, etc.

    Args:
        url: The URL to fetch
        method: The HTTP method to use (GET, POST, etc.)
        headers: Optional HTTP headers

    Returns:
        The response body as a string
    """
    import requests

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers or {},
            timeout=10,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise ToolError(f"Failed to fetch URL: {e}") from e


def register_example_tools() -> None:
    """Register all example tools with the global registry."""
    # The @tool decorator automatically registers the tools,
    # but we need to import them to trigger the registration.
    # This function is provided for explicitness.
    pass
