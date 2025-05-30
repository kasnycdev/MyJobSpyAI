"""Utility functions for retry logic."""

from typing import Callable, TypeVar, ParamSpec
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

P = ParamSpec("P")
T = TypeVar("T")

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    reraise: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add retry logic to a function.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to retry on
        reraise: Whether to re-raise the last exception if all retries fail

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            reraise=reraise
        )(func)
    return decorator
