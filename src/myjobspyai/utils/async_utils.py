"""Asynchronous utility functions for MyJobSpy AI."""

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import Any, Optional, TypeVar, Union, cast, overload

from typing_extensions import ParamSpec, TypeGuard

logger = logging.getLogger(__name__)


T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def is_async_callable(obj: Any) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    """Check if an object is an async callable (function or method)."""
    if inspect.iscoroutinefunction(obj):
        return True

    if inspect.isclass(obj):
        return False

    if callable(obj):
        # Check for objects with __call__ method
        if hasattr(obj, "__call__"):
            return is_async_callable(obj.__call__)  # type: ignore
        return asyncio.iscoroutinefunction(obj)

    return False


def sync_to_async(
    func: Callable[P, R],
    *,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Callable[P, Awaitable[R]]:
    """Convert a synchronous function to an asynchronous function.

    Args:
        func: The synchronous function to convert.
        executor: Optional ThreadPoolExecutor to use. If None, a new one will be created.

    Returns:
        An asynchronous function that runs the synchronous function in a thread pool.
    """
    if is_async_callable(func):
        return cast(Callable[P, Awaitable[R]], func)

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        loop = asyncio.get_running_loop()
        func_with_args = partial(func, *args, **kwargs)

        if executor is None:
            # Use the default executor if none provided
            return await loop.run_in_executor(None, func_with_args)
        else:
            return await loop.run_in_executor(executor, func_with_args)

    return wrapper


@overload
async def run_async(
    func: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R: ...


@overload
async def run_async(
    func: Callable[P, Awaitable[R]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R: ...


async def run_async(
    func: Union[Callable[P, R], Callable[P, Awaitable[R]]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run a function asynchronously, whether it's sync or async.

    Args:
        func: The function to run (can be sync or async).
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.
    """
    if is_async_callable(func):
        return await cast(Callable[P, Awaitable[R]], func)(*args, **kwargs)
    else:
        # For synchronous functions, run in a thread pool
        loop = asyncio.get_running_loop()
        func_with_args = partial(cast(Callable[P, R], func), *args, **kwargs)
        return await loop.run_in_executor(None, func_with_args)


async def gather_with_concurrency(
    n: int,
    *tasks: Union[Awaitable[T], Callable[[], Awaitable[T]]],
    return_exceptions: bool = False,
) -> list[T]:
    """Run coroutines with limited concurrency.

    Args:
        n: Maximum number of concurrent tasks.
        *tasks: Coroutines or callables that return coroutines.
        return_exceptions: If True, exceptions are treated the same as successful results.

    Returns:
        List of results in the same order as the input tasks.
    """
    semaphore = asyncio.Semaphore(n)

    async def run_task(task: Union[Awaitable[T], Callable[[], Awaitable[T]]]) -> T:
        async with semaphore:
            if callable(task):
                task = task()
            return await task

    return await asyncio.gather(
        *(run_task(task) for task in tasks),
        return_exceptions=return_exceptions,
    )


async def async_filter(
    predicate: Callable[[T], Union[bool, Awaitable[bool]]],
    iterable: Union[Iterable[T], AsyncGenerator[T, None]],
) -> AsyncGenerator[T, None]:
    """Asynchronously filter an iterable or async iterable.

    Args:
        predicate: A function that returns a boolean or a coroutine that returns a boolean.
        iterable: An iterable or async iterable to filter.

    Yields:
        Items from the iterable for which the predicate returns True.
    """
    if inspect.isasyncgen(iterable):
        async for item in iterable:
            result = predicate(item)
            if inspect.isawaitable(result):
                result = await result
            if result:
                yield item
    else:
        for item in iterable:
            result = predicate(item)
            if inspect.isawaitable(result):
                result = await result
            if result:
                yield item


async def async_map(
    func: Callable[[T], Union[R, Awaitable[R]]],
    iterable: Union[Iterable[T], AsyncGenerator[T, None]],
) -> AsyncGenerator[R, None]:
    """Asynchronously map a function over an iterable or async iterable.

    Args:
        func: A function to apply to each item (can be async).
        iterable: An iterable or async iterable to map over.

    Yields:
        The result of applying func to each item.
    """
    if inspect.isasyncgen(iterable):
        async for item in iterable:
            result = func(item)
            if inspect.isawaitable(result):
                result = await result
            yield result
    else:
        for item in iterable:
            result = func(item)
            if inspect.isawaitable(result):
                result = await result
            yield result


async def async_enumerate(
    async_iterable: AsyncGenerator[T, None],
    start: int = 0,
) -> AsyncGenerator[tuple[int, T], None]:
    """Asynchronously enumerate an async iterable.

    Args:
        async_iterable: The async iterable to enumerate.
        start: The starting index.

    Yields:
        Tuples of (index, item) pairs.
    """
    index = start
    async for item in async_iterable:
        yield index, item
        index += 1


async def async_sleep(
    delay: float,
    result: Optional[T] = None,
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> T:
    """Asynchronous sleep with optional result.

    Args:
        delay: Number of seconds to sleep.
        result: Optional result to return after sleeping.
        loop: Optional event loop to use.

    Returns:
        The result, if provided.
    """
    if loop is None:
        loop = asyncio.get_running_loop()
    await asyncio.sleep(delay)
    return result


class AsyncLock:
    """Asynchronous lock context manager."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, *args: Any):
        self._lock.release()


class AsyncEvent:
    """Asynchronous event that can be waited on."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    async def set(self) -> None:
        """Set the event, waking up all waiters."""
        self._event.set()

    async def clear(self) -> None:
        """Clear the event."""
        self._event.clear()

    async def is_set(self) -> bool:
        """Return True if the event is set."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait until the event is set."""
        await self._event.wait()


async def cancel_task(task: asyncio.Task[Any]) -> None:
    """Cancel an asyncio task and wait for it to complete.

    Args:
        task: The task to cancel.
    """
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
