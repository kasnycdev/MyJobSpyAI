"""
Decorators for adding monitoring and observability to functions and methods.
"""
from __future__ import annotations

import functools
import inspect
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .monitoring import monitor

# Type variables
F = TypeVar('F', bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_metric: bool = True,
    metric_name: Optional[str] = None,
    metric_unit: str = "1",
) -> Callable[[F], F]:
    """
    Decorator to add tracing and metrics to a function.
    
    Args:
        name: Optional name for the span (defaults to function name)
        attributes: Additional attributes to add to the span
        record_metric: Whether to record a metric for this function call
        metric_name: Name for the metric (defaults to 'function.<function_name>')
        metric_unit: Unit for the metric
        
    Returns:
        Decorated function with tracing and metrics
    """
    def decorator(func: F) -> F:
        nonlocal name, metric_name
        
        if name is None:
            name = func.__name__
        
        if metric_name is None:
            metric_name = f"function.{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create span for the function call
            with monitor.start_span(name or func.__name__) as span:
                # Add function info to span
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                
                # Add provided attributes
                if attributes:
                    for key, value in attributes.items():
                        if value is not None:
                            span.set_attribute(key, str(value))
                
                # Record start time for duration
                start_time = time.time()
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute("duration.ms", duration * 1000)
                    
                    # Record metric if enabled
                    if record_metric:
                        monitor.record_histogram(
                            f"{metric_name}.duration", 
                            duration, 
                            unit="s",
                            status="success"
                        )
                    
                    return result
                    
                except Exception as e:
                    # Record the error
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record error metric if enabled
                    if record_metric:
                        monitor.record_metric(
                            f"{metric_name}.errors", 
                            1, 
                            unit="1",
                            error_type=e.__class__.__name__
                        )
                    
                    raise
                
        return cast(F, wrapper)
    return decorator


def traced_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_metric: bool = True,
    metric_name: Optional[str] = None,
    metric_unit: str = "1",
) -> Callable[[F], F]:
    """
    Decorator to add tracing and metrics to an async function.
    
    Args:
        name: Optional name for the span (defaults to function name)
        attributes: Additional attributes to add to the span
        record_metric: Whether to record a metric for this function call
        metric_name: Name for the metric (defaults to 'function.<function_name>')
        metric_unit: Unit for the metric
        
    Returns:
        Decorated async function with tracing and metrics
    """
    def decorator(func: F) -> F:
        nonlocal name, metric_name
        
        if name is None:
            name = func.__name__
        
        if metric_name is None:
            metric_name = f"function.{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create span for the function call
            with monitor.start_span(name or func.__name__) as span:
                # Add function info to span
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                
                # Add provided attributes
                if attributes:
                    for key, value in attributes.items():
                        if value is not None:
                            span.set_attribute(key, str(value))
                
                # Record start time for duration
                start_time = time.time()
                
                try:
                    # Call the async function
                    result = await func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute("duration.ms", duration * 1000)
                    
                    # Record metric if enabled
                    if record_metric:
                        monitor.record_histogram(
                            f"{metric_name}.duration", 
                            duration, 
                            unit="s",
                            status="success"
                        )
                    
                    return result
                    
                except Exception as e:
                    # Record the error
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    # Record error metric if enabled
                    if record_metric:
                        monitor.record_metric(
                            f"{metric_name}.errors", 
                            1, 
                            unit="1",
                            error_type=e.__class__.__name__
                        )
                    
                    raise
                
        return cast(F, async_wrapper)
    return decorator


def traced_method(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_metric: bool = True,
    metric_name: Optional[str] = None,
    metric_unit: str = "1",
) -> Callable[[F], F]:
    """
    Decorator to add tracing and metrics to a method.
    Automatically adds class name to the span name and attributes.
    
    Args:
        name: Optional name for the span (defaults to 'ClassName.method_name')
        attributes: Additional attributes to add to the span
        record_metric: Whether to record a metric for this method call
        metric_name: Name for the metric (defaults to 'method.ClassName.method_name')
        metric_unit: Unit for the metric
        
    Returns:
        Decorated method with tracing and metrics
    """
    def decorator(method: F) -> F:
        nonlocal name, metric_name
        
        method_name = method.__name__
        
        if name is None:
            # Get the class name from the method's qualname
            qualname_parts = method.__qualname__.split('.')
            if len(qualname_parts) > 1:
                class_name = qualname_parts[-2]
                name = f"{class_name}.{method_name}"
            else:
                name = method_name
        
        if metric_name is None:
            metric_name = f"method.{method.__module__}.{method.__qualname__}"
        
        # Check if the method is async
        is_async = inspect.iscoroutinefunction(method)
        
        if is_async:
            @functools.wraps(method)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                with monitor.start_span(name) as span:
                    # Add method and class info
                    span.set_attribute("code.function", method_name)
                    span.set_attribute("code.namespace", method.__module__)
                    span.set_attribute("code.class", self.__class__.__name__)
                    
                    # Add instance attributes if available
                    if hasattr(self, '__dict__'):
                        for attr in ['job_id', 'request_id', 'user_id', 'session_id']:
                            if hasattr(self, attr):
                                span.set_attribute(f"app.{attr}", str(getattr(self, attr)))
                    
                    # Add provided attributes
                    if attributes:
                        for key, value in attributes.items():
                            if value is not None:
                                span.set_attribute(key, str(value))
                    
                    # Record start time for duration
                    start_time = time.time()
                    
                    try:
                        # Call the async method
                        result = await method(self, *args, **kwargs)
                        
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("duration.ms", duration * 1000)
                        
                        # Record metric if enabled
                        if record_metric:
                            monitor.record_histogram(
                                f"{metric_name}.duration", 
                                duration, 
                                unit="s",
                                status="success"
                            )
                        
                        return result
                        
                    except Exception as e:
                        # Record the error
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Record error metric if enabled
                        if record_metric:
                            monitor.record_metric(
                                f"{metric_name}.errors", 
                                1, 
                                unit="1",
                                error_type=e.__class__.__name__
                            )
                        
                        raise
            
            return cast(F, async_wrapper)
        
        else:
            @functools.wraps(method)
            def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                with monitor.start_span(name) as span:
                    # Add method and class info
                    span.set_attribute("code.function", method_name)
                    span.set_attribute("code.namespace", method.__module__)
                    span.set_attribute("code.class", self.__class__.__name__)
                    
                    # Add instance attributes if available
                    if hasattr(self, '__dict__'):
                        for attr in ['job_id', 'request_id', 'user_id', 'session_id']:
                            if hasattr(self, attr):
                                span.set_attribute(f"app.{attr}", str(getattr(self, attr)))
                    
                    # Add provided attributes
                    if attributes:
                        for key, value in attributes.items():
                            if value is not None:
                                span.set_attribute(key, str(value))
                    
                    # Record start time for duration
                    start_time = time.time()
                    
                    try:
                        # Call the method
                        result = method(self, *args, **kwargs)
                        
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("duration.ms", duration * 1000)
                        
                        # Record metric if enabled
                        if record_metric:
                            monitor.record_histogram(
                                f"{metric_name}.duration", 
                                duration, 
                                unit="s",
                                status="success"
                            )
                        
                        return result
                        
                    except Exception as e:
                        # Record the error
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Record error metric if enabled
                        if record_metric:
                            monitor.record_metric(
                                f"{metric_name}.errors", 
                                1, 
                                unit="1",
                                error_type=e.__class__.__name__
                            )
                        
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


def traced_llm_call(
    provider: str,
    model: Optional[str] = None,
    **attrs
) -> Callable[[F], F]:
    """
    Decorator to trace and monitor LLM API calls.
    
    Args:
        provider: LLM provider (e.g., 'openai', 'ollama', 'gemini')
        model: Optional model name (can also be passed as a callable)
        **attrs: Additional attributes to add to the span and metrics
        
    Returns:
        Decorated function with LLM call tracing and monitoring
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine model name (can be a callable or a string)
            model_name = model
            if callable(model_name):
                model_name = model_name(*args, **kwargs)
            
            # Create span for the LLM call
            span_name = f"llm.{provider}.{model_name or 'unknown'}.{func.__name__}"
            
            with monitor.start_span(span_name) as span:
                # Add LLM provider and model info
                span.set_attribute("llm.provider", provider)
                if model_name:
                    span.set_attribute("llm.model", model_name)
                
                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                
                # Add provided attributes
                for key, value in attrs.items():
                    if value is not None:
                        span.set_attribute(f"llm.{key}", str(value))
                
                # Record start time for duration
                start_time = time.time()
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    
                    # Record LLM call metrics
                    monitor.record_llm_call(
                        provider=provider,
                        model=model_name or "unknown",
                        duration=duration,
                        success=True,
                        **attrs
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record the error
                    duration = time.time() - start_time
                    error_type = e.__class__.__name__
                    
                    monitor.record_llm_call(
                        provider=provider,
                        model=model_name or "unknown",
                        duration=duration,
                        success=False,
                        error_type=error_type,
                        **attrs
                    )
                    
                    # Re-raise the exception
                    raise
        
        return cast(F, wrapper)
    return decorator


def traced_rag_operation(
    operation: str,
    collection: Optional[str] = None,
    **attrs
) -> Callable[[F], F]:
    """
    Decorator to trace and monitor RAG operations.
    
    Args:
        operation: Type of RAG operation (e.g., 'retrieve', 'index', 'search')
        collection: Optional collection name (can also be passed as a callable)
        **attrs: Additional attributes to add to the span and metrics
        
    Returns:
        Decorated function with RAG operation tracing and monitoring
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine collection name (can be a callable or a string)
            collection_name = collection
            if callable(collection_name):
                collection_name = collection_name(*args, **kwargs)
            
            # Create span for the RAG operation
            span_name = f"rag.{operation}"
            if collection_name:
                span_name = f"{span_name}.{collection_name}"
            
            with monitor.start_span(span_name) as span:
                # Add RAG operation info
                span.set_attribute("rag.operation", operation)
                if collection_name:
                    span.set_attribute("rag.collection", collection_name)
                
                # Add function info
                span.set_attribute("code.function", func.__name__)
                span.set_attribute("code.namespace", func.__module__)
                
                # Add provided attributes
                for key, value in attrs.items():
                    if value is not None:
                        span.set_attribute(f"rag.{key}", str(value))
                
                # Record start time for duration
                start_time = time.time()
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Record duration
                    duration = time.time() - start_time
                    
                    # Record RAG operation metrics
                    monitor.record_rag_retrieval(
                        query_type=operation,
                        document_count=len(result) if hasattr(result, '__len__') else 1,
                        duration=duration,
                        collection=collection_name or "default",
                        **attrs
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record the error
                    duration = time.time() - start_time
                    error_type = e.__class__.__name__
                    
                    monitor.record_metric(
                        "rag.errors",
                        1,
                        unit="1",
                        operation=operation,
                        error_type=error_type,
                        **attrs
                    )
                    
                    # Re-raise the exception
                    raise
        
        return cast(F, wrapper)
    return decorator
