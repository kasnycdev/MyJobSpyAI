"""
Monitoring and Observability Module

This module provides comprehensive monitoring capabilities using OpenTelemetry
for tracing, metrics, and logging across the application. It supports:
- Distributed tracing with context propagation
- Custom metrics collection and aggregation
- Log correlation with trace context
- Automatic instrumentation of LLM and RAG operations
- Resource usage monitoring
- Error tracking and reporting
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import json
import logging
import inspect
import functools
import threading
import contextlib
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    AsyncContextManager,
    ContextManager,
    AsyncIterator,
    Awaitable,
    Tuple,
    Set,
    MutableMapping,
)

from opentelemetry import trace, metrics
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
    NonRecordingSpan,
    SpanKind,
    format_span_id,
    format_trace_id,
    get_current_span,
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    use_span,
)
from opentelemetry.metrics import (
    Counter,
    Histogram,
    Meter,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    get_meter_provider,
    set_meter_provider,
    CallbackOptions,
    Observation,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
    MetricExporter,
    MetricExportResult,
    MetricsData,
    AggregationTemporality,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, DEPLOYMENT_ENVIRONMENT, ResourceAttributes
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.context import (
    Context,
    attach,
    detach,
    get_current,
    set_value,
    get_value,
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from opentelemetry.util.types import AttributeValue, AttributeKey
from opentelemetry.semconv.resource import ResourceAttributes as SemConvResourceAttributes
from opentelemetry.semconv.trace import SpanAttributes

from .otel_config import get_otel_config
from ..exceptions import AnalysisError

# Type variables for generic type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SERVICE_NAME = "myjobspyai"
DEFAULT_NAMESPACE = "myjobspyai"
DEFAULT_RESOURCE_ATTRIBUTES = {
    "service.namespace": DEFAULT_NAMESPACE,
    "telemetry.sdk.name": "opentelemetry",
    "telemetry.sdk.language": "python",
    "telemetry.sdk.version": "1.22.0",
}

# Enums
class MetricUnit(str, Enum):
    """Standard metric units."""
    NONE = "1"
    MS = "ms"
    S = "s"
    BYTES = "by"
    KB = "kb"
    MB = "mb"
    GB = "gb"
    PERCENT = "%"
    TOKENS = "tokens"
    REQUESTS = "requests"
    ERRORS = "errors"
    REQUESTS_PER_SECOND = "rps"
    UTILIZATION = "utilization"

class SpanStatus(str, Enum):
    """Standard span status values."""
    OK = "ok"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    UNAUTHENTICATED = "unauthenticated"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    FAILED_PRECONDITION = "failed_precondition"
    ABORTED = "aborted"
    OUT_OF_RANGE = "out_of_range"
    UNIMPLEMENTED = "unimplemented"
    INTERNAL = "internal"
    DATA_LOSS = "data_loss"

# Context keys
_MONITOR_ACTIVE_SPAN = "monitor_active_span"
_MONITOR_SPAN_STACK = "monitor_span_stack"

# Thread-local storage
_thread_local = threading.local()


class Monitor:
    """
    Centralized monitoring and observability class for the application.
    Handles tracing, metrics, and logging in a unified way.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Monitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, service_name: str = "myjobspyai", config: Optional[dict] = None):
        if self._initialized:
            return
            
        self._initialized = True
        self.service_name = service_name
        self.config = config or {}
        
        # Initialize OpenTelemetry components
        self._initialize_opentelemetry()
        
        # Create meters and metrics
        self._initialize_metrics()
        
        logger.info(f"Monitoring initialized for service: {service_name}")
    
    def _initialize_opentelemetry(self) -> None:
        """Initialize OpenTelemetry components."""
        try:
            # Get configuration
            config = get_otel_config()
            
            # Set up resource
            self.resource = Resource.create(
                {
                    SERVICE_NAME: self.service_name,
                    DEPLOYMENT_ENVIRONMENT: config.get('deployment_environment', 'development'),
                    "service.version": config.get('service_version', '1.0.0'),
                    **config.get('resource_attributes', {})
                }
            )
            
            # Initialize tracing
            self._initialize_tracing()
            
            # Initialize metrics
            self._initialize_metrics_provider()
            
            logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
            # Fall back to no-op implementations
            self.tracer = trace.NoOpTracer()
            self.meter = NoOpMeter()
    
    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        config = get_otel_config()
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=self.resource,
        )
        
        # Configure exporters
        if config.get('otlp_endpoint'):
            # OTLP exporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=config['otlp_endpoint'],
                headers=config.get('otlp_headers', {}),
                timeout=config.get('otlp_timeout', 10.0),
                compression=config.get('otlp_compression'),
                insecure=config.get('otlp_insecure', False),
            )
            
            # Add batch processor
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(
                    otlp_exporter,
                    max_export_batch_size=config.get('max_export_batch_size', 512),
                    schedule_delay_millis=config.get('scheduled_delay_millis', 5000),
                    export_timeout_millis=config.get('export_timeout_millis', 30000),
                    max_queue_size=config.get('max_queue_size', 2048),
                )
            )
        
        # Always add console exporter in debug mode
        if config.get('debug', False):
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                SimpleSpanProcessor(console_exporter)
            )
        
        # Set the tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        self.tracer = trace.get_tracer(self.service_name)
    
    def _initialize_metrics_provider(self) -> None:
        """Initialize OpenTelemetry metrics provider."""
        config = get_otel_config()
        
        if not config.get('metrics_enabled', True):
            self.meter = NoOpMeter()
            return
        
        # Create metric readers
        readers = []
        
        # OTLP metrics exporter if endpoint is configured
        if config.get('otlp_endpoint'):
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            
            otlp_exporter = OTLPMetricExporter(
                endpoint=config['otlp_endpoint'],
                headers=config.get('otlp_headers', {}),
                timeout=config.get('otlp_timeout', 10.0),
                compression=config.get('otlp_compression'),
                insecure=config.get('otlp_insecure', False),
            )
            
            readers.append(
                PeriodicExportingMetricReader(
                    exporter=otlp_exporter,
                    export_interval_millis=config.get('metrics_export_interval_millis', 60000),
                    export_timeout_millis=config.get('metrics_export_timeout_millis', 30000),
                )
            )
        
        # Add console exporter in debug mode
        if config.get('debug', False):
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=5000,  # More frequent in debug mode
                )
            )
        
        # Create and set meter provider
        self.meter_provider = MeterProvider(
            resource=self.resource,
            metric_readers=readers,
        )
        set_meter_provider(self.meter_provider)
        self.meter = self.meter_provider.get_meter(self.service_name)
    
    def _initialize_metrics(self) -> None:
        """Initialize application metrics."""
        # LLM metrics
        self.llm_requests = self.meter.create_counter(
            "llm.requests.total",
            unit="1",
            description="Total number of LLM requests",
        )
        
        self.llm_errors = self.meter.create_counter(
            "llm.errors.total",
            unit="1",
            description="Total number of LLM errors",
        )
        
        self.llm_duration = self.meter.create_histogram(
            "llm.duration.seconds",
            unit="s",
            description="Duration of LLM requests in seconds",
        )
        
        # RAG metrics
        self.rag_retrieval_duration = self.meter.create_histogram(
            "rag.retrieval.duration.seconds",
            unit="s",
            description="Duration of RAG retrieval operations in seconds",
        )
        
        self.rag_retrieval_count = self.meter.create_counter(
            "rag.retrieval.count",
            unit="1",
            description="Number of RAG retrieval operations",
        )
        
        # Job processing metrics
        self.jobs_processed = self.meter.create_counter(
            "jobs.processed.total",
            unit="1",
            description="Total number of jobs processed",
        )
        
        self.job_processing_duration = self.meter.create_histogram(
            "jobs.processing.duration.seconds",
            unit="s",
            description="Duration of job processing in seconds",
        )
        
        # Vector DB metrics
        self.vectordb_operations = self.meter.create_counter(
            "vectordb.operations.total",
            unit="1",
            description="Total number of vector database operations",
        )
        
        self.vectordb_duration = self.meter.create_histogram(
            "vectordb.operations.duration.seconds",
            unit="s",
            description="Duration of vector database operations in seconds",
        )
    
    def trace(self, name: Optional[str] = None, **attrs) -> Callable[[F], F]:
        """
        Decorator to trace a function with OpenTelemetry.
        
        Args:
            name: Optional name for the span (defaults to function name)
            **attrs: Additional attributes to add to the span
            
        Returns:
            Decorated function with tracing
        """
        def decorator(func: F) -> F:
            nonlocal name
            
            if name is None:
                name = func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    # Add attributes
                    for key, value in attrs.items():
                        if value is not None:
                            span.set_attribute(key, str(value))
                    
                    # Add function info
                    span.set_attribute("code.function", func.__name__)
                    span.set_attribute("code.namespace", func.__module__)
                    
                    try:
                        # Record start time for duration
                        start_time = time.time()
                        
                        # Call the function
                        result = func(*args, **kwargs)
                        
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("duration.ms", duration * 1000)
                        
                        return result
                        
                    except Exception as e:
                        # Record the error
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Increment error counter if it's an LLM error
                        if isinstance(e, AnalysisError):
                            self.llm_errors.add(1, {"error_type": e.__class__.__name__})
                        
                        raise
                    finally:
                        # Update metrics based on function name
                        if func.__name__.startswith('llm_'):
                            self.llm_requests.add(1, {"function": func.__name__})
                        elif func.__name__.startswith('rag_'):
                            self.rag_retrieval_count.add(1, {"function": func.__name__})
            
            return cast(F, wrapper)
        return decorator
    
    def start_span(self, name: str, **attrs) -> Span:
        """
        Start a new span.
        
        Args:
            name: Name of the span
            **attrs: Additional attributes for the span
            
        Returns:
            The created span
        """
        return self.tracer.start_span(name, attributes=attrs)
    
    def record_metric(self, name: str, value: float, unit: str = "1", **attrs) -> None:
        """
        Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value to record
            unit: Unit of measurement
            **attrs: Additional attributes for the metric
        """
        counter = self.meter.create_counter(
            name=name,
            unit=unit,
            description=f"Custom metric: {name}",
        )
        counter.add(value, attrs)
    
    def record_histogram(self, name: str, value: float, unit: str = "1", **attrs) -> None:
        """
        Record a histogram value.
        
        Args:
            name: Name of the histogram
            value: Value to record
            unit: Unit of measurement
            **attrs: Additional attributes for the histogram
        """
        histogram = self.meter.create_histogram(
            name=name,
            unit=unit,
            description=f"Custom histogram: {name}",
        )
        histogram.record(value, attrs)
    
    def record_llm_call(
        self, 
        provider: str, 
        model: str, 
        duration: float, 
        success: bool = True,
        error_type: Optional[str] = None,
        **attrs
    ) -> None:
        """
        Record metrics for an LLM API call.
        
        Args:
            provider: LLM provider (e.g., 'openai', 'ollama', 'gemini')
            model: Model name
            duration: Duration of the call in seconds
            success: Whether the call was successful
            error_type: Type of error if the call failed
            **attrs: Additional attributes
        """
        attributes = {
            "llm.provider": provider,
            "llm.model": model,
            **attrs
        }
        
        # Record request count
        self.llm_requests.add(1, attributes)
        
        # Record duration
        self.llm_duration.record(duration, attributes)
        
        # Record error if applicable
        if not success:
            error_attrs = {**attributes, "error_type": error_type or "unknown"}
            self.llm_errors.add(1, error_attrs)
    
    def record_rag_retrieval(
        self, 
        query_type: str, 
        document_count: int, 
        duration: float, 
        **attrs
    ) -> None:
        """
        Record metrics for a RAG retrieval operation.
        
        Args:
            query_type: Type of RAG query (e.g., 'semantic', 'hybrid')
            document_count: Number of documents retrieved
            duration: Duration of the retrieval in seconds
            **attrs: Additional attributes
        """
        attributes = {
            "rag.query_type": query_type,
            **attrs
        }
        
        # Record retrieval count and duration
        self.rag_retrieval_count.add(1, attributes)
        self.rag_retrieval_duration.record(duration, attributes)
        
        # Record document count as a separate metric
        self.record_metric("rag.documents.retrieved", document_count, unit="1", **attributes)
    
    def record_job_processing(
        self, 
        job_type: str, 
        duration: float, 
        success: bool = True,
        **attrs
    ) -> None:
        """
        Record metrics for job processing.
        
        Args:
            job_type: Type of job (e.g., 'scraping', 'analysis', 'matching')
            duration: Duration of processing in seconds
            success: Whether processing was successful
            **attrs: Additional attributes
        """
        attributes = {
            "job.type": job_type,
            "job.status": "success" if success else "error",
            **attrs
        }
        
        # Record job count and duration
        self.jobs_processed.add(1, attributes)
        self.job_processing_duration.record(duration, attributes)
    
    def record_vectordb_operation(
        self, 
        operation: str, 
        collection: str, 
        duration: float, 
        success: bool = True,
        **attrs
    ) -> None:
        """
        Record metrics for vector database operations.
        
        Args:
            operation: Type of operation (e.g., 'insert', 'query', 'delete')
            collection: Name of the collection
            duration: Duration of the operation in seconds
            success: Whether the operation was successful
            **attrs: Additional attributes
        """
        attributes = {
            "vectordb.operation": operation,
            "vectordb.collection": collection,
            "vectordb.status": "success" if success else "error",
            **attrs
        }
        
        # Record operation count and duration
        self.vectordb_operations.add(1, attributes)
        self.vectordb_duration.record(duration, attributes)
    
    def shutdown(self) -> None:
        """Shutdown the monitoring system and flush any pending exports."""
        try:
            # Shutdown tracer provider
            if hasattr(self, 'tracer_provider'):
                self.tracer_provider.shutdown()
            
            # Shutdown meter provider
            if hasattr(self, 'meter_provider'):
                self.meter_provider.shutdown()
                
            logger.info("Monitoring system shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down monitoring: {e}", exc_info=True)


class NoOpMeter:
    """A no-op implementation of the Meter interface."""
    
    def create_counter(
        self, 
        name: str, 
        unit: str = "1", 
        description: Optional[str] = None
    ) -> 'NoOpCounter':
        return NoOpCounter()
    
    def create_histogram(
        self, 
        name: str, 
        unit: str = "1", 
        description: Optional[str] = None
    ) -> 'NoOpHistogram':
        return NoOpHistogram()
    
    def create_observable_counter(
        self, 
        name: str, 
        callbacks: Optional[list] = None, 
        unit: str = "1", 
        description: Optional[str] = None
    ) -> 'NoOpObservableCounter':
        return NoOpObservableCounter()
    
    def create_observable_gauge(
        self, 
        name: str, 
        callbacks: Optional[list] = None, 
        unit: str = "1", 
        description: Optional[str] = None
    ) -> 'NoOpObservableGauge':
        return NoOpObservableGauge()
    
    def create_observable_up_down_counter(
        self, 
        name: str, 
        callbacks: Optional[list] = None, 
        unit: str = "1", 
        description: Optional[str] = None
    ) -> 'NoOpObservableUpDownCounter':
        return NoOpObservableUpDownCounter()


class NoOpCounter:
    """A no-op implementation of the Counter interface."""
    def add(self, amount: float, attributes: Optional[dict] = None) -> None:
        pass


class NoOpHistogram:
    """A no-op implementation of the Histogram interface."""
    def record(self, amount: float, attributes: Optional[dict] = None) -> None:
        pass


class NoOpObservableCounter:
    """A no-op implementation of the ObservableCounter interface."""
    pass


class NoOpObservableGauge:
    """A no-op implementation of the ObservableGauge interface."""
    pass


class NoOpObservableUpDownCounter:
    """A no-op implementation of the ObservableUpDownCounter interface."""
    pass


# Global monitor instance
monitor = Monitor()

# Re-export commonly used functions
trace = monitor.trace
start_span = monitor.start_span
record_metric = monitor.record_metric
record_histogram = monitor.record_histogram
record_llm_call = monitor.record_llm_call
record_rag_retrieval = monitor.record_rag_retrieval
record_job_processing = monitor.record_job_processing
record_vectordb_operation = monitor.record_vectordb_operation
