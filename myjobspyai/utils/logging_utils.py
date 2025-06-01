"""
Logging and OpenTelemetry configuration utilities for MyJobSpyAI.

This module provides comprehensive logging and OpenTelemetry integration,
including structured logging, trace correlation, and metrics collection.
"""
from __future__ import annotations

# Standard library imports
import glob
import logging
import logging.config
import os
import sys  # Used for sys.path in debug logging
import time  # Used for time-related operations
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from typing import Any, Literal, Optional, TypeVar, Optional as Opt

# Third-party imports
import structlog
from rich.console import Console as RichConsole
from rich.logging import RichHandler
from structlog.types import EventDict, Processor

# Local application imports
from myjobspyai.utils.config_utils import config as settings
from myjobspyai.utils.otel_config import OTELSDKConfig, get_otel_config


class OverwritingRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that can overwrite existing log files.
    
    This handler will delete any existing log files matching the pattern
    when it's first instantiated, then behave like a normal RotatingFileHandler.
    """
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: Opt[str] = None,
        delay: bool = False,
        errors: Opt[str] = None,
    ) -> None:
        # Delete any existing log files before initializing the handler
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
                
        # Also clean up any rotated log files
        if backupCount > 0:
            for f in glob.glob(f"{filename}.*"):
                try:
                    os.remove(f)
                except OSError:
                    pass
        
        # Now initialize the parent class
        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            errors=errors
        )
        
    def doRollover(self) -> None:
        """Override to handle rollover with our custom behavior."""
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore
            
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}"
                dfn = f"{self.baseFilename}.{i + 1}"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            dfn = f"{self.baseFilename}.1"
            if os.path.exists(dfn):
                os.remove(dfn)
            
            if os.path.exists(self.baseFilename):
                os.rename(self.baseFilename, dfn)
        
        if not self.delay:
            self.stream = self._open()

# Type variable for generic exporter factory function
T = TypeVar('T')
ExporterFactory = TypeVar('ExporterFactory', bound=Callable[..., Any])
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# Global state for OpenTelemetry
_otel_initialized = False
_otel_config: Optional[OTELSDKConfig] = None

# Structured logging processors
PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt='iso', utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]

# Add OpenTelemetry trace correlation if available
try:
    from opentelemetry import trace
    from opentelemetry.trace.span import NonRecordingSpan
    
    def add_trace_context(_, __, event_dict: EventDict) -> EventDict:
        """Add OpenTelemetry trace context to log records."""
        span = trace.get_current_span()
        if not isinstance(span, NonRecordingSpan):
            ctx = span.get_span_context()
            if ctx and hasattr(span, 'is_recording') and span.is_recording():
                event_dict['trace_id'] = f"{ctx.trace_id:032x}" if hasattr(ctx, 'trace_id') else None
                event_dict['span_id'] = f"{ctx.span_id:016x}" if hasattr(ctx, 'span_id') else None
                event_dict['trace_flags'] = f"{ctx.trace_flags:02x}" if hasattr(ctx, 'trace_flags') else None
                if hasattr(ctx, 'trace_state') and ctx.trace_state:
                    event_dict['trace_state'] = str(ctx.trace_state)
        return event_dict
    
    # Insert after the contextvars processor
    if add_trace_context not in PROCESSORS:
        PROCESSORS.insert(1, add_trace_context)
    
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not available. Tracing will be limited.", exc_info=True)

# Configure structlog to work with standard library's logging module
structlog.configure(
    processors=PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Module-level logger
logger = structlog.get_logger(__name__)

# Lazy imports for OpenTelemetry components that might not be available
# or needed in all environments
try:
    # Only import what's actually used in this module
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    from opentelemetry.trace.status import StatusCode  # Status is used in the except block
    # Import OTELConfig only when needed in _get_otel_config()
    
    # Import OTLP exporters only when needed
    def _import_otlp_exporters():
        """Lazily import OTLP exporters to avoid import errors if not needed."""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        
        return (
            OTLPSpanExporter, OTLPMetricExporter, OTLPLogExporter,
            set_logger_provider, LoggerProvider, LoggingHandler, BatchLogRecordProcessor
        )
    
    # Initialize OTLP exporters as None, will be imported when needed
    OTLPSpanExporter = OTLPMetricExporter = OTLPLogExporter = None
    set_logger_provider = LoggerProvider = LoggingHandler = BatchLogRecordProcessor = None
    
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
    logger.warning("OpenTelemetry OTLP exporters not available. Tracing will be limited.")
    
    # Define stubs for type checking
    class OTLPSpanExporter:
        """Stub for OTLPSpanExporter when OpenTelemetry is not available."""
        pass
        
    class OTLPMetricExporter:
        """Stub for OTLPMetricExporter when OpenTelemetry is not available."""
        pass
        
    class OTLPLogExporter:
        """Stub for OTLPLogExporter when OpenTelemetry is not available."""
        pass
        
# Define a simple stub class for OTELConfig
    class _OTELConfigStub:
        """Stub for OTELConfig when OpenTelemetry is not available."""
        def __init__(self, *args, **kwargs):
            self.service_name = "myjobspy-ai"
            self.log_level = "INFO"
            self.otel_enabled = False
            
        def model_dump(self) -> dict:
            """Return the configuration as a dictionary."""
            return {
                "service_name": self.service_name,
                "log_level": self.log_level,
                "otel_enabled": self.otel_enabled
            }
    
    # Define missing status codes
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
        
    class Status:
        def __init__(self, status_code=StatusCode.OK, description=None):
            self.status_code = status_code
            self.description = description
    
    # Stub functions
    def set_logger_provider(*args, **kwargs): pass
    
    class LoggerProvider:
        def get_logger(self, *args, **kwargs):
            return logging.getLogger(args[0] if args else __name__)
    
    class LoggingHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__()
    
    class BatchLogRecordProcessor:
        def __init__(self, *args, **kwargs):
            pass

# Create a specific console instance for RichHandler to avoid conflicts if logging.console is used elsewhere
rich_console_for_handler = RichConsole(stderr=True)

# Module-level flag for OpenTelemetry status
otel_enabled = False

# Global tracer and meter for use in other modules
tracer = None
meter = None
_otel_enabled_runtime = True # Internal flag to track runtime OTEL status

def _initialize_exporter_with_retry(
    exporter_type_name: str, 
    exporter_factory_func: Callable[[], Any], 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> Optional[Any]:
    """
    Tries to initialize an OTLP exporter with retries.
    
    Args:
        exporter_type_name: Name of the exporter type for logging purposes.
        exporter_factory_func: Function that creates and returns the exporter.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
        
    Returns:
        The initialized exporter or None if all attempts fail.
    """
    if not OTLP_AVAILABLE:
        logger.warning(
            "OpenTelemetry OTLP exporters not available. "
            "Cannot initialize %s exporter.", 
            exporter_type_name
        )
        return None
        
    for attempt in range(max_retries + 1):
        try:
            exporter = exporter_factory_func()
            logger.debug(
                "Successfully initialized OTLP %s exporter", 
                exporter_type_name
            )
            return exporter
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d to initialize OTLP %s exporter failed: %s. "
                    "Retrying in %d seconds...",
                    attempt + 1, 
                    max_retries + 1, 
                    exporter_type_name, 
                    str(e), 
                    retry_delay,
                    exc_info=logger.isEnabledFor(logging.DEBUG)
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    "All %d attempts to initialize OTLP %s exporter failed. "
                    "This component of OpenTelemetry will be disabled.",
                    max_retries + 1, 
                    exporter_type_name,
                    exc_info=logger.isEnabledFor(logging.DEBUG)
                )
                return None

def _get_otel_config():
    """
    Internal function to get the OTEL config with proper imports.
    
    Returns:
        Union[dict, OTELConfig]: The OpenTelemetry configuration as a dictionary or model.
        
    Raises:
        ImportError: If required modules cannot be imported.
        RuntimeError: If the configuration cannot be loaded.
    """
    try:
        from myjobspyai.utils.otel_config import get_otel_config as _get_otel_config_func
        from myjobspyai.utils.config_utils import config as app_settings
        
        if not app_settings:
            logger.warning("App settings not available, using default OpenTelemetry configuration")
            return {}
            
        return _get_otel_config_func(app_settings)
        
    except ImportError as e:
        logger.error("Failed to import required modules for OpenTelemetry configuration: %s", e)
    """
    Get OpenTelemetry configuration with proper precedence:
    1. Command-line arguments
    2. Environment variables
    3. config.yaml
    
    Returns:
        dict: A dictionary containing the OpenTelemetry configuration.
        Returns an empty dict if configuration cannot be loaded.
        
    Example:
        >>> config = get_otel_config()
        >>> print(config.get('service_name', 'default_service'))
    """
    try:
        # Try to get OTEL config from the config module
        if hasattr(settings, 'otel') and settings.otel:
            return settings.otel.dict()
            
        # Fall back to environment variables
        return {
            'service_name': os.getenv('OTEL_SERVICE_NAME', 'myjobspy-ai'),
            'log_level': os.getenv('OTEL_LOG_LEVEL', 'INFO'),
            'otel_enabled': os.getenv('OTEL_ENABLED', 'false').lower() == 'true'
        }
    except Exception as e:
        logger.warning(f"Failed to load OpenTelemetry config: {e}")
        logger.error("Error getting OpenTelemetry configuration: %s", e, exc_info=True)
        return {}

def get_otel_config_model() -> '_OTELConfigStub':
    """
    Get the OTEL config as a Pydantic model or a stub if not available.
    
    Returns:
        _OTELConfigStub: The OpenTelemetry configuration model or stub.
        
    Raises:
        RuntimeError: If the OTELConfig class cannot be imported or instantiated.
        
    Example:
        >>> config_model = get_otel_config_model()
        >>> print(config_model.service_name)
    """
    if not OTLP_AVAILABLE:
        logger.warning("OpenTelemetry is not available, using stub configuration")
        return _OTELConfigStub()
        
    try:
        # Import here to avoid circular imports
        from myjobspyai.utils.otel_config import OTELConfig as RealOTELConfig
        return RealOTELConfig()  # type: ignore
    except ImportError as e:
        logger.error("Failed to import OTELConfig: %s", e)
        logger.warning("Falling back to stub configuration")
        return _OTELConfigStub()
    except Exception as e:
        logger.error("Failed to create OTELConfig instance: %s", e, exc_info=True)
        raise RuntimeError("Failed to create OpenTelemetry configuration model") from e

def configure_opentelemetry():
    """Configures OpenTelemetry for logs, traces, and metrics."""
    global tracer, meter, _otel_enabled_runtime
    
    # Get configuration with proper precedence
    config = get_otel_config()
    
    # Check if OpenTelemetry is disabled
    if not config.get("enabled", True):
        logger = logging.getLogger(__name__)
        logger.info("OpenTelemetry is disabled via configuration")
        _otel_enabled_runtime = False
        return
    
    # Set module-level otel_enabled flag
    global otel_enabled
    otel_enabled = True
    
    logger = logging.getLogger(__name__)
    logger.info("OpenTelemetry is ENABLED by configuration. Proceeding with exporter setup...")

    # Get OpenTelemetry configuration with defaults
    service_name = config.get("service_name", "myjobspyai")
    otlp_endpoint = config.get("otlp_endpoint", None)
    otlp_protocol = config.get("otlp_protocol", "http/protobuf")
    otlp_headers = config.get("otlp_headers", {})
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 1.0)

    # Set up sampler (default to ALWAYS_ON if not specified)
    sampler = config.get("traces_sampler_instance", None)
    if sampler is None:
        logger.warning("No sampler instance configured, defaulting to ALWAYS_ON")
        sampler = ALWAYS_ON
        
    # Get resource attributes
    resource_attributes = {
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": "1.0.0",
    }
    
    # Add any additional resource attributes from config
    if "resource_attributes" in config:
        resource_attributes.update(config["resource_attributes"])
    
    resource = Resource(attributes=resource_attributes)

    otel_components_initialized = 0

    # --- Trace Provider Setup ---
    def create_span_exporter():
        if not otlp_endpoint:
            logger.error("OTLP endpoint not configured. Cannot initialize trace exporter.")
            return None
            
        try:
            if otlp_protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
                return GrpcExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            if otlp_protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter
                return HttpExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            
            logger.error("Invalid OTLP protocol '%s' for traces. Using gRPC default.", otlp_protocol)
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as FallbackExporter
            return FallbackExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            
        except (ImportError, ValueError) as e:
            logger.error("Failed to initialize trace exporter: %s", e)
            return None

    span_exporter = _initialize_exporter_with_retry("Trace", create_span_exporter, max_retries, retry_delay)
    if span_exporter:
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)
        logger.info(f"OpenTelemetry Tracer configured for service '{service_name}' sending to '{otlp_endpoint}' via {otlp_protocol}.")
        otel_components_initialized += 1
    else:
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.error("Failed to initialize Trace Exporter. Tracing will use NoOpTracer.")

    # --- Logger Provider Setup ---
    try:
        # Lazy import to avoid dependency if not using logging
        from opentelemetry._logs import set_logger_provider  # pylint: disable=import-outside-toplevel
        from opentelemetry.sdk._logs import LoggerProvider  # pylint: disable=import-outside-toplevel
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # pylint: disable=import-outside-toplevel
        from opentelemetry.instrumentation.logging import LoggingInstrumentor  # type: ignore

        def create_log_exporter():
            if not otlp_endpoint:
                logger.error("OTLP endpoint not configured. Cannot initialize log exporter.")
                return None
                
            try:
                if otlp_protocol == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
                    return OTLPLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
                if otlp_protocol == "http/protobuf":
                    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter
                    return HttpLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            
                logger.error(
                    "Invalid OTLP protocol '%s' for logs. Using gRPC default.",
                    otlp_protocol
                )
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as FallbackExporter
                return FallbackExporter(endpoint=otlp_endpoint, headers=otlp_headers)
                
            except (ImportError, ValueError) as e:
                logger.error("Failed to initialize log exporter: %s", str(e))
                return None

        log_exporter = _initialize_exporter_with_retry("Log", create_log_exporter, max_retries, retry_delay)
        if log_exporter:
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(log_exporter)
            )
            set_logger_provider(logger_provider)
            LoggingInstrumentor().instrument(logger_provider=logger_provider, set_logging_format=True)
            logger.info(
                "OpenTelemetry Logger configured for service '%s' sending to '%s' via %s.",
                service_name, otlp_endpoint, otlp_protocol
            )
            otel_components_initialized += 1
        else:
            logger.error("Failed to initialize Log Exporter. OTel logging will be disabled.")
            # Standard Python logging will still work.
            
    except ModuleNotFoundError:
        logger.warning("OpenTelemetry Logs SDK (opentelemetry.sdk.logs) or its OTLP exporter not found. OTel logging disabled.")
    
    # --- Metric Exporter Setup ---
    def create_metric_exporter():
        if not otlp_endpoint:
            logger.error("OTLP endpoint not configured. Cannot initialize metric exporter.")
            return None
            
        try:
            if otlp_protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as GrpcExporter
                return GrpcExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            if otlp_protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as HttpExporter
                return HttpExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            
            logger.error(
                "Invalid OTLP protocol '%s' for metrics. Using gRPC default.",
                otlp_protocol
            )
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter as FallbackExporter
            return FallbackExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            
        except (ImportError, ValueError) as e:
            logger.error("Failed to initialize metric exporter: %s", str(e))
            return None

    metric_exporter = _initialize_exporter_with_retry("Metric", create_metric_exporter, max_retries, retry_delay)
    if metric_exporter:
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter(__name__)
        logger.info("OpenTelemetry Meter configured for service '%s' sending to '%s' via %s.", 
                    service_name, otlp_endpoint, otlp_protocol)
        otel_components_initialized += 1
    else:
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        logger.error("Failed to initialize Metric Exporter. Meter will use NoOpMeterProvider.")

    # Set the global _otel_enabled_runtime flag based on whether any components were initialized
    _otel_enabled_runtime = otel_components_initialized > 0
    logger.info("OpenTelemetry initialization complete. %d/3 components initialized.", otel_components_initialized)
    if not _otel_enabled_runtime and otel_enabled:
        logger.warning("OpenTelemetry is enabled in config but no exporters could be initialized. Check your configuration.")
        # Fallback globals to NoOp if not already set
        if tracer is None:
            tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        if meter is None:
            meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())


# Call configuration at import time
configure_opentelemetry()

# Check runtime status for other modules that might import tracer/meter
def is_otel_enabled_runtime():
    return _otel_enabled_runtime

MODEL_OUTPUT_LOGGER_NAME = "model_output"

def setup_logging() -> None:
    """
    Configures logging with Rich console output and separate file outputs
    for INFO, DEBUG, and ERROR levels.
    """
    try:
        # Set the root logger level to DEBUG to capture all logs
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Ensure logs directory exists in the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        print(f"Logs directory set to: {logs_dir}", file=sys.stderr)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        simple_formatter = logging.Formatter("%(message)s")
        
        # Console handler with Rich
        console_handler = RichHandler(
            console=rich_console_for_handler,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=False,
            show_path=True,
            show_level=True,
            level=logging.INFO
        )
        console_handler.setFormatter(simple_formatter)
        
        # Debug file handler - logs everything
        debug_log_path = os.path.join(logs_dir, "debug.log")
        debug_file_handler = OverwritingRotatingFileHandler(
            filename=debug_log_path,
            maxBytes=getattr(settings.logging, 'max_size', 10 * 1024 * 1024) if hasattr(settings, 'logging') else 10 * 1024 * 1024,
            backupCount=getattr(settings.logging, 'backup_count', 5) if hasattr(settings, 'logging') else 5,
            encoding="utf-8"
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(detailed_formatter)
        
        # Error file handler - only logs errors and above
        error_log_path = os.path.join(logs_dir, "error.log")
        error_file_handler = OverwritingRotatingFileHandler(
            filename=error_log_path,
            maxBytes=getattr(settings.logging, 'max_size', 10 * 1024 * 1024) if hasattr(settings, 'logging') else 10 * 1024 * 1024,
            backupCount=getattr(settings.logging, 'backup_count', 5) if hasattr(settings, 'logging') else 5,
            encoding="utf-8"
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        
        # Add all handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(debug_file_handler)
        root_logger.addHandler(error_file_handler)
        
        # Get logger after setting up handlers
        logger = logging.getLogger(__name__)
        
        # Log Python environment for debugging
        logger.debug("Python executable: %s", sys.executable)
        logger.debug("Python version: %s", sys.version)
        logger.debug("Python path: %s", os.environ.get('PYTHONPATH', 'Not set'))
        logger.debug("sys.path:\n%s", '\n'.join(f"  {p}" for p in sys.path))
        logger.debug("Debug logging enabled to %s", debug_log_path)
        logger.error("Error logging enabled to %s", error_log_path)
        
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        raise

    # Log the effective level of each handler
    for handler in root_logger.handlers:
        handler_name = handler.__class__.__name__
        handler_level = logging.getLevelName(handler.level) if hasattr(handler, 'level') else 'NOTSET'
        logger.info("Handler '%s' has level: %s", handler_name, handler_level)
    
    # Log the final configuration
    logger.info("Logging configuration complete. Root logger level: %s", logging.getLevelName(root_logger.getEffectiveLevel()))
    logger.debug("Debug logging is enabled")
    logger.info("Info logging is enabled")
    logger.warning("Warning logging is enabled")
    logger.error("Error logging is enabled")
    logger.critical("Critical logging is enabled")
    
    # --- Model Output File Handler (Separate Logger) ---
    model_output_log_path = os.path.join(logs_dir, "model_output.log")
    model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)
    model_output_logger.setLevel(logging.DEBUG)  # Capture all levels for this specific logger
    model_output_logger.propagate = False  # Prevent model output from going to root logger's handlers
    
    # Create a separate file handler for model output
    model_output_handler = OverwritingRotatingFileHandler(
        filename=model_output_log_path,
        maxBytes=getattr(settings.logging, 'max_size', 10 * 1024 * 1024),  # Default to 10 MB
        backupCount=getattr(settings.logging, 'backup_count', 5),  # Default to 5 backups
        encoding='utf-8'
    )
    model_output_handler.setLevel(logging.INFO)
    model_output_formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    model_output_handler.setFormatter(model_output_formatter)
    model_output_logger.addHandler(model_output_handler)
    
    # Log model output configuration
    model_output_logger.info("Model output logging initialized")
    logger.info("Model output will be written to: %s", model_output_log_path)

def log_model_output(model_name: str, prompt: str, response: str, metadata: dict = None) -> None:
    """
    Log model input/output to a dedicated log file.
    
    Args:
        model_name: Name or identifier of the model
        prompt: The input prompt sent to the model
        response: The model's response
        metadata: Optional additional metadata to include in the log
    """
    try:
        model_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)
        
        # Prepare log entry
        log_entry = {
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        # Log as JSON for easy parsing
        import json
        model_logger.info(json.dumps(log_entry, ensure_ascii=False))
        
    except Exception as e:
        # Fallback to regular error logging if model logging fails
        logger.error(f"Failed to log model output: {str(e)}", exc_info=True)

# Example usage:
# log_model_output(
#     model_name="gpt-4",
#     prompt="What is the capital of France?",
#     response="The capital of France is Paris.",
#     metadata={"temperature": 0.7, "max_tokens": 50}
# )
