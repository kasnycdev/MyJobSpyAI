import logging
import time
from typing import Any, Callable, Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from rich.console import Console
from rich.logging import RichHandler

from myjobspyai.config import config

# Global tracer and meter for use in other modules
tracer: Optional[trace.Tracer] = None
meter: Optional[metrics.Meter] = None
_otel_enabled_runtime: bool = True


def _initialize_exporter_with_retry(
    exporter_type_name: str,
    exporter_factory_func: Callable[[], Any],
    max_retries: int,
    retry_delay: float,
) -> Optional[Any]:
    """Attempts to initialize an OTLP exporter with retries."""
    for attempt in range(max_retries + 1):
        try:
            exporter = exporter_factory_func()
            logger = logging.getLogger(__name__)
            logger.info(
                f"Successfully initialized OTLP {exporter_type_name} exporter on attempt {attempt + 1}."
            )
            return exporter
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to initialize OTLP {exporter_type_name} exporter on attempt {attempt + 1}/{max_retries + 1}: {e}"
            )
            if attempt < max_retries:
                logger.info(
                    f"Retrying OTLP {exporter_type_name} exporter initialization in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts to initialize OTLP {exporter_type_name} exporter failed. "
                    "This component of OpenTelemetry will be disabled."
                )
                return None


def _create_span_exporter(
    otlp_protocol: str,
    otlp_endpoint: str,
    otlp_headers: Optional[str] = None,
) -> Any:
    """Creates a span exporter based on the OTLP protocol."""
    if otlp_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)
    elif otlp_protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPHttpSpanExporter,
        )

        return OTLPHttpSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)
    else:
        logger = logging.getLogger(__name__)
        logger.error(
            f"Invalid OTLP protocol '{otlp_protocol}' for traces. Using gRPC default."
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)


def _create_log_exporter(
    otlp_protocol: str,
    otlp_endpoint: str,
    otlp_headers: Optional[str] = None,
) -> Any:
    """Creates a log exporter based on the OTLP protocol."""
    if otlp_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter

        return OTLPLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
    elif otlp_protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.log_exporter import (
            OTLPLogExporter as OTLPHttpLogExporter,
        )

        return OTLPHttpLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
    else:
        logger = logging.getLogger(__name__)
        logger.error(
            f"Invalid OTLP protocol '{otlp_protocol}' for logs. Using gRPC default."
        )
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter

        return OTLPLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)


def _create_metric_exporter(
    otlp_endpoint: str,
    otlp_headers: Optional[str] = None,
) -> Any:
    """Creates a metric exporter."""
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )

    return OTLPMetricExporter(endpoint=otlp_endpoint, headers=otlp_headers)


def configure_opentelemetry():
    """Configures OpenTelemetry for logs, traces, and metrics."""
    global tracer, meter, _otel_enabled_runtime
    otel_cfg = getattr(config, 'opentelemetry', {})

    if not getattr(otel_cfg, 'OTEL_ENABLED', True):
        logger = logging.getLogger(__name__)
        logger.info(
            "OpenTelemetry is DISABLED by configuration. "
            "Initializing NoOp providers."
        )
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        _otel_enabled_runtime = False
        return

    logger = logging.getLogger(__name__)
    logger.info(
        "OpenTelemetry is ENABLED by configuration. "
        "Proceeding with exporter setup..."
    )

    service_name = otel_cfg.get("OTEL_SERVICE_NAME")
    otlp_endpoint = otel_cfg.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_protocol = otel_cfg.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    otlp_headers = otel_cfg.get("OTEL_EXPORTER_OTLP_HEADERS")
    max_retries = otel_cfg.get("OTEL_EXPORTER_MAX_RETRIES", 3)
    retry_delay = otel_cfg.get("OTEL_EXPORTER_RETRY_DELAY_SECONDS", 5)

    sampler = otel_cfg.get("OTEL_TRACES_SAMPLER_INSTANCE")
    if sampler is None:
        logger.warning(
            "OTEL_TRACES_SAMPLER_INSTANCE not found in otel_cfg, "
            "defaulting to ALWAYS_ON."
        )
        sampler = ALWAYS_ON

    resource_attributes = otel_cfg.get("OTEL_RESOURCE_ATTRIBUTES", {})
    resource = Resource.create({"service.name": service_name, **resource_attributes})

    otel_components_initialized = 0

    span_exporter = _initialize_exporter_with_retry(
        "Trace",
        lambda: _create_span_exporter(otlp_protocol, otlp_endpoint, otlp_headers),
        max_retries,
        retry_delay,
    )
    if span_exporter:
        tracer_provider = TracerProvider(resource=resource, sampler=sampler)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        tracer = trace.get_tracer(__name__)
        logger.info(
            f"OpenTelemetry Tracer configured for service '{service_name}' "
            f"sending to '{otlp_endpoint}' via {otlp_protocol}."
        )
        otel_components_initialized += 1
    else:
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.error(
            "Failed to initialize Trace Exporter. " "Tracing will use NoOpTracer."
        )

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.sdk.logs import LoggerProvider, set_logger_provider
        from opentelemetry.sdk.logs.export import BatchLogRecordProcessor

        log_exporter = _initialize_exporter_with_retry(
            "Log",
            lambda: _create_log_exporter(otlp_protocol, otlp_endpoint, otlp_headers),
            max_retries,
            retry_delay,
        )
        if log_exporter:
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(log_exporter)
            )
            set_logger_provider(logger_provider)
            LoggingInstrumentor().instrument(
                logger_provider=logger_provider, set_logging_format=True
            )
            logger.info(
                f"OpenTelemetry Logger configured for service '{service_name}' "
                f"sending to '{otlp_endpoint}' via {otlp_protocol}."
            )
            otel_components_initialized += 1
        else:
            logger.error(
                "Failed to initialize Log Exporter. " "OTel logging will be disabled."
            )

    except ModuleNotFoundError:
        logger.warning(
            "OpenTelemetry logging instrumentation not available. "
            "Standard logging will be used."
        )

    try:
        metric_exporter = _initialize_exporter_with_retry(
            "Metric",
            lambda: _create_metric_exporter(otlp_endpoint, otlp_headers),
            max_retries,
            retry_delay,
        )
        if metric_exporter:
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[PeriodicExportingMetricReader(metric_exporter)],
            )
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter(__name__)
            logger.info(
                f"OpenTelemetry Meter configured for service '{service_name}' "
                f"sending to '{otlp_endpoint}'."
            )
            otel_components_initialized += 1
        else:
            meter = metrics.get_meter(
                __name__, meter_provider=metrics.NoOpMeterProvider()
            )
            logger.error(
                "Failed to initialize Metric Exporter. " "Metrics will use NoOpMeter."
            )

    except ModuleNotFoundError:
        logger.warning(
            "OpenTelemetry metrics instrumentation not available. "
            "Metrics will be disabled."
        )

    logger.info(
        f"OpenTelemetry configuration complete. {otel_components_initialized} components initialized."
    )


def is_otel_enabled_runtime() -> bool:
    """Returns whether OpenTelemetry is enabled at runtime."""
    return _otel_enabled_runtime


# Specific logger for model outputs
MODEL_OUTPUT_LOGGER_NAME = "model_output"


def setup_logging():
    """Configures logging with Rich console output and separate file outputs for INFO, DEBUG, and ERROR levels."""
    # Create log directory if it doesn't exist
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Configure Rich console output
    console = Console()
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        keywords=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(rich_handler)

    # Configure file handlers with different levels
    info_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    info_handler.setLevel(logging.INFO)

    debug_handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
    debug_handler.setLevel(logging.DEBUG)

    error_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
    error_handler.setLevel(logging.WARNING)

    # Configure model output logger
    model_output_handler = logging.FileHandler(
        os.path.join(log_dir, 'model_output.log')
    )
    model_output_handler.setLevel(logging.INFO)

    # Add handlers to root logger
    root_logger.addHandler(info_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(error_handler)

    # Configure model output logger
    model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)
    model_output_logger.handlers.clear()
    model_output_logger.addHandler(model_output_handler)
    model_output_logger.setLevel(logging.INFO)

    # Set consistent logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    model_output_handler.setFormatter(formatter)

    # Configure logging levels from config
    logging_config = config.get('logging', {})
    log_level = logging_config.get('log_level', 'INFO').upper()
    root_logger.setLevel(log_level)

    # Configure log file mode
    log_file_mode = logging_config.get('log_file_mode', 'a')
    if log_file_mode not in ['a', 'w']:
        log_file_mode = 'a'
    info_handler.mode = debug_handler.mode = error_handler.mode = log_file_mode

    # Configure log rotation if enabled
    if logging_config.get('rolling_strategy') == 'size':
        max_size = logging_config.get('max_size', 10485760)  # Default 10MB
        backup_count = logging_config.get('backup_count', 5)
        # Add size-based rotation logic here if needed
    elif logging_config.get('rolling_strategy') == 'time':
        when = logging_config.get('when', 'midnight')
        interval = logging_config.get('interval', 1)
        backup_count = logging_config.get('backup_count', 30)
        # Add time-based rotation logic here if needed

    # Log configuration
    root_logger.info(
        f"Logging configured with level {log_level} and mode {log_file_mode}"
    )

    # Initialize model output logger
    model_output_logger.info("Model output logging initialized")


# Call configuration at import time
configure_opentelemetry()
setup_logging()
