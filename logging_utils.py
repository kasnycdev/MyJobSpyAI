import logging
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console as RichConsole
from config import settings # Import the globally loaded settings

# OpenTelemetry Imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON # Moved here
# Importers will be selected based on protocol
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
# from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
# from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as OTLPHttpMetricExporter

# Create a specific console instance for RichHandler to avoid conflicts if logging.console is used elsewhere
rich_console_for_handler = RichConsole(stderr=True)

import time # Add time import for retry delay

# Global tracer and meter for use in other modules
tracer = None
meter = None
_otel_enabled_runtime = True # Internal flag to track runtime OTEL status

def _initialize_exporter_with_retry(exporter_type_name, exporter_factory_func, max_retries, retry_delay):
    """Tries to initialize an OTLP exporter with retries."""
    for attempt in range(max_retries + 1):
        try:
            exporter = exporter_factory_func()
            logging.getLogger(__name__).info(f"Successfully initialized OTLP {exporter_type_name} exporter on attempt {attempt + 1}.")
            return exporter
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize OTLP {exporter_type_name} exporter on attempt {attempt + 1}/{max_retries + 1}: {e}"
            )
            if attempt < max_retries:
                logging.getLogger(__name__).info(f"Retrying OTLP {exporter_type_name} exporter initialization in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logging.getLogger(__name__).error(
                    f"All {max_retries + 1} attempts to initialize OTLP {exporter_type_name} exporter failed. "
                    f"This component of OpenTelemetry will be disabled."
                )
                return None

def configure_opentelemetry():
    """Configures OpenTelemetry for logs, traces, and metrics."""
    global tracer, meter, _otel_enabled_runtime
    otel_cfg = settings.get("opentelemetry", {})
    
    if not otel_cfg.get("OTEL_ENABLED", True): # Check processed OTEL_ENABLED from config.py
        logger = logging.getLogger(__name__)
        logger.info("OpenTelemetry is DISABLED by configuration. Initializing NoOp providers.")
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        _otel_enabled_runtime = False
        return

    logger = logging.getLogger(__name__)
    logger.info("OpenTelemetry is ENABLED by configuration. Proceeding with exporter setup...")

    service_name = otel_cfg.get("OTEL_SERVICE_NAME")
    otlp_endpoint = otel_cfg.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_protocol = otel_cfg.get("OTEL_EXPORTER_OTLP_PROTOCOL")
    otlp_headers = otel_cfg.get("OTEL_EXPORTER_OTLP_HEADERS")
    max_retries = otel_cfg.get("OTEL_EXPORTER_MAX_RETRIES", 3) # Default from config.py if not found
    retry_delay = otel_cfg.get("OTEL_EXPORTER_RETRY_DELAY_SECONDS", 5) # Default from config.py

    sampler = otel_cfg.get("OTEL_TRACES_SAMPLER_INSTANCE")
    if sampler is None: # Fallback if instance wasn't set (e.g. error in config.py processing)
        # from opentelemetry.sdk.trace.sampling import ALWAYS_ON # Removed from here
        logging.getLogger(__name__).warning("OTEL_TRACES_SAMPLER_INSTANCE not found in otel_cfg, defaulting to ALWAYS_ON.")
        sampler = ALWAYS_ON
    resource_attributes = otel_cfg.get("OTEL_RESOURCE_ATTRIBUTES", {})
    resource = Resource.create({"service.name": service_name, **resource_attributes})

    otel_components_initialized = 0

    # --- Trace Provider Setup ---
    def create_span_exporter():
        if otlp_protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            return OTLPSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)
        elif otlp_protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
            return OTLPHttpSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)
        else:
            logging.getLogger(__name__).error(f"Invalid OTLP protocol '{otlp_protocol}' for traces. Using gRPC default.")
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            return OTLPSpanExporter(endpoint=otlp_endpoint, headers=otlp_headers)

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
        from opentelemetry.sdk.logs import LoggerProvider, set_logger_provider
        from opentelemetry.sdk.logs.export import BatchLogRecordProcessor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor

        def create_log_exporter():
            if otlp_protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
                return OTLPLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            elif otlp_protocol == "http/protobuf":
                from opentelemetry.exporter.otlp.proto.http.log_exporter import OTLPLogExporter as OTLPHttpLogExporter
                return OTLPHttpLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)
            else:
                logging.getLogger(__name__).error(f"Invalid OTLP protocol '{otlp_protocol}' for logs. Using gRPC default.")
                from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
                return OTLPLogExporter(endpoint=otlp_endpoint, headers=otlp_headers)

        log_exporter = _initialize_exporter_with_retry("Log", create_log_exporter, max_retries, retry_delay)
        if log_exporter:
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
            set_logger_provider(logger_provider)
            LoggingInstrumentor().instrument(logger_provider=logger_provider, set_logging_format=True)
            logger.info(f"OpenTelemetry Logger configured for service '{service_name}' sending to '{otlp_endpoint}' via {otlp_protocol}.")
            otel_components_initialized += 1
        else:
            logger.error("Failed to initialize Log Exporter. OTel logging will be disabled.")
            # Standard Python logging will still work.
            
    except ModuleNotFoundError:
        logger.warning("OpenTelemetry Logs SDK (opentelemetry.sdk.logs) or its OTLP exporter not found. OTel logging disabled.")
    
    # --- Meter Provider Setup ---
    def create_metric_exporter():
        if otlp_protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            return OTLPMetricExporter(endpoint=otlp_endpoint, headers=otlp_headers)
        elif otlp_protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter as OTLPHttpMetricExporter
            return OTLPHttpMetricExporter(endpoint=otlp_endpoint, headers=otlp_headers)
        else:
            logging.getLogger(__name__).error(f"Invalid OTLP protocol '{otlp_protocol}' for metrics. Using gRPC default.")
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            return OTLPMetricExporter(endpoint=otlp_endpoint, headers=otlp_headers)

    metric_exporter = _initialize_exporter_with_retry("Metric", create_metric_exporter, max_retries, retry_delay)
    if metric_exporter:
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        meter = metrics.get_meter(__name__)
        logger.info(f"OpenTelemetry Meter configured for service '{service_name}' sending to '{otlp_endpoint}' via {otlp_protocol}.")
        otel_components_initialized += 1
    else:
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        logger.error("Failed to initialize Metric Exporter. Meter will use NoOpMeterProvider.")

    if otel_components_initialized == 0 and otel_cfg.get("OTEL_ENABLED", True):
        logger.critical("All OpenTelemetry components (Trace, Log, Metric) failed to initialize after retries. OTEL effectively disabled.")
        _otel_enabled_runtime = False # Update runtime status
        # Fallback globals to NoOp if not already set
        if tracer is None: tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        if meter is None: meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())


# Call configuration at import time
configure_opentelemetry()

# Check runtime status for other modules that might import tracer/meter
def is_otel_enabled_runtime():
    return _otel_enabled_runtime

# Specific logger for model outputs
MODEL_OUTPUT_LOGGER_NAME = "model_output"

def setup_logging():
    """
    Configures logging with Rich console output and separate file outputs
    for INFO, DEBUG, and ERROR levels.
    """
    log_cfg = settings.get("logging", {})
    root_logger = logging.getLogger()
    
    # Set overall root logger level - this acts as a general floor.
    # Handlers can have their own more specific levels.
    # If a handler's level is more restrictive (e.g., INFO) than the root (e.g. DEBUG),
    # the handler's level takes precedence for that handler.
    # If a handler is less restrictive (e.g. DEBUG) than the root (e.g. INFO),
    # the root's level will prevent messages below INFO from even reaching that handler.
    # So, set root to the lowest level any handler might use (typically DEBUG).
    root_logger.setLevel(logging.DEBUG) # Set root to DEBUG to allow all handlers to process up to their own level

    # Clear existing handlers to prevent duplicate logging if this function is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- File Formatter ---
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- DEBUG File Handler ---
    debug_log_path = log_cfg.get("debug_log_path", "logs/debug.log")
    debug_level_str = log_cfg.get("file_log_level_debug", "DEBUG").upper()
    debug_handler = RotatingFileHandler(debug_log_path, mode='w', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    debug_handler.setLevel(getattr(logging, debug_level_str, logging.DEBUG))
    debug_handler.setFormatter(file_formatter)
    root_logger.addHandler(debug_handler)
    logging.info(f"Debug logs will be written to: {debug_log_path} at level {debug_level_str}")

    # --- INFO File Handler (captures INFO and above, but not DEBUG unless root is DEBUG) ---
    info_log_path = log_cfg.get("info_log_path", "logs/info.log")
    info_level_str = log_cfg.get("file_log_level_info", "INFO").upper()
    info_handler = RotatingFileHandler(info_log_path, mode='w', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    info_handler.setLevel(getattr(logging, info_level_str, logging.INFO))
    info_handler.setFormatter(file_formatter)
    # Filter to only allow INFO level messages for this specific handler
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)
    root_logger.addHandler(info_handler)
    logging.info(f"Info logs (INFO only) will be written to: {info_log_path} at level {info_level_str}")

    # --- ERROR File Handler (captures ERROR and CRITICAL) ---
    error_log_path = log_cfg.get("error_log_path", "logs/error.log")
    error_level_str = log_cfg.get("file_log_level_error", "ERROR").upper()
    error_handler = RotatingFileHandler(error_log_path, mode='w', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
    error_handler.setLevel(getattr(logging, error_level_str, logging.ERROR))
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    logging.info(f"Error logs will be written to: {error_log_path} at level {error_level_str}")

    # --- Rich Console Handler ---
    if log_cfg.get("log_to_console", True):
        console_level_str = log_cfg.get("console_log_level", "INFO").upper()
        console_handler = RichHandler(
            console=rich_console_for_handler, # Use the specific RichConsole instance
            rich_tracebacks=True,
            markup=True, # Enable Rich markup in log messages
            log_time_format=log_cfg.get("date_format", "[%X]") # Use date_format from config
        )
        console_handler.setLevel(getattr(logging, console_level_str, logging.INFO))
        # For RichHandler, use a simpler formatter as Rich handles most of the styling
        # The default format for RichHandler is usually just "%(message)s"
        # It automatically includes level and time based on its own settings.
        # We can set a basic formatter if we want to ensure only the message part is passed.
        console_formatter = logging.Formatter(log_cfg.get("format", "%(message)s"))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        logging.info(f"Console logging enabled with RichHandler at level {console_level_str}")
    else:
        logging.info("Console logging is disabled via config.")

    # --- Initial log message to confirm setup ---
    # Use root_logger directly for this initial message to ensure it goes through the new setup
    root_logger.info("Logging configured with multiple file handlers and Rich console output.")
    root_logger.debug("This is a debug message to test debug file logging.")
    root_logger.error("This is an error message to test error file logging.")

    # --- Model Output File Handler (Separate Logger) ---
    model_output_log_path = log_cfg.get("model_output_log_path", "logs/model_output.log")
    model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)
    model_output_logger.setLevel(logging.DEBUG) # Capture all levels for this specific logger
    model_output_logger.propagate = False # Prevent model output from going to root logger's handlers (console, other files)

    model_output_file_handler = RotatingFileHandler(
        model_output_log_path, mode='w', maxBytes=10*1024*1024, backupCount=3, encoding='utf-8'
    )
    # Use a very simple formatter for model outputs, just the message and timestamp
    model_output_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    model_output_file_handler.setFormatter(model_output_formatter)
    model_output_logger.addHandler(model_output_file_handler)
    root_logger.info(f"Model outputs will be logged to: {model_output_log_path}")


# --- Example of how to get a logger in other modules ---
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is an info message from my_module.")
# logger.debug("This is a debug message from my_module.")
# logger.error("This is an error message from my_module.", exc_info=True) # For errors with tracebacks
