"""
OpenTelemetry Configuration Module

This module provides comprehensive configuration support for OpenTelemetry,
supporting all standard environment variables and configuration parameters.
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    TypeVar,
    Type,
    Tuple,
    cast,
)
from pydantic import BaseModel, Field, validator, root_validator

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar('T')
ConfigValue = Union[str, int, float, bool, List[str], Dict[str, str], None]

class OTELResourceConfig(BaseModel):
    """OpenTelemetry resource configuration."""
    service_name: str = Field(
        default="myjobspy-ai",
        description="Service name for resource attributes"
    )
    namespace: str = Field(
        default="",
        description="Service namespace for resource attributes"
    )
    version: str = Field(
        default="1.0.0",
        description="Service version for resource attributes"
    )
    instance_id: str = Field(
        default="",
        description="Service instance ID for resource attributes"
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (e.g., development, staging, production)"
    )
    attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional resource attributes"
    )

class OTELExporterConfig(BaseModel):
    """OpenTelemetry exporter configuration."""
    endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP endpoint URL"
    )
    protocol: str = Field(
        default="grpc",
        description="Protocol to use (grpc, http/protobuf, or http/json)",
        pattern=r"^(grpc|http/protobuf|http/json)$"
    )
    timeout: float = Field(
        default=10.0,
        description="Export timeout in seconds",
        gt=0
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to include in export requests"
    )
    compression: Optional[str] = Field(
        default=None,
        description="Compression to use (gzip, deflate, or none)",
        pattern=r"^(gzip|deflate|none)?$"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Path to certificate file for TLS verification"
    )
    insecure: bool = Field(
        default=False,
        description="Disable TLS verification (insecure)"
    )
    client_key_path: Optional[str] = Field(
        default=None,
        description="Path to client key file for mTLS"
    )
    client_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate file for mTLS"
    )

    @validator('endpoint')
    def validate_endpoint(cls, v: str) -> str:
        """Ensure endpoint has a valid scheme."""
        if not any(v.startswith(proto) for proto in ('http://', 'https://', 'grpc://', 'grpcs://')):
            raise ValueError("Endpoint must start with http://, https://, grpc://, or grpcs://")
        return v

class OTELBatchConfig(BaseModel):
    """Batch processing configuration."""
    max_export_batch_size: int = Field(
        default=512,
        description="Maximum number of spans to export in a single batch",
        gt=0
    )
    scheduled_delay_millis: int = Field(
        default=5000,
        description="Delay between export batches in milliseconds",
        gt=0
    )
    export_timeout_millis: int = Field(
        default=30000,
        description="Maximum time to wait for an export batch to complete in milliseconds",
        gt=0
    )
    max_queue_size: int = Field(
        default=2048,
        description="Maximum number of spans to buffer before dropping",
        gt=0
    )

class OTELSamplingConfig(BaseModel):
    """Sampling configuration."""
    sampler: str = Field(
        default="parentbased_always_on",
        description="Sampler to use (always_on, always_off, traceidratio, parentbased_*)",
        pattern=r"^(always_on|always_off|traceidratio|parentbased_always_on|parentbased_always_off|parentbased_traceidratio)$"
    )
    sampler_arg: float = Field(
        default=1.0,
        description="Argument for the sampler (e.g., probability for traceidratio)",
        ge=0.0,
        le=1.0
    )

class OTELLoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to log file (if None, logs to console only)"
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes",
        gt=0
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep",
        ge=0
    )
    propagate: bool = Field(
        default=True,
        description="Whether to propagate logs to the root logger"
    )

class OTELSDKConfig(BaseModel):
    """OpenTelemetry SDK configuration."""
    enabled: bool = Field(
        default=True,
        description="Whether OpenTelemetry SDK is enabled"
    )
    resource: OTELResourceConfig = Field(
        default_factory=OTELResourceConfig,
        description="Resource configuration"
    )
    exporter: OTELExporterConfig = Field(
        default_factory=OTELExporterConfig,
        description="Exporter configuration"
    )
    batch: OTELBatchConfig = Field(
        default_factory=OTELBatchConfig,
        description="Batch processing configuration"
    )
    sampling: OTELSamplingConfig = Field(
        default_factory=OTELSamplingConfig,
        description="Sampling configuration"
    )
    logging: OTELLoggingConfig = Field(
        default_factory=OTELLoggingConfig,
        description="Logging configuration"
    )
    propagators: List[str] = Field(
        default_factory=lambda: ["tracecontext", "baggage"],
        description="List of propagators to use"
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Whether metrics collection is enabled"
    )
    logs_enabled: bool = Field(
        default=False,
        description="Whether logs collection is enabled"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    @root_validator(pre=True)
    def set_defaults_based_on_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set default values based on environment variables."""
        # Set service name from environment if not explicitly set
        if 'resource' not in values:
            values['resource'] = {}
        
        resource = values['resource']
        
        if 'service_name' not in resource and 'OTEL_SERVICE_NAME' in os.environ:
            resource['service_name'] = os.environ['OTEL_SERVICE_NAME']
        
        if 'environment' not in resource and 'OTEL_ENV' in os.environ:
            resource['environment'] = os.environ['OTEL_ENV']
        elif 'environment' not in resource and 'ENVIRONMENT' in os.environ:
            resource['environment'] = os.environ['ENVIRONMENT']
        
        return values

class OTELConfig:
    """OpenTelemetry Configuration Manager"""
    
    def __init__(self):
        # Default configuration with all supported options
        self._config: Dict[str, Any] = {
            # General SDK configuration
            "enabled": True,
            "sdk_disabled": False,
            "service_name": "MyJobSpyAI",
            "service_namespace": "",
            "service_version": "1.0.0",
            "deployment_environment": "development",
            "service_instance_id": "",
            
            # Resource attributes (key=value pairs)
            "resource_attributes": {},
            
            # OTLP Exporter configuration
            "otlp_endpoint": "http://localhost:4317",
            "otlp_protocol": "grpc",  # grpc, http/protobuf, http/json
            "otlp_headers": {},
            "otlp_compression": None,  # gzip, deflate, none
            "otlp_timeout": 10.0,  # seconds
            "otlp_certificate": None,  # Path to certificate file
            "otlp_insecure": False,  # Disable TLS verification
            "otlp_client_key": None,  # Path to client key file (for mTLS)
            "otlp_client_certificate": None,  # Path to client certificate file (for mTLS)
            
            # Batch Span Processor configuration
            "max_export_batch_size": 512,
            "scheduled_delay_millis": 5000,
            "export_timeout_millis": 30000,
            "max_queue_size": 2048,
            
            # Sampling configuration
            "sampler": "parentbased_always_on",  # always_on, always_off, traceidratio, parentbased_*
            "sampler_arg": 1.0,  # For traceidratio sampler (0.0 - 1.0)
            
            # Metrics configuration
            "metrics_enabled": True,
            "metrics_export_interval_millis": 60000,
            "metrics_export_timeout_millis": 30000,
            
            # Logs configuration
            "logs_enabled": False,
            "logs_export_interval_millis": 15000,
            "logs_export_timeout_millis": 30000,
            
            # Resource Detection
            "detectors": ["all"],  # all, env, process, host, container, etc.
            
            # Propagation
            "propagators": ["tracecontext", "baggage"],  # tracecontext, baggage, b3, b3multi, jaeger, etc.
            
            # SDK Limits
            "attribute_count_limit": 128,
            "attribute_value_length_limit": None,  # None means unlimited
            "span_attribute_count_limit": 128,
            "span_event_count_limit": 128,
            "span_link_count_limit": 1000,
            "event_attribute_count_limit": 128,
            "link_attribute_count_limit": 128,
            
            # Debugging
            "debug": False,
            "log_level": "INFO",
            "console": False,
        }
        
        # Environment variable mappings
        self._env_mappings = self._get_env_mappings()
        
    def _get_env_mappings(self) -> Dict[str, tuple]:
        """Return environment variable to config key mappings with converters"""
        return {
            # General SDK configuration
            'OTEL_SDK_DISABLED': ('sdk_disabled', self._str_to_bool),
            'OTEL_SERVICE_NAME': ('service_name', str),
            'OTEL_SERVICE_NAMESPACE': ('service_namespace', str),
            'OTEL_SERVICE_VERSION': ('service_version', str),
            'OTEL_SERVICE_INSTANCE_ID': ('service_instance_id', str),
            'OTEL_RESOURCE_ATTRIBUTES': ('resource_attributes', self._parse_key_value_pairs),
            'OTEL_DEPLOYMENT_ENVIRONMENT': ('deployment_environment', str),
            
            # OTLP Exporter configuration
            'OTEL_EXPORTER_OTLP_ENDPOINT': ('otlp_endpoint', str),
            'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT': ('otlp_endpoint', str),
            'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT': ('otlp_metrics_endpoint', str),
            'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT': ('otlp_logs_endpoint', str),
            'OTEL_EXPORTER_OTLP_PROTOCOL': ('otlp_protocol', str),
            'OTEL_EXPORTER_OTLP_HEADERS': ('otlp_headers', self._parse_key_value_pairs),
            'OTEL_EXPORTER_OTLP_COMPRESSION': ('otlp_compression', str),
            'OTEL_EXPORTER_OTLP_TIMEOUT': ('otlp_timeout', float),
            'OTEL_EXPORTER_OTLP_CERTIFICATE': ('otlp_certificate', str),
            'OTEL_EXPORTER_OTLP_CLIENT_KEY': ('otlp_client_key', str),
            'OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE': ('otlp_client_certificate', str),
            'OTEL_EXPORTER_OTLP_INSECURE': ('otlp_insecure', self._str_to_bool),
            
            # Batch Span Processor configuration
            'OTEL_BSP_SCHEDULE_DELAY': ('scheduled_delay_millis', self._seconds_to_millis),
            'OTEL_BSP_EXPORT_TIMEOUT': ('export_timeout_millis', self._seconds_to_millis),
            'OTEL_BSP_MAX_QUEUE_SIZE': ('max_queue_size', int),
            'OTEL_BSP_MAX_EXPORT_BATCH_SIZE': ('max_export_batch_size', int),
            
            # Sampling configuration
            'OTEL_TRACES_SAMPLER': ('sampler', str),
            'OTEL_TRACES_SAMPLER_ARG': ('sampler_arg', float),
            
            # Metrics configuration
            'OTEL_METRICS_ENABLED': ('metrics_enabled', self._str_to_bool),
            'OTEL_METRIC_EXPORT_INTERVAL': ('metrics_export_interval_millis', self._seconds_to_millis),
            'OTEL_METRIC_EXPORT_TIMEOUT': ('metrics_export_timeout_millis', self._seconds_to_millis),
            
            # Logs configuration
            'OTEL_LOGS_EXPORTER': ('logs_enabled', lambda x: 'otlp' in x.lower() if x else False),
            
            # Resource Detection
            'OTEL_RESOURCE_DETECTORS': ('detectors', lambda x: [d.strip() for d in x.split(',') if d.strip()]),
            
            # Propagation
            'OTEL_PROPAGATORS': ('propagators', lambda x: [p.strip() for p in x.split(',') if p.strip()]),
            
            # SDK Limits
            'OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT': ('attribute_value_length_limit', int),
            'OTEL_ATTRIBUTE_COUNT_LIMIT': ('attribute_count_limit', int),
            'OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT': ('span_attribute_count_limit', int),
            'OTEL_SPAN_EVENT_COUNT_LIMIT': ('span_event_count_limit', int),
            'OTEL_SPAN_LINK_COUNT_LIMIT': ('span_link_count_limit', int),
            'OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT': ('event_attribute_count_limit', int),
            'OTEL_LINK_ATTRIBUTE_COUNT_LIMIT': ('link_attribute_count_limit', int),
            
            # Debugging
            'OTEL_DEBUG': ('debug', self._str_to_bool),
            'OTEL_LOG_LEVEL': ('log_level', str),
            'OTEL_PYTHON_LOG_CORRELATION': ('log_correlation', self._str_to_bool),
        }
    
    # Helper methods for value conversion
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        """Convert string to boolean"""
        return str(value).lower() in ('true', '1', 't', 'y', 'yes')
    
    @staticmethod
    def _parse_key_value_pairs(value: str) -> Dict[str, str]:
        """Parse key=value pairs from string"""
        if not value:
            return {}
        result = {}
        for pair in value.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                result[k.strip()] = v.strip()
        return result
    
    @staticmethod
    def _seconds_to_millis(seconds: Union[str, float, int]) -> int:
        """Convert seconds to milliseconds"""
        try:
            return int(float(seconds) * 1000)
        except (ValueError, TypeError):
            return 0
    
    def _update_from_env(self) -> None:
        """Update configuration from environment variables"""
        for env_var, (config_key, converter) in self._env_mappings.items():
            if env_var in os.environ:
                try:
                    self._config[config_key] = converter(os.environ[env_var])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {e}")
    
    def _update_from_cli(self) -> None:
        """Update configuration from command line arguments"""
        parser = argparse.ArgumentParser(description='OpenTelemetry Configuration', add_help=False)
        
        # Add arguments based on environment variable mappings
        for env_var, (config_key, _) in self._env_mappings.items():
            # Skip some environment variables that don't make sense as CLI args
            if env_var in ('OTEL_RESOURCE_ATTRIBUTES', 'OTEL_EXPORTER_OTLP_HEADERS'):
                continue
                
            # Convert env var name to CLI arg name (e.g., OTEL_SERVICE_NAME -> --otel-service-name)
            arg_name = '--' + env_var.lower().replace('_', '-')
            
            # Determine argument type and help text
            arg_type = str
            help_text = f"Override {env_var}"
            
            # Special handling for boolean values
            if self._config.get(config_key) is True or self._config.get(config_key) is False:
                parser.add_argument(arg_name, action='store_true', help=help_text)
                continue
                
            # Add the argument
            parser.add_argument(arg_name, type=arg_type, help=help_text)
        
        # Parse known args only to avoid conflicts with other argument parsers
        args, _ = parser.parse_known_args()
        
        # Update config from parsed args
        for env_var, (config_key, converter) in self._env_mappings.items():
            arg_name = env_var.lower().replace('_', '-')
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self._config[config_key] = converter(getattr(args, arg_name))
    
    def _update_from_settings(self, settings: Any) -> None:
        """Update configuration from application settings"""
        if hasattr(settings, 'opentelemetry'):
            # Handle both dictionary and object-style settings
            if hasattr(settings.opentelemetry, 'dict'):
                settings_dict = settings.opentelemetry.dict(exclude_unset=True)
            else:
                settings_dict = settings.opentelemetry
            
            # Update config with settings, handling nested dictionaries
            for key, value in settings_dict.items():
                if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                    self._config[key].update(value)
                else:
                    self._config[key] = value
    
    def get_config(self, settings: Any = None) -> Dict[str, Any]:
        """
        Get the complete OpenTelemetry configuration with proper precedence:
        1. Command-line arguments
        2. Environment variables
        3. Application settings (from config.yaml)
        4. Default values
        """
        # Start with defaults
        config = self._config.copy()
        
        # Apply settings from config.yaml (lowest priority)
        if settings is not None:
            self._update_from_settings(settings)
        
        # Apply environment variables (medium priority)
        self._update_from_env()
        
        # Apply command-line arguments (highest priority)
        self._update_from_cli()
        
        # Final validation and cleanup
        self._validate_config()
        
        return config
    
    def _validate_config(self) -> None:
        """Validate the configuration and ensure consistency"""
        # If SDK is disabled, disable all components
        if self._config.get('sdk_disabled'):
            self._config['enabled'] = False
            
        # If OpenTelemetry is disabled, ensure all components are disabled
        if not self._config.get('enabled', True):
            self._config['metrics_enabled'] = False
            self._config['logs_enabled'] = False
            
        # Ensure sampler_arg is within valid range for traceidratio
        if 'traceidratio' in self._config.get('sampler', '') and \
           not (0.0 <= self._config.get('sampler_arg', 1.0) <= 1.0):
            logger.warning("Invalid sampler_arg for traceidratio sampler. Must be between 0.0 and 1.0. Using 1.0.")
            self._config['sampler_arg'] = 1.0


# Singleton instance
_global_otel_config = OTELConfig()


def get_otel_config(settings: Any = None) -> Dict[str, Any]:
    """
    Get the OpenTelemetry configuration.
    
    Args:
        settings: Optional settings object (e.g., from config.yaml)
        
    Returns:
        dict: The complete OpenTelemetry configuration
    """
    return _global_otel_config.get_config(settings)
