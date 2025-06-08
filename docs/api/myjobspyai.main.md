# myjobspyai.main module

## Overview

The main module provides the entry point and core functionality for the MyJobSpyAI application. It handles command-line arguments, job analysis, and resource management.

## Classes

### Main

Main class for MyJobSpyAI application.

#### Initialization

```python
main = Main()
```

#### Methods

##### \_analyze_jobs

```python
# Analyze jobs using the configured analyzer
await main._analyze_jobs(job_list)
```

Internal method that processes the job analysis using the configured analyzer.

##### \_load_job_mandates

```python
# Load job mandates from file
job_list = await main._load_job_mandates("jobs.json")
```

Loads job data from a specified file, supports multiple file formats.

##### \_parse_args

```python
# Parse command line arguments
args = main._parse_args()
```

Parses command-line arguments and validates them.

##### \_setup_logging

```python
# Set up logging configuration
main._setup_logging()
```

Configures logging with appropriate handlers and formatters.

##### analyze_jobs

```python
# Analyze jobs
results = await main.analyze_jobs(job_list)
```

Public method for analyzing jobs. Handles setup and cleanup.

##### close

```python
# Clean up resources
await main.close()
```

Closes the analyzer and releases all resources.

##### load_job_mandates

```python
# Load job mandates
job_list = await main.load_job_mandates("jobs.json")
```

Public method for loading job data from file.

##### main

```python
# Main entry point
if __name__ == "__main__":
    main.main()
```

Main entry point for the application.

##### main_async

```python
# Async main function
await main.main_async()
```

Async version of the main function for standalone execution.

##### parse_args

```python
# Parse command line arguments
args = main.parse_args()
```

Public method for parsing command-line arguments.

##### setup_logging

```python
# Set up logging
main.setup_logging()
```

Public method for configuring logging.

## Command Line Usage

```bash
# Basic usage
python -m myjobspyai.main --jobs jobs.json --output results.json

# With resume analysis
python -m myjobspyai.main --jobs jobs.json --resume resume.pdf --output results.json

# With filtering
python -m myjobspyai.main --jobs jobs.json --filter "location:New York" --output results.json
```

## Configuration

The main module supports the following configuration options:

```python
{
    "provider": "openai",  # LLM provider to use
    "model": "gpt-3.5-turbo",  # Model name
    "temperature": 0.7,    # Temperature for randomness
    "max_tokens": 2000,    # Maximum tokens in response
    "api_key": "your-api-key",  # API key (optional)
    "output_format": "json",  # Output format (json, csv, etc.)
    "log_level": "INFO",    # Logging level
    "trace_enabled": True   # Enable OpenTelemetry tracing
}
```

## Error Handling

The main module implements comprehensive error handling:

- Invalid configuration
- File loading errors
- Analysis failures
- Resource cleanup
- Network errors
- Model response validation

All errors are logged with detailed context and stack traces.

## OpenTelemetry Integration

The main module includes OpenTelemetry instrumentation for:

- Application startup
- Job loading
- Analysis operations
- Resource management
- Error tracking

Traces are tagged with:
- Operation name
- Status
- Duration
- Resource usage

## Best Practices

1. Always validate input data
2. Use appropriate logging levels
3. Clean up resources properly
4. Handle errors gracefully
5. Monitor performance
6. Use configuration validation
7. Follow security best practices
8. Implement proper error handling
9. Use structured output formats
10. Monitor resource usage
