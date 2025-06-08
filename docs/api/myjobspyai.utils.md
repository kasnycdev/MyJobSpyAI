# myjobspyai.utils

## Overview

The utils module provides utility classes and functions for common operations across the application. It includes utilities for async operations, environment variables, file handling, HTTP requests, logging, prompts, and validation.

## Classes

### AsyncUtils

Utility class for async operations.

#### Methods

##### \_run_async

```python
# Run an async function internally
result = await AsyncUtils._run_async(async_func, *args, **kwargs)
```

Internal method to run an async function with error handling.

##### run_async

```python
# Run an async function
result = await AsyncUtils.run_async(async_func, *args, **kwargs)
```

Public method to run an async function with proper error handling and resource cleanup.

### Env

Utility class for environment variables.

#### Methods

##### \_get_env

```python
# Get an environment variable internally
value = Env._get_env("API_KEY", default="default_value")
```

Internal method to get an environment variable with type conversion.

##### get_env

```python
# Get an environment variable
value = Env.get_env("API_KEY", default="default_value")
```

Public method to safely get environment variables with validation.

### Files

Utility class for file operations.

#### Methods

##### \_read_file

```python
# Read a file internally
content = await Files._read_file("path/to/file.txt")
```

Internal method to read file content with error handling.

##### read_file

```python
# Read a file
content = await Files.read_file("path/to/file.txt")
```

Public method to safely read file content with proper resource cleanup.

### HttpClient

Utility class for HTTP requests.

#### Methods

##### \_get

```python
# Make a GET request internally
response = await HttpClient._get("https://api.example.com")
```

Internal method to make GET requests with error handling.

##### \_post

```python
# Make a POST request internally
response = await HttpClient._post("https://api.example.com", data={"key": "value"})
```

Internal method to make POST requests with error handling.

##### get

```python
# Make a GET request
response = await HttpClient.get("https://api.example.com")
```

Public method to make GET requests with proper error handling and resource cleanup.

##### post

```python
# Make a POST request
response = await HttpClient.post("https://api.example.com", data={"key": "value"})
```

Public method to make POST requests with proper error handling and resource cleanup.

### Logging

Utility class for logging.

#### Methods

##### \_setup_logger

```python
# Set up a logger internally
logger = Logging._setup_logger("my_logger", level="INFO")
```

Internal method to configure a logger with proper handlers and formatters.

##### setup_logger

```python
# Set up a logger
logger = Logging.setup_logger("my_logger", level="INFO")
```

Public method to configure a logger with proper error handling.

### LoggingUtils

Utility class for logging operations.

#### Methods

##### \_log_error

```python
# Log an error internally
LoggingUtils._log_error(exception, context="operation")
```

Internal method to log errors with proper context and traceback.

##### log_error

```python
# Log an error
LoggingUtils.log_error(exception, context="operation")
```

Public method to log errors with proper error handling and context.

### Prompts

Utility class for prompts.

#### Methods

##### \_get_prompt

```python
# Get a prompt internally
prompt = Prompts._get_prompt("job_analysis")
```

Internal method to retrieve a prompt template.

##### get_prompt

```python
# Get a prompt
prompt = Prompts.get_prompt("job_analysis")
```

Public method to safely retrieve and format prompt templates.

### Validation

Utility class for validation.

#### Methods

##### \_validate_config

```python
# Validate a configuration internally
is_valid = Validation._validate_config(config, schema)
```

Internal method to validate configuration against a schema.

##### validate_config

```python
# Validate a configuration
is_valid = Validation.validate_config(config, schema)
```

Public method to validate configuration with proper error handling.

## Usage Example

```python
from myjobspyai.utils import AsyncUtils, Env, Files, HttpClient, Logging, Prompts, Validation

# Async operations
result = await AsyncUtils.run_async(async_func, *args, **kwargs)

# Environment variables
api_key = Env.get_env("API_KEY", default="default_value")

# File operations
content = await Files.read_file("path/to/file.txt")

# HTTP requests
response = await HttpClient.get("https://api.example.com")

# Logging
logger = Logging.setup_logger("my_logger", level="INFO")

# Prompts
prompt = Prompts.get_prompt("job_analysis")

# Validation
is_valid = Validation.validate_config(config, schema)
```

## Best Practices

1. Always use proper error handling
2. Clean up resources properly
3. Use appropriate logging levels
4. Validate configurations
5. Handle async operations safely
6. Use environment variables for configuration
7. Follow security best practices
8. Implement proper error handling
9. Use type hints and validation
10. Monitor resource usage
