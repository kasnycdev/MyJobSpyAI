# myjobspyai.filtering package

SSSSSSSSSS

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> maxdepth
>
> :   4
>
> myjobspyai.filtering.filter myjobspyai.filtering.filter_utils

# myjobspyai.filtering

## Overview

The filtering module provides functionality for filtering and processing job analysis results. It includes base classes and utilities for implementing custom filters.

## Classes

### Filter

Base class for filtering operations.

#### Initialization

```python
filter = Filter(
    config={
        "type": "location",
        "value": "New York",
        "case_sensitive": False
    }
)
```

#### Methods

##### \_validate_filters

```python
# Validate the filter configuration
is_valid = filter._validate_filters()
```

Validates the current filter configuration against requirements.

##### apply

```python
# Apply the filter to input data
filtered_results = await filter.apply(job_results)
```

Applies the filter to the input data, returning filtered results.

##### close

```python
# Clean up resources
await filter.close()
```

Closes the filter's resources, including any active connections or sessions.

### FilterUtils

Utility class for filter operations.

#### Methods

##### \_apply_filters

```python
# Apply multiple filters internally
filtered = await FilterUtils._apply_filters(job_results, [filter1, filter2])
```

Internal method for applying multiple filters in sequence.

##### \_validate_filter_chain

```python
# Validate a chain of filters
is_valid = await FilterUtils._validate_filter_chain([filter1, filter2])
```

Validates a chain of filters for compatibility and correctness.

##### apply_filters

```python
# Apply multiple filters to input data
filtered_results = await FilterUtils.apply_filters(job_results, [filter1, filter2])
```

Convenience method for applying multiple filters to input data.

##### create_filter_chain

```python
# Create a chain of filters
filter_chain = await FilterUtils.create_filter_chain(config_list)
```

Creates a chain of filters from configuration.

##### validate_filters

```python
# Validate filter configurations
valid = await FilterUtils.validate_filters(config_list)
```

Validates multiple filter configurations at once.

## Configuration

The filtering module supports the following configuration options:

```python
{
    "type": "location",    # Filter type (location, skill, experience, etc.)
    "value": "New York",   # Filter value
    "case_sensitive": False,  # Case sensitivity
    "min_score": 0.7,     # Minimum match score
    "exclude": False      # Whether to exclude matching items
}
```

## Error Handling

The filtering module implements robust error handling:

- Invalid filter configuration
- Filter chain validation failures
- Data processing errors
- Resource cleanup failures

All errors are wrapped in `FilterError` with detailed context and status codes.

## Usage Example

```python
from myjobspyai.filtering import Filter, FilterUtils

# Create filters
location_filter = Filter(
    config={
        "type": "location",
        "value": "New York",
        "case_sensitive": False
    }
)

skill_filter = Filter(
    config={
        "type": "skill",
        "value": "Python",
        "min_score": 0.8
    }
)

try:
    # Apply filters
    filtered_results = await FilterUtils.apply_filters(
        job_results,
        [location_filter, skill_filter]
    )
    print(f"Filtered results: {filtered_results}")
finally:
    # Clean up
    await location_filter.close()
    await skill_filter.close()
```

## Best Practices

1. Always validate filter configurations
2. Use appropriate filter types
3. Handle errors gracefully
4. Clean up resources properly
5. Monitor performance
6. Use configuration validation
7. Follow security best practices
8. Implement proper error handling
9. Use filter chaining for complex logic
10. Monitor resource usage
