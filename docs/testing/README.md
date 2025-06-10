# Testing in MyJobSpyAI

This document provides an overview of the testing strategy and guidelines for the MyJobSpyAI project.

## Test Organization

Tests are organized in the `tests/` directory with the following structure:

```
tests/
├── conftest.py         # Pytest fixtures and configuration
├── utils.py            # Test utilities and helpers
├── test_data/          # Test data files
├── unit/               # Unit tests
├── integration/        # Integration tests
├── e2e/                # End-to-end tests
└── performance/        # Performance tests
```

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Specific Test Types

- Unit tests:
  ```bash
  pytest tests/unit
  ```

- Integration tests:
  ```bash
  pytest tests/integration
  ```

- End-to-end tests:
  ```bash
  pytest tests/e2e
  ```

- Performance tests:
  ```bash
  pytest tests/performance
  ```

### Test Coverage

To run tests with coverage reporting:

```bash
pytest --cov=myjobspyai --cov-report=term-missing
```

## Writing Tests

### Unit Tests

- Test one thing at a time
- Mock external dependencies
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

### Integration Tests

- Test interactions between components
- Use real dependencies when possible
- Test error conditions and edge cases

### End-to-End Tests

- Test complete workflows
- Use test doubles for external services
- Focus on user journeys

## Fixtures

Common test fixtures are defined in `conftest.py`. Use these to reduce code duplication.

## Test Data

Store test data in the `test_data/` directory. Use descriptive file names and organize by feature or component.

## Best Practices

- Write tests before or alongside the code they test
- Keep tests fast and independent
- Avoid testing implementation details
- Use meaningful assertions
- Document complex test cases

## Continuous Integration

Tests are automatically run on push and pull requests via GitHub Actions. See `.github/workflows/tests.yml` for details.
