# MyJobSpyAI Tests

This directory contains the test suite for the MyJobSpyAI application.

## Running Tests

### Prerequisites

1. Install the test dependencies:

```bash
pip install -r ../requirements-test.txt
```

### Running All Tests

To run all tests:

```bash
pytest -v
```

### Running Specific Tests

To run a specific test file:

```bash
pytest tests/test_langchain_provider.py -v
```

To run a specific test function:

```bash
pytest tests/test_langchain_provider.py::test_langchain_provider_initialization -v
```

### Running with Coverage

To run tests with coverage report:

```bash
pytest --cov=analysis --cov-report=term-missing
```

### Running in Parallel

To run tests in parallel (using multiple CPU cores):

```bash
pytest -n auto
```

## Test Organization

- `test_*.py`: Test modules containing test functions
- `conftest.py`: Pytest configuration and fixtures
- `__init__.py`: Makes the directory a Python package

## Writing Tests

Follow these guidelines when writing tests:

1. **Test Naming**:
   - Test files should be named `test_*.py`
   - Test functions should be named `test_*`
   - Test classes should be named `Test*`

2. **Fixtures**:
   - Common test fixtures should be defined in `conftest.py`
   - Use fixtures for common setup/teardown code

3. **Async Tests**:
   - Use `@pytest.mark.asyncio` for async test functions
   - Use `AsyncMock` for mocking async functions

4. **Mocks**:
   - Use `unittest.mock` or `pytest-mock` for mocking
   - Keep mocks simple and focused

## Continuous Integration

Tests are automatically run in CI/CD pipelines. Make sure all tests pass before merging code.

## Debugging Tests

To debug a failing test:

1. Run the test with `-s` to see print output:
   ```bash
   pytest tests/test_langchain_provider.py -v -s
   ```

2. Use `pdb` to debug:
   ```python
   import pdb

   pdb.set_trace()  # Add this line where you want to break
   ```
   Then run with `-s` flag.

## Test Coverage

To generate an HTML coverage report:

```bash
pytest --cov=analysis --cov-report=html
open htmlcov/index.html  # On macOS/Linux
```
