# MyJobSpy AI - Code Organization and Refactoring Plan

## Current Structure

```
src/myjobspyai/
├── analysis/              # Analysis components
├── cli/                   # Command-line interface
├── config/                # Configuration management
├── core/                  # Core application logic
├── filtering/             # Data filtering logic
├── llm/                   # LLM integration
├── models/                # Data models
├── prompts/               # Prompt templates
├── scrapers/              # Job board scrapers
├── services/              # Service layer
└── utils/                 # Utility functions
```

## Proposed Structure

### Core Principles
1. **Separation of Concerns**: Each directory has a single responsibility
2. **Testability**: Code is organized to facilitate unit and integration testing
3. **Maintainability**: Clear boundaries between components
4. **Extensibility**: Easy to add new features or scrapers

### Directory Structure

```
src/myjobspyai/
├── core/                    # Core application logic
│   ├── __init__.py
│   ├── analyzer.py         # Job/resume analysis
│   ├── matcher.py          # Matching algorithms
│   └── processor.py        # Data processing pipelines
│
├── models/                # Data models (Pydantic)
│   ├── __init__.py
│   ├── base.py             # Base models and mixins
│   ├── job.py             # Job-related models
│   ├── resume.py          # Resume models
│   └── analysis.py        # Analysis result models
│
├── scrapers/             # Web scrapers
│   ├── __init__.py
│   ├── base.py            # Base scraper class
│   ├── linkedin/          # LinkedIn scraper
│   ├── indeed/           # Indeed scraper
│   └── jobspy/           # JobSpy scraper
│
├── services/             # Service layer
│   ├── __init__.py
│   ├── database/         # Database operations
│   └── llm/             # LLM service integration
│
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── logging.py       # Logging configuration
│   └── http.py         # HTTP client utilities
│
├── config/             # Configuration
│   ├── __init__.py
│   └── settings.py     # App settings
│
└── cli/                # Command-line interface
    ├── __init__.py
    └── commands/       # CLI commands
```

## Refactoring Phases

### Phase 1: Model Standardization (1-2 days)
- [x] Modernize `base.py` with Pydantic v2
- [x] Update `job.py` with proper typing
- [x] Refactor `resume.py` to match patterns
- [x] Add comprehensive model tests

### Phase 2: Core Logic (2-3 days)
- [x] Move business logic from `main.py` to `services/`
  - Created `JobService` for job search and analysis
  - Created `ResumeService` for resume parsing and analysis
  - Implemented proper error handling and logging
- [x] Implement service layer
  - Added service interfaces for job search and resume analysis
  - Implemented proper dependency injection
- [x] Add proper error handling
  - Added comprehensive error handling in services
  - Implemented graceful degradation
- [x] Implement retry mechanisms
  - Added retry logic for API calls
  - Implemented exponential backoff

### Phase 3: Scrapers (2-3 days)
- [x] Standardize scraper interfaces
  - [Implementation Plan](./scraper_standardization_plan.md)
- [ ] Add proper error handling
- [ ] Implement rate limiting
- [ ] Add comprehensive tests

### Phase 4: CLI & Integration (1-2 days)
- [ ] Clean up CLI commands
- [ ] Add progress indicators
- [ ] Improve error reporting

## Best Practices

### Code Style
- Follow PEP 8
- Use type hints
- Document public APIs
- Keep functions small and focused

### Error Handling
- Use custom exceptions
- Add context to errors
- Implement proper logging

### Testing
- Aim for 80%+ coverage
- Test edge cases
- Add integration tests

## Migration Guide

### For Developers
1. Update imports to new structure
2. Use new service interfaces
3. Update tests

### For Contributors
1. Follow the new structure
2. Add tests for new code
3. Document public APIs

## Future Improvements
- Add async support throughout
- Implement caching layer
- Add more comprehensive logging
- Improve documentation
