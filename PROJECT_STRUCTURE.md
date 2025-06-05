# MyJobSpyAI Project Structure

```
MyJobSpyAI/
├── config.yaml             # Main configuration file
├── config.example.yaml     # Example configuration
├── requirements.txt        # Main dependencies
├── requirements-test.txt   # Test dependencies
│
├── src/                   # Source code
│   └── myjobspyai/
│       ├── __init__.py
│       ├── __main__.py     # CLI entry point
│       ├── config.py       # Configuration management
│       ├── main.py         # Main application logic
│       ├── main_matcher.py # Job matching logic
│       │
│       ├── analysis/       # AI analysis components
│       │   ├── __init__.py
│       │   ├── analyzer.py # Main analysis logic
│       │   └── providers/  # LLM provider implementations
│       │
│       ├── filtering/     # Job filtering logic
│       ├── llm/           # LLM providers
│       ├── models/        # Data models
│       ├── parsers/       # Resume and job parsing
│       ├── prompts/       # Prompt templates
│       ├── scrapers/      # Job site scrapers
│       └── utils/         # Utility functions
│
├── tests/               # Test suite
├── data/                 # Input data (e.g., resumes)
├── docs/                # Documentation
├── logs/                # Log files (ignored by git)
└── venv/                # Virtual environment (ignored by git)
```

## Key Files and Directories

### Configuration
- `config.yaml`: Main configuration file for the application
- `config.example.yaml`: Example configuration with all available options
- `.env`: Environment variables (not versioned, create from .env.example)

### Source Code (`src/myjobspyai/`)
- `__main__.py`: Entry point when running as a module (`python -m myjobspyai`)
- `main.py`: Main application logic
- `config.py`: Configuration management and validation
- `main_matcher.py`: Job matching and scoring logic

### Core Components
- `analysis/`: AI analysis components and LLM providers
- `filtering/`: Job filtering logic
- `llm/`: LLM provider implementations
- `models/`: Data models and schemas
- `parsers/`: Resume and job description parsing
- `prompts/`: Templates for LLM prompts
- `scrapers/`: Job site scrapers
- `utils/`: Utility functions and helpers

### Development
- `tests/`: Unit and integration tests
- `docs/`: Project documentation
- `data/`: Sample data and test fixtures
- `logs/`: Application logs (not versioned)

## Ignored Files

The following files and directories are ignored by git (see `.gitignore`):
- `venv/`: Python virtual environment
- `.venv/`: Alternative virtual environment directory
- `__pycache__/`: Python bytecode cache
- `logs/`: Application logs
- `data/`: Local data files
- `.env`: Environment variables
- `.pytest_cache/`: Test cache
- `.mypy_cache/`: Type checking cache
- `.ruff_cache/`: Linting cache
