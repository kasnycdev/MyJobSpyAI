# MyJobSpyAI

[![PyPI version](https://badge.fury.io/py/myjobspyai.svg)](https://badge.fury.io/py/myjobspyai)
[![Python Version](https://img.shields.io/pypi/pyversions/myjobspyai)](https://pypi.org/project/myjobspyai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Codecov](https://codecov.io/gh/kasnycdev/MyJobSpyAI/graph/badge.svg)](https://codecov.io/gh/kasnycdev/MyJobSpyAI)

**MyJobSpyAI** is an advanced job search and analysis platform that combines web scraping with AI-powered analysis to help you find the perfect job match. It leverages multiple LLM backends (Ollama, OpenAI-compatible APIs, Google Gemini) for intelligent job matching, filtering, and analysis.

## 🌟 Features

- **Advanced RAG Pipeline**: Implements Retrieval-Augmented Generation for intelligent job matching
- **Multi-LLM Support**: Seamlessly works with Ollama, OpenAI, and Google Gemini
- **Vector Database Integration**: Utilizes Milvus for efficient semantic search and similarity matching
- **Asynchronous Processing**: Built with asyncio for high-performance job processing
- **Comprehensive Analysis**: Provides in-depth analysis of job listings and resume matching
- **Observability**: Integrated OpenTelemetry for monitoring and debugging
- **Type Safety**: Full type hints and mypy integration for better code quality
- **Testing**: Comprehensive test suite with pytest and coverage reporting

## 🏗️ Project Architecture

### Core Components

#### 1. Configuration System (`/config`)
- Central `config.yaml` with environment variable overrides
- Multi-layered configuration (env vars > YAML > defaults)
- Provider-specific settings (Ollama, OpenAI, Gemini)
- Feature flags for streaming, caching, and logging

#### 2. Analysis Module (`/myjobspyai/analysis`)
- **BaseAnalyzer**: Core analysis functionality with retry logic and error handling
- **JobAnalyzer**: Processes and analyzes job descriptions
- **ResumeAnalyzer**: Extracts and normalizes resume data
- **Provider System**: Unified interface for multiple LLM backends

#### 3. RAG Pipeline (`/myjobspyai/rag`)
- **RAGProcessor**: Main pipeline controller for document processing
- **JobRAGProcessor**: Specialized job data processing
- **TextProcessor**: Advanced text chunking and processing
- **Milvus Integration**: Vector similarity search and storage

#### 4. Data Processing (`/myjobspyai/parsers`)
- **Job Parser**: Extracts structured data from job listings
- **Resume Parser**: Processes multiple formats (PDF, DOCX, TXT)
- **Schema Validation**: Pydantic models for data integrity

#### 5. Filtering System (`/myjobspyai/filtering`)
- Custom filter chains
- Score-based filtering
- Configurable thresholds

### Project Structure

```
MyJobSpyAI/
├── myjobspyai/               # Main package
│   ├── analysis/            # Job analysis and matching logic
│   │   ├── prompts/        # Prompt templates for LLM analysis
│   │   ├── providers/      # LLM provider implementations
│   │   ├── base.py         # Base analyzer class
│   │   ├── analyzer.py     # Core analysis logic
│   │   └── models.py       # Data models
│   │
│   ├── filtering/        # Job filtering and processing
│   │   ├── filter.py       # Core filtering logic
│   │   └── filter_utils.py # Filter utilities
│   │
│   ├── parsers/          # Data parsing
│   │   ├── job_parser.py   # Job description parsing
│   │   └── resume_parser.py # Resume/CV parsing
│   │
│   ├── rag/              # RAG pipeline
│   │   ├── rag_processor.py # Main RAG implementation
│   │   ├── milvus.py       # Vector store integration
│   │   └── text_processor.py # Text processing
│   │
│   ├── utils/           # Utility functions
│   │   ├── config_utils.py # Configuration management
│   │   ├── logging_utils.py # Logging setup
│   │   └── monitoring/     # Monitoring and observability
│   │
│   ├── __init__.py      # Package initialization
│   └── exceptions.py       # Custom exceptions
│
├── config/              # Configuration files
│   └── monitoring.yaml    # Monitoring configuration
│
├── tests/               # Test suite
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
│
├── .env.example        # Example environment variables
├── config.yaml           # Main application configuration
├── main.py               # Main entry point
├── pyproject.toml        # Project metadata and dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/kasnycdev/MyJobSpyAI.git
cd MyJobSpyAI

# Install dependencies using pip
pip install -e .[all]  # Install with all optional dependencies

# Or using Poetry
poetry install --with all  # Install with all optional dependencies

# For development, install additional dependencies
pip install -e ".[all,dev]"  # Using pip
# or
poetry install --with all,dev  # Using Poetry

# Copy the example environment file and update with your credentials
cp .env.example .env
# Edit .env with your actual API keys and settings
```

## ⚙️ Configuration

### Active Configuration

- **Default Provider**: Ollama
- **Model**: Mistral-7B-Instruct (default: `mistral:instruct`)
- **Chunking**:
  - Size: 3000 characters
  - Overlap: 200 characters
  - Semantic: Enabled
- **Caching**:
  - Enabled: Yes
  - TTL: 1 hour
  - Max size: 1000 items
- **Ollama Configuration**:
  - Base URL: http://localhost:11434
  - Default Model: mistral:instruct
  - Timeout: 300 seconds
  - Max Retries: 3

### Environment Variables

1. Copy the example environment file and update with your credentials:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your actual API keys and settings. The following environment variables are supported:

   ```bash
   # LLM Providers
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_google_key
   
   # Application Settings
   DEBUG=false
   LOG_LEVEL=INFO
   
   # Ollama Configuration (Default)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=mistral:instruct
   OLLAMA_TIMEOUT=300
   OLLAMA_MAX_RETRIES=3
   
   # Milvus Vector Database
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   
   # Scraping Configuration
   DEFAULT_SITES=linkedin,indeed
   DEFAULT_RESULTS_LIMIT=10
   DEFAULT_DAYS_OLD=14
   
   # Network
   HTTP_PROXY=http://proxy.example.com:8080
   HTTPS_PROXY=http://proxy.example.com:8080
   ```

### Configuration File (config.yaml)

The application uses a `config.yaml` file for comprehensive configuration. Here's an example with all available options:

```yaml
# Application Configuration
app:
  name: "MyJobSpyAI"
  version: "1.0.0"
  debug: ${DEBUG:-false}
  log_level: "${LOG_LEVEL:-INFO}"
  environment: "development"

# LLM Configuration
llm:
  provider: "ollama"  # Options: ollama, openai, gemini
  model: "mistral:instruct"  # Default model to use
  
  # Ollama provider configuration
  ollama:
    base_url: "${OLLAMA_BASE_URL:-http://localhost:11434}"
    model: "${OLLAMA_MODEL:-mistral:instruct}"
    timeout: ${OLLAMA_TIMEOUT:-300}  # seconds
    max_retries: ${OLLAMA_MAX_RETRIES:-3}
  
  # Global streaming settings
  streaming:
    enabled: true
    chunk_size: 128
    timeout: 30
    buffer_size: 5

  # Provider-specific configurations
  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    
  gemini:
    api_key: ${GEMINI_API_KEY}
    model: "gemini-pro"
    temperature: 0.7
    
  ollama:
    base_url: ${OLLAMA_BASE_URL:-http://localhost:11434}
    model: ${OLLAMA_MODEL:-mistral:instruct}
    timeout: 300

# RAG Pipeline Configuration
rag:
  chunk_size: 3000
  chunk_overlap: 200
  semantic_chunking: true
  
  # Vector Store (Milvus)
  milvus:
    host: ${MILVUS_HOST:-localhost}
    port: ${MILVUS_PORT:-19530}
    collection_name: "job_listings"
    embedding_dim: 768
    
  # Embeddings
  embeddings:
    model: "all-MiniLM-L6-v2"
    device: "cpu"  # or "cuda" for GPU acceleration

# Scraping Configuration
scraping:
  default_sites: ${DEFAULT_SITES:-linkedin,indeed}
  default_results_limit: ${DEFAULT_RESULTS_LIMIT:-10}
  default_days_old: ${DEFAULT_DAYS_OLD:-14}
  default_country: "usa"
  
  # Job Filters
  is_remote: false
  job_type: null  # fulltime, contract, parttime, etc.
  easy_apply: null
  distance: 50  # miles
  
  # Proxies (optional)
  # proxies:
  #   - "http://proxy1.example.com:8080"
  #   - "http://proxy2.example.com:8080"
  # ca_cert: "/path/to/ca_cert.pem"

# Logging Configuration
logging:
  level: "${LOG_LEVEL:-INFO}"
  file: "logs/app.log"
  max_size: 10485760  # 10MB
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
# Monitoring (OpenTelemetry)
monitoring:
  enabled: true
  service_name: "myjobspyai"
  otlp_endpoint: "http://localhost:4317"
  metrics_enabled: true
  traces_enabled: true
  logs_enabled: true
```

### Configuration Precedence

Configuration values are loaded in the following order of precedence:
1. Command-line arguments (highest)
2. Environment variables
3. `config.yaml` file
4. Default values in the code (lowest)

### Accessing Configuration in Code

Use the `settings` object to access configuration values:

```python
from myjobspyai.utils.config_utils import settings

# Access configuration values
provider = settings.llm.provider
model = settings.llm.model

# Access nested configuration
chunk_size = settings.rag.chunk_size
milvus_host = settings.rag.milvus.host

# Check if debug mode is enabled
if settings.app.debug:
    print("Debug mode is enabled")
    
# Access with defaults
job_type = settings.scraping.job_type or "fulltime"
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/kasnycdev/MyJobSpyAI.git
cd MyJobSpyAI
```

2. **Set up a virtual environment** (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

Using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install -r requirements.txt -r requirements-test.txt
```

4. **Set up environment variables**

Copy the example environment file and update the values:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Edit `config.yaml` to configure your job search parameters, LLM settings, and RAG pipeline:

```yaml
# LLM Configuration
llm:
  provider: "ollama"  # or "openai", "gemini"
  model: "codellama:latest"
  temperature: 0.7
  max_tokens: 2048

# Ollama Configuration
ollama:
  base_url: "http://localhost:11434"
  timeout: 300

# Milvus Configuration
milvus:
  host: "localhost"
  port: 19530
  collection_name: "job_listings"
  embedding_dim: 768

# RAG Pipeline
rag:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 5

# Job Search
job_search:
  keywords: "software engineer"
  location: "New York, NY"
  distance: 25  # miles
  job_types: ["full-time", "contract"]
  remote: true
  min_salary: 100000
```

### Usage

Run the job search pipeline:

```bash
python -m myjobspyai.search --config config.yaml
```

Or use the Python API:

```python
from myjobspyai import JobSearchPipeline

pipeline = JobSearchPipeline.from_config("config.yaml")
results = pipeline.search()
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ --cov=myjobspyai --cov-report=term-missing
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or support, please open an issue on [GitHub](https://github.com/kasnycdev/MyJobSpyAI/issues).

## 🙏 Acknowledgments

- Built with ❤️ by the MyJobSpyAI team
- Powered by [Ollama](https://ollama.ai/), [Milvus](https://milvus.io/), and other amazing open-source projects
