# MyJobSpy AI: Advanced Job Search and Analysis Tool

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MyJobSpy AI enhances job searching by combining the scraping power of **[JobSpy](https://github.com/speedyapply/JobSpy)** with advanced Generative AI analysis and filtering capabilities.

## 📖 Documentation

### Local Development

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements-docs.txt

# Build the documentation
cd docs
make html

# View the documentation
python -m http.server 8000 --directory _build/html
```

### Windsurf IDE Integration

The project includes full support for Windsurf IDE with the following features:

1. **Documentation Preview**
   - Real-time preview of RST and Markdown files
   - Live reload on file changes
   - Syntax highlighting and validation

2. **Quick Start**
   ```bash
   # Start the live documentation server
   ./scripts/serve_docs.sh

   # Or use the built-in Windsurf commands
   # - Press F5 to start debugging
   # - Use the command palette (Ctrl+Shift+P) and search for "Sphinx: Live Preview"
   ```

3. **Key Features**
   - **Auto-completion** for RST and Markdown
   - **Live Preview** with instant updates
   - **Sphinx Integration** with build and serve commands
   - **Python Environment** with all dependencies

4. **Keyboard Shortcuts**
   - `Ctrl+Shift+V`: Toggle preview
   - `Ctrl+K V`: Open preview to the side
   - `F5`: Start debugging
   - `Ctrl+Shift+B`: Run build task

### Cascade AI Deployment

The documentation is configured to be deployed to Cascade AI. The deployment is handled automatically through the `.cascade/config.yaml` configuration.

1. **Prerequisites**:
   - Cascade AI CLI installed and configured
   - Access to the Cascade AI project

2. **Deployment**:
   ```bash
   # Build and deploy the documentation
   ./scripts/deploy_docs.sh

   # Or manually trigger deployment through Cascade AI CLI
   cascade deploy
   ```

3. **Configuration**:
   - Documentation source: `docs/`
   - Build output: `docs/_build/html`
   - Deploy directory: `public`

### Development Workflow

1. **Editing Documentation**
   - Edit files in the `docs/` directory
   - Use RST or Markdown syntax
   - Preview changes in real-time

2. **Building**
   ```bash
   # Full clean build
   make clean html

   # Incremental build (faster)
   make html
   ```

3. **Testing**
   ```bash
   # Check for broken links
   make linkcheck

   # Check spelling
   make spelling

   # Run documentation tests
   make doctest
   ```

## ✨ Features

### Core Functionality
- **Multi-Site Job Scraping**: Aggregates listings from LinkedIn, Indeed, Glassdoor, and more
- **AI-Powered Analysis**: Leverages state-of-the-art LLMs for job matching and analysis
- **Resume Integration**: Parses and analyzes your resume for better job matching
- **Advanced Filtering**: Filter by salary, location, job type, and more
- **Customizable Search**: Fine-tune search parameters via config file or CLI
- **Unified LLM Interface**: Single interface for multiple LLM providers via LangChain

### Technical Highlights
- **Asynchronous Processing**: Fast, concurrent job processing
- **Modular Architecture**: Easy to extend and customize
- **Robust Error Handling**: Graceful recovery from failures
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Comprehensive Logging**: Structured logging with multiple log levels and rotation

## 📚 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
  - [Job Search Configuration](#job-search-configuration)
  - [LLM Configuration](#llm-configuration)
  - [Logging Configuration](#logging-configuration)
- [Job Search Features](#-job-search-features)
- [LLM Integration](#-llm-integration)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MyJobSpyAI.git
   cd MyJobSpyAI
   ```

2. **Set up a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure your settings**:
   Copy the example configuration and update it with your preferences:
   ```bash
   cp config.example.yaml config.yaml
   ```

4. **Run the application**:
   ```bash
   python -m myjobspyai
   ```

## ⚙️ Configuration

### Environment Variables

Before configuring the application, you'll need to set up your environment variables. See the [Environment Variables Documentation](docs/configuration/environment_variables.md) for a complete reference.

Create a `.env` file in your project root with the required API keys:

```env
# Required API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Local LLM with Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

### Job Search Configuration

Configure job search parameters in `config.yaml`:

```yaml
jobspy:
  search_term: "Python Developer"
  location: "Remote"
  site_name: ["linkedin", "indeed"]
  distance: 50  # miles
  is_remote: true
  results_wanted: 10

  # Timeout settings (in seconds)
  timeouts:
    default: 30
    sites:
      linkedin: 45
      indeed: 30
```

### LLM Configuration

MyJobSpyAI uses LangChain to provide a unified interface to multiple LLM providers:

```yaml
llm:
  default_provider: "langchain_default"

  providers:
    langchain_default:
      type: langchain
      enabled: true
      config:
        provider: "openai"  # Options: openai, anthropic, google, ollama, etc.
        model: "gpt-4"
        temperature: 0.7
        max_tokens: 1000

        # Provider-specific configuration
        provider_config:
          openai:
            api_key: ${OPENAI_API_KEY}
          anthropic:
            api_key: ${ANTHROPIC_API_KEY}
          ollama:
            base_url: "http://localhost:11434"
```

### Logging Configuration

```yaml
logging:
  log_dir: "logs"
  log_level: "INFO"

  # File destinations
  files:
    app:
      path: "app.log"
      level: "INFO"
    debug:
      path: "debug.log"
      level: "DEBUG"
    error:
      path: "error.log"
      level: "WARNING"

  # Log rotation
  rolling_strategy: "size"  # or "time"
  max_size: 10485760  # 10MB
  backup_count: 5
```

## 🔍 Job Search Features

MyJobSpyAI provides powerful job search capabilities:

- **Multi-site Search**: Search across multiple job boards simultaneously
- **Advanced Filtering**: Filter by salary, location, job type, and more
- **Resume Analysis**: Parse and analyze your resume for better job matching
- **Customizable Timeouts**: Configure timeouts per job site for optimal performance
- **Error Handling**: Automatic retries and fallback mechanisms

## 🤖 LLM Integration

The LangChain integration provides a unified interface to multiple LLM providers:

### Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude models
- **Google AI**: Gemini models
- **Ollama**: Local LLM models
- **And more**: Support for any LangChain-compatible provider

### Features

- **Unified Interface**: Same API for all providers
- **Chat Model Support**: Native support for chat-based models
- **Streaming**: Real-time token streaming
- **Error Handling**: Built-in retry logic and fallback options
- **Asynchronous Support**: Full async/await support

## 🛠 Development

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MyJobSpyAI.git
   cd MyJobSpyAI
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run tests:
   ```bash
   poetry run pytest
   ```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- **Comprehensive Logging**: Built-in observability with OpenTelemetry
- **Type Annotations**: Improved code reliability and IDE support

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MyJobSpyAI.git
   cd MyJobSpyAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your settings in `config.yaml`

## 📝 Usage

### Basic Usage
```bash
python -m myjobspyai --search "Software Engineer" --location "Remote"
```

## 🧠 Unified LLM Provider System

MyJobSpyAI features a powerful, unified interface for multiple LLM providers through **LangChain**, making it easy to switch between different AI models while maintaining a consistent API.

### ✨ Key Benefits

- **Single Interface**: One consistent API for all LLM providers
- **Broad Compatibility**: Support for 15+ AI providers including OpenAI, Anthropic, Google, and more
- **Simplified Configuration**: Easy setup through YAML or environment variables
- **Advanced Features**: Streaming, async support, and automatic retries
- **Future-Proof**: Easily add new providers as they become available

### 🛠 Supported Providers

| Provider | Example Model | Required Environment Variable |
|----------|---------------|-------------------------------|
| OpenAI | gpt-4-turbo | `OPENAI_API_KEY` |
| Anthropic | claude-3-opus | `ANTHROPIC_API_KEY` |
| Google AI | gemini-pro | `GOOGLE_API_KEY` |
| Ollama | llama2 | `OLLAMA_BASE_URL` |
| LM Studio | Local Models | - |
| OpenRouter | openai/gpt-4 | `OPENROUTER_API_KEY` |
| Together AI | llama-2-70b | `TOGETHER_API_KEY` |
| Fireworks | llama-v2-70b | `FIREWORKS_API_KEY` |
| DeepInfra | llama-2-70b | `DEEPINFRA_API_TOKEN` |
| Perplexity | pplx-70b | `PERPLEXITY_API_KEY` |
| Cohere | command-nightly | `COHERE_API_KEY` |
| HuggingFace | llama-2-70b | `HUGGINGFACEHUB_TOKEN` |
| Vertex AI | gemini-pro | `GOOGLE_APPLICATION_CREDENTIALS` |
| Tongyi | qwen-turbo | `DASHSCOPE_API_KEY` |
| YandexGPT | yandexgpt | `YANDEX_API_KEY` |
| ZhipuAI | chatglm-pro | `ZHIPUAI_API_KEY` |

### ⚙️ Configuration Example

Edit your `config.yaml` to configure the LLM provider:

```yaml
# LLM Configuration
llm:
  # Default provider to use (must match one of the enabled providers below)
  default_provider: "langchain_default"

  # Configure multiple LLM providers
  providers:
    # Unified LangChain provider (recommended)
    langchain_default:
      type: "langchain"
      enabled: true
      config:
        # General settings
        provider: "openai"  # Options: openai, anthropic, google, ollama, etc.
        model: "gpt-4-turbo"  # Model name specific to the provider

        # Generation parameters
        temperature: 0.7    # 0.0 to 2.0, higher = more random
        max_tokens: 1000    # Maximum tokens to generate
        top_p: 1.0          # Nucleus sampling (0.0 to 1.0)

        # Execution settings
        streaming: true     # Stream responses as they're generated
        timeout: 60         # Request timeout in seconds
        max_retries: 3      # Number of retries for failed requests

        # Provider configurations (only configure what you need)
        provider_config:
          # OpenAI / Azure OpenAI
          openai:
            api_key: ${OPENAI_API_KEY}  # Set via environment variable
            organization: ${OPENAI_ORG_ID:}  # Optional
            # base_url: "https://api.openai.com/v1"  # For Azure or custom endpoints

          # Anthropic (Claude)
          anthropic:
            api_key: ${ANTHROPIC_API_KEY}
            max_tokens_to_sample: 1000

          # Google AI (Gemini)
          google:
            api_key: ${GOOGLE_API_KEY}
            model_name: "gemini-pro"

          # Ollama (Local Models)
          ollama:
            base_url: "http://localhost:11434"
            # model: "llama2"  # Override model for this provider
```

### 🔑 Environment Variables

For security, always use environment variables for API keys and sensitive data:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# Google AI
GOOGLE_API_KEY=your-google-key

# For local models (Ollama/LM Studio)
OLLAMA_BASE_URL=http://localhost:11434  # Default

# Other providers
TOGETHER_API_KEY=your-together-key
FIREWORKS_API_KEY=your-fireworks-key
PERPLEXITY_API_KEY=your-perplexity-key
COHERE_API_KEY=your-cohere-key
```

### 📚 Documentation

For more information, please refer to:
- [LangChain Integration](docs/features/langchain_integration.md) - Comprehensive guide to LangChain integration
- [Migration Guide](docs/migrations/README.md) - How to migrate from previous versions
- [Configuration Reference](config.example.yaml) - Complete configuration reference with all options

### Environment Variables

For security, always use environment variables for API keys and sensitive configuration:

```bash
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-google-key"
# etc.
```

### Advanced Options
```bash
python -m myjobspyai \
  --search "Machine Learning Engineer" \
  --location "San Francisco, CA" \
  --max-results 50 \
  --min-salary 120000 \
  --job-type fulltime \
  --remote-only
```

## 🛠 Configuration

Edit `config.yaml` to customize:
- Search parameters
- Logging and output preferences
- Caching behavior

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

For detailed documentation, please see the [documentation](docs/README.md) directory.

## 📊 Project Status

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/MyJobSpyAI/graphs/commit-activity)
