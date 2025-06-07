# Changelog

## [Unreleased]
### Added
- Unified LLM provider system using LangChain as the primary interface
- Support for multiple LLM providers through a single configuration
- Comprehensive documentation for the new provider system
- Migration guide for transitioning from direct providers to LangChain

### Changed
- Removed direct provider implementations (OpenAI, Ollama, Gemini)
- Updated configuration structure to use `langchain_default` provider
- Simplified dependency management by removing direct provider dependencies
- Improved error handling and logging for provider initialization

### Removed
- Direct provider implementations (`OllamaProvider`, `OpenAIProvider`, `GeminiProvider`)
- Direct provider configurations from `config.yaml`
- Unused provider dependencies

## [Previous Versions]

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-06-06

### Added
- Configurable timeouts for job site scraping
- Site-specific timeout configuration in `config.yaml`
- Documentation for timeout configuration in `docs/features/timeout_configuration.md`
- Better error handling for site timeouts

### Changed
- Updated JobSpy scraper to respect configured timeouts
- Improved error messages for timeout scenarios
- Default timeout increased from 10s to 30s for better reliability
- Naukri site set to lower timeout (10s) due to reliability issues

### Fixed
- Fixed YAML configuration structure issues
- Resolved missing imports in jobspy_scraper.py
- Fixed cache directory creation for resume data

## [0.3.0] - 2025-06-06

## [0.3.0] - 2025-06-06

### Added
- Centralized error handling utilities
- Common test utilities in tests/conftest.py
- Comprehensive code cleanup and refactoring
- New README with improved documentation
- Badges for Python version, license, and code style

### Changed
- Consolidated duplicate analyzer implementations
- Removed deprecated code and unused imports
- Centralized configuration loading
- Consolidated logging utilities
- Unified LLM provider implementations
- Updated dependencies and removed unused packages
- Improved error messages and logging
- Optimized job processing pipeline

### Removed
- Duplicate analyzer.py in favor of analyzer_new.py
- Redundant LLM provider implementations
- Unused logging utilities
- Deprecated code marked for removal
- Unused test fixtures and helpers

## [0.2.0] - 2025-06-05

### Added
- Initial project structure and setup
- Core functionality for job search and analysis
- Support for multiple LLM providers (Ollama, OpenAI, Gemini)
- Configuration management system
- Basic command-line interface

### Changed
- Updated project dependencies to latest stable versions
- Improved error handling and logging
- Enhanced documentation and code organization

### Added
- Initial project structure and setup
- Core functionality for job search and analysis
- Support for multiple LLM providers (Ollama, OpenAI, Gemini)
- Configuration management system
- Basic command-line interface

### Changed
- Updated project dependencies to latest stable versions
- Improved error handling and logging
- Enhanced documentation and code organization

## [0.1.0] - 2025-06-04

### Added
- Initial release of MyJobSpy AI
- Basic job search and analysis functionality
- Support for local LLM models via Ollama
- Configuration via YAML file
- Basic filtering and ranking of job results
