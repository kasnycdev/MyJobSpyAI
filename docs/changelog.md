`changelog`{.interpreted-text role="ref"} `changelog`{.interpreted-text
role="ref"} `changelog`{.interpreted-text role="ref"}
`changelog`{.interpreted-text role="ref"} `changelog`{.interpreted-text
role="ref"} `changelog`{.interpreted-text role="ref"}

# Changelog

## [Unreleased]

### Added

- New `JobService` class for handling job search and analysis
- New `ResumeService` class for resume parsing and analysis
- Display utilities for rich console output of job listings and resume analysis
- Comprehensive error handling and retry mechanisms
- Support for multiple output formats (JSON, CSV, XLSX, Markdown)
- Progress indicators for long-running operations

### Changed

- Moved business logic from `main.py` to dedicated service classes
- Refactored job search functionality into reusable components
- Improved error handling with graceful degradation
- Enhanced logging throughout the service layer
- Updated documentation to reflect new service architecture
- Improved type hints and code organization

### Fixed

- Issues with job search result handling
- Circular imports in service layer
- Type hinting issues
- Improved error messages for better debugging

---

All notable changes to MyJobSpyAI will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

-   Comprehensive documentation for configuration, usage, and examples
-   Detailed contribution guidelines and development setup instructions
-   Enhanced logging system with multiple log files and rotation
-   Support for Ollama LLM provider
-   Job analysis and resume matching functionality
-   Configuration management system with environment variable support

\### Changed - Updated project structure for better organization -
Improved error handling and logging - Refactored configuration loading
to support multiple formats - Enhanced documentation with detailed
examples

\### Fixed - Fixed issues with LLM provider initialization - Resolved
configuration loading from multiple paths - Addressed logging
configuration issues - Fixed documentation build process

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

#### ===================

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

-   Initial project setup and basic structure
-   Core functionality for job and resume analysis
-   Basic configuration system
-   Initial documentation

\### Changed - N/A

\### Fixed - N/A

\## Versioning

This project uses semantic versioning. Given a version number
MAJOR.MINOR.PATCH:

-   **MAJOR** version for incompatible API changes
-   **MINOR** version for added functionality in a backward-compatible
    manner
-   **PATCH** version for backward-compatible bug fixes

\## Deprecation Policy

-   Deprecated features will be marked with a deprecation warning in the
    documentation
-   Features will be removed after being deprecated for at least one
    minor version
-   Breaking changes will only be introduced in major versions

\## Security

For security-related issues, please email <security@example.com> instead
of using the public issue tracker.
