import os
import shutil
from pathlib import Path


def cleanup_docs():
    docs_dir = Path("docs")
    backup_dir = docs_dir / "backup_rst"

    # Remove duplicate RST files
    for rst_file in docs_dir.glob("**/*.rst"):
        if not rst_file.name in ['README.rst', 'test.rst']:
            rst_file.unlink()
            print(f"Removed duplicate RST: {rst_file}")

    # Remove backup directory if empty
    if backup_dir.exists() and not any(backup_dir.iterdir()):
        backup_dir.rmdir()
        print("Removed empty backup_rst directory")

    # Create missing directories
    required_dirs = ["development", "enhancement_plans", "features", "getting_started"]

    for dir_name in required_dirs:
        dir_path = docs_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"Created directory: {dir_name}")

    # Create missing files with templates
    missing_files = {
        "development/contributing.md": r"""# Contributing Guide

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Code Style

Follow PEP 8 guidelines with these exceptions:
- Line length: 100 characters
- Use type hints
- Docstrings for all public functions

## Testing

Run tests using:
```bash
pytest
```

## Documentation

Documentation is written in Markdown and built using MkDocs.""",
        "development/code_style.md": r"""# Code Style Guide

## Python

- Follow PEP 8 guidelines
- Use type hints
- Maximum line length: 100 characters
- Use black for formatting
- Use isort for imports

## Documentation

- Use Markdown for documentation
- Follow consistent heading levels
- Use proper code blocks
- Include examples where possible

## Git

- Commit messages: conventional commits
- Branch names: feature/bugfix/[feature-name]
- Pull requests: descriptive titles and descriptions""",
        "enhancement_plans/roadmap.md": r"""# Development Roadmap

## Current Version

- [x] Basic job scraping
- [x] Resume matching
- [x] Job analysis
- [x] API integration

## Next Milestone

- [ ] Advanced filtering
- [ ] Job recommendation system
- [ ] Enhanced analytics
- [ ] Mobile app integration

## Future Features

- [ ] Job market analysis
- [ ] Salary prediction
- [ ] Interview preparation
- [ ] Career path planning""",
        "enhancement_plans/future_features.md": r"""# Future Features

## AI Enhancements

- Advanced natural language processing
- Machine learning for job recommendations
- Automated resume optimization
- Interview question prediction

## Platform Integration

- LinkedIn integration
- Glassdoor integration
- Indeed integration
- Custom job board integration

## Analytics

- Job market trends
- Salary analysis
- Skills demand analysis
- Location-based analytics

## User Experience

- Mobile application
- Enhanced UI/UX
- Personalized dashboard
- Notification system""",
        "features/resume_matching.md": r"""# Resume Matching

## Overview

MyJobSpyAI provides advanced resume matching capabilities to help users find the best job opportunities based on their skills and experience.

## Features

- Automated skill extraction
- Job requirement matching
- Experience level analysis
- Education verification
- Custom matching rules

## Usage

```python
from myjobspyai.matching import ResumeMatcher

matcher = ResumeMatcher()
match_score = matcher.match(resume_text, job_description)
print(f"Match score: {match_score}")
```

## Best Practices

1. Keep resumes up to date
2. Use clear, professional language
3. Include relevant keywords
4. Highlight achievements
5. Customize for each job application""",
        "features/job_scraping.md": r"""# Job Scraping

## Overview

MyJobSpyAI can scrape job listings from multiple sources to provide comprehensive job search capabilities.

## Supported Platforms

- LinkedIn
- Indeed
- Glassdoor
- Custom job boards

## Features

- Automated job collection
- Real-time updates
- Custom filters
- Job de-duplication
- Error handling

## Usage

```python
from myjobspyai.scraping import JobScraper

scraper = JobScraper()
jobs = scraper.scrape(platform="linkedin", query="software engineer")
print(f"Found {len(jobs)} jobs")
```

## Best Practices

1. Use appropriate filters
2. Set reasonable scraping intervals
3. Handle rate limits
4. Monitor for changes
5. Validate job data""",
    }

    for file_path, content in missing_files.items():
        path = docs_dir / file_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            print(f"Created file: {file_path}")

    # Remove Sphinx configuration since we're using MkDocs
    if (docs_dir / "conf.py").exists():
        (docs_dir / "conf.py").unlink()
        print("Removed Sphinx configuration")

    # Remove build directory
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("Removed build directory")


if __name__ == "__main__":
    cleanup_docs()
