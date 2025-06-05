# Project Management Scripts

This directory contains scripts to help manage the GitHub project board and track progress.

## Setup

1. Install the GitHub CLI: https://cli.github.com/
2. Authenticate with GitHub:
   ```bash
   gh auth login
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Scripts

### `board_manager.py`

A Python script to manage GitHub project boards using the GitHub CLI.

**Usage:**
```bash
python board_manager.py
```

### `create_issues.py`

Create GitHub issues from tasks in the project board markdown file.

**Usage:**
```bash
python create_issues.py
```

## Project Board Structure

The project board is defined in `.project/boards/2024-06-05_resume_job_suitability_enhancement.md`.

## GitHub Workflow

The `.github/workflows/project-sync.yml` workflow automates project board updates when issues are created or updated.

## Environment Variables

- `GH_TOKEN`: GitHub personal access token with `repo` scope

## License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.
