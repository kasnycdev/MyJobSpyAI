#!/usr/bin/env python3
"""
Create GitHub issues from project board tasks.

This script reads tasks from the project board markdown file and creates
corresponding GitHub issues.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Project configuration
REPO = "kasnycdev/MyJobSpyAI"
PROJECT_BOARD_FILE = Path("../../.project/boards/2024-06-05_resume_job_suitability_enhancement.md")
ISSUE_LABELS = ["enhancement", "project-board"]

class IssueCreator:
    """Helper class to create GitHub issues."""

    def __init__(self, repo: str, labels: Optional[List[str]] = None):
        """Initialize with repository and labels."""
        self.repo = repo
        self.labels = labels or []

    def create_issue(self, title: str, body: str, labels: Optional[List[str]] = None) -> Dict:
        """Create a GitHub issue."""
        issue_labels = self.labels + (labels or [])

        cmd = [
            "gh", "issue", "create",
            "--repo", self.repo,
            "--title", title,
            "--body", body,
        ]

        for label in issue_labels:
            cmd.extend(["--label", label])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Created issue: {result.stdout.strip()}")
            return {"url": result.stdout.strip()}
        except subprocess.CalledProcessError as e:
            print(f"Error creating issue: {e.stderr}")
            raise


def parse_tasks_from_markdown(file_path: Path) -> List[Dict[str, str]]:
    """Parse tasks from markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Project board file not found: {file_path}")

    tasks = []
    current_section = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Check for section headers
            section_match = re.match(r'^##\s+(.+)$', line)
            if section_match:
                current_section = section_match.group(1).strip()
                continue

            # Check for task items
            task_match = re.match(r'^-\s+\[([ xX])\]\s+(.+)$', line)
            if task_match and current_section:
                status, description = task_match.groups()
                tasks.append({
                    "section": current_section,
                    "description": description,
                    "completed": status.lower() == 'x',
                })

    return tasks


def main():
    """Main function to create issues from tasks."""
    try:
        # Parse tasks from markdown
        tasks = parse_tasks_from_markdown(PROJECT_BOARD_FILE)
        print(f"Found {len(tasks)} tasks in project board.")

        # Filter out completed tasks
        pending_tasks = [t for t in tasks if not t["completed"]]
        print(f"Found {len(pending_tasks)} pending tasks.")

        # Initialize issue creator
        creator = IssueCreator(REPO, ISSUE_LABELS)

        # Create issues for pending tasks
        for task in pending_tasks:
            title = f"{task['section']}: {task['description']}"
            body = f"## {task['section']}\n\n{task['description']}\n\n*This task was automatically created from the project board.*"

            print(f"\nCreating issue: {title}")
            try:
                creator.create_issue(title, body)
            except Exception as e:
                print(f"Failed to create issue: {e}")
                continue

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
