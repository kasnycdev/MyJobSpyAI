#!/usr/bin/env python3
"""
GitHub Project Board Manager

This script helps manage GitHub project boards using the GitHub CLI.
It provides utilities to create, update, and manage project board items.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project configuration
PROJECT_URL = "https://github.com/users/kasnycdev/projects/1"
PROJECT_ID = "1"  # This might need to be updated based on your project

class ProjectBoardManager:
    """Manager for GitHub project board operations."""

    def __init__(self, token: Optional[str] = None):
        """Initialize with optional GitHub token."""
        self.token = token or os.environ.get("GH_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not provided. Set GH_TOKEN environment variable.")

    def _run_gh_command(self, command: str, args: List[str]) -> Dict:
        """Run a GitHub CLI command and return the JSON output."""
        cmd = ["gh", command] + args + ["--json", "id,title,url"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            raise

    def get_project_info(self) -> Dict:
        """Get information about the project."""
        return self._run_gh_command("project", ["view", PROJECT_ID])

    def list_columns(self) -> List[Dict]:
        """List all columns in the project."""
        result = self._run_gh_command("project", ["column-list", PROJECT_ID])
        return result.get("columns", [])

    def add_issue_to_project(self, issue_url: str, column_name: str = "To do") -> Dict:
        """Add an issue to the project board in the specified column."""
        # First, list all columns to find the column ID
        columns = self.list_columns()
        column_id = None

        for col in columns:
            if col.get("name", "").lower() == column_name.lower():
                column_id = col.get("id")
                break

        if not column_id:
            raise ValueError(f"Column '{column_name}' not found in project.")

        # Add the issue to the column
        return self._run_gh_command("project", [
            "item-add",
            PROJECT_ID,
            issue_url,
            "--column", column_id
        ])

    def move_item_to_column(self, item_id: str, column_name: str) -> Dict:
        """Move an item to a different column."""
        columns = self.list_columns()
        column_id = None

        for col in columns:
            if col.get("name", "").lower() == column_name.lower():
                column_id = col.get("id")
                break

        if not column_id:
            raise ValueError(f"Column '{column_name}' not found in project.")

        return self._run_gh_command("project", [
            "item-edit",
            item_id,
            "--column", column_id
        ])


def main():
    """Main function to demonstrate usage."""
    try:
        manager = ProjectBoardManager()

        # Example: Get project info
        print("Project Info:")
        print(json.dumps(manager.get_project_info(), indent=2))

        # Example: List columns
        print("\nColumns:")
        columns = manager.list_columns()
        for col in columns:
            print(f"- {col.get('name')} (ID: {col.get('id')})")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
