"""
Display utilities for MyJobSpyAI.

This module provides utilities for displaying job listings and other data
in a user-friendly format.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table


def display_jobs_table(
    jobs: List[Dict[str, Any]],
    console: Console,
    show_details: bool = False,
    title: str = "Job Listings"
) -> None:
    """Display job listings in a formatted table.

    Args:
        jobs: List of job dictionaries
        console: Rich console instance for output
        show_details: Whether to show detailed columns
        title: Table title
    """
    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    # Create table with appropriate columns
    table = Table(title=title, show_lines=True)

    # Add basic columns
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan")
    table.add_column("Company", style="green")
    table.add_column("Location", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Posted", style="purple")

    # Add detailed columns if requested
    if show_details:
        table.add_column("Salary", style="magenta")
        table.add_column("Match", style="green")
        table.add_column("Strengths", style="green")
        table.add_column("Areas to Improve", style="yellow")

    # Add rows
    for i, job in enumerate(jobs, 1):
        # Format basic info
        title = job.get('title', 'N/A')
        company = job.get('company', 'N/A')
        location = job.get('location', 'N/A')
        job_type = job.get('job_type', 'N/A')

        # Format posted date
        posted = job.get('posted_date', '')
        if posted:
            try:
                posted_dt = datetime.fromisoformat(posted)
                posted = posted_dt.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                posted = str(posted)

        # Start building row
        row = [
            str(i),
            title[:50] + ('...' if len(title) > 50 else ''),
            company[:30] + ('...' if len(company) > 30 else ''),
            location[:20] + ('...' if len(location) > 20 else ''),
            str(job_type).title(),
            posted
        ]

        # Add detailed info if requested
        if show_details:
            # Format salary
            salary = job.get('salary', {})
            if isinstance(salary, dict):
                min_sal = salary.get('min_amount', '')
                max_sal = salary.get('max_amount', '')
                currency = salary.get('currency', '')
                period = salary.get('period', '').lower()

                if min_sal or max_sal:
                    salary_str = f"{min_sal or 'N/A'}-{max_sal or 'N/A'} {currency} {period}"
                else:
                    salary_str = "N/A"
            else:
                salary_str = str(salary)

            # Get analysis if available
            analysis = job.get('_analysis', {})
            match_score = analysis.get('suitability_score', 0) if isinstance(analysis, dict) else 0
            strengths = "; ".join(analysis.get('pros', [])[:2]) if isinstance(analysis, dict) else ""
            improvements = "; ".join(analysis.get('cons', [])[:2]) if isinstance(analysis, dict) else ""

            # Add detailed columns
            row.extend([
                salary_str,
                f"{match_score:.0%}" if match_score else "N/A",
                strengths[:30] + ('...' if len(strengths) > 30 else ''),
                improvements[:30] + ('...' if len(improvements) > 30 else '')
            ])

        table.add_row(*[str(cell) for cell in row])

    # Display the table
    console.print(table)


def display_resume_analysis(analysis: Dict[str, Any], console: Console) -> None:
    """Display resume analysis results in a formatted way.

    Args:
        analysis: Dictionary containing resume analysis results
        console: Rich console instance for output
    """
    if not analysis:
        console.print("[yellow]No analysis results to display.[/yellow]")
        return

    # Display summary
    console.rule("Resume Analysis Summary")

    # Basic info
    if 'name' in analysis:
        console.print(f"[bold]Name:[/bold] {analysis['name']}")
    if 'email' in analysis:
        console.print(f"[bold]Email:[/bold] {analysis['email']}")
    if 'phone' in analysis:
        console.print(f"[bold]Phone:[/bold] {analysis['phone']}")

    # Skills
    if 'skills' in analysis:
        console.print("\n[bold]Skills:[/bold]")
        for category, skills in analysis['skills'].items():
            if skills:
                console.print(f"  [bold]{category.title()}:[/bold] {', '.join(skills[:5])}" +
                             ("..." if len(skills) > 5 else ""))

    # Experience
    if 'experience' in analysis and analysis['experience']:
        console.print("\n[bold]Experience:[/bold]")
        for exp in analysis['experience'][:3]:  # Show most recent 3
            title = exp.get('title', 'N/A')
            company = exp.get('company', 'N/A')
            duration = f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}"
            console.print(f"  • {title} at {company} ({duration})")

    # Education
    if 'education' in analysis and analysis['education']:
        console.print("\n[bold]Education:[/bold]")
        for edu in analysis['education']:
            degree = edu.get('degree', 'N/A')
            institution = edu.get('institution', 'N/A')
            year = edu.get('year', '')
            console.print(f"  • {degree} from {institution}" + (f" ({year})" if year else ""))

    console.rule()
