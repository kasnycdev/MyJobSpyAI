import json
import os
import html
import re
from colorama import Fore, Style  # Import colorama
from rich.console import Console

# Initialize rich console
console = Console()

def load_job_mandates(json_file_path: str) -> list[dict]:
    """
    Loads job mandates from a JSON file.

    Args:
        json_file_path: Path to the JSON file containing a list of job objects.

    Returns:
        A list of job dictionaries, or an empty list on error.
    """
    if not os.path.exists(json_file_path):
        console.log(f"[red]Job mandates file not found: {json_file_path}[/red]")
        return []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            console.log(
                f"[green]Successfully loaded {len(data)} job mandates from {html.escape(json_file_path)}[/green]"
            )
            # Basic validation: ensure items are dictionaries
            if all(isinstance(job, dict) for job in data):
                return data
            else:
                console.log(
                    "[red]JSON file does not contain a list of job objects (dictionaries).[/red]"
                )
                return []
        else:
            console.log("[red]JSON file root is not a list.[/red]")
            return []
    except json.JSONDecodeError as e:
        console.log(
            f"[red]Error decoding JSON file {html.escape(json_file_path)}: {str(e)}[/red]"
        )
        return []
    except Exception as e:
        console.log(
            f"[red]An unexpected error occurred while loading {html.escape(json_file_path)}: {str(e)}[/red]"
        )
        return []

def extract_fields_from_prompt(prompt_path: str) -> list[str]:
    """
    Extracts field names from the JSON schema in the job_extraction.prompt file.

    Args:
        prompt_path: Path to the job_extraction.prompt file.

    Returns:
        A list of field names defined in the JSON schema.
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()

        # Extract JSON schema block
        schema_match = re.search(r'```json\n({.*?})\n```', prompt_content, re.DOTALL)
        if not schema_match:
            console.log("[red]Failed to extract JSON schema from prompt.[/red]")
            return []

        schema_content = schema_match.group(1)
        # Extract field names from the JSON schema
        field_names = re.findall(r'"(.*?)":', schema_content)
        return field_names
    except Exception as e:
        console.log(f"[red]Error extracting fields from prompt: {e}[/red]")
        return []

def parse_job_data(job_data: dict) -> dict:
    """
    Parses raw job data to match the schema defined in `job_extraction.prompt`.

    Args:
        job_data: A dictionary containing raw job data.

    Returns:
        A dictionary conforming to the schema defined in `job_extraction.prompt`.
    """
    return {
        "job_title_extracted": job_data.get("title"),
        "key_responsibilities": job_data.get("responsibilities", []),
        "required_skills": job_data.get("required_skills", []),
        "preferred_skills": job_data.get("preferred_skills", []),
        "required_experience_years": job_data.get("min_experience"),
        "preferred_experience_years": job_data.get("preferred_experience"),
        "required_education": job_data.get("education"),
        "preferred_education": job_data.get("preferred_education"),
        "salary_range_extracted": job_data.get("salary"),
        "work_model_extracted": job_data.get("work_model"),
        "company_culture_hints": job_data.get("culture_hints", []),
        "tools_technologies": job_data.get("tools", []),
        "job_type": job_data.get("job_type"),
        "industry": job_data.get("industry"),
        "company_name": job_data.get("company"),
        "company_size": job_data.get("company_size"),
        "location": job_data.get("location"),
        "job_description": job_data.get("description"),
        "required_certifications": job_data.get("required_certifications", []),
        "preferred_certifications": job_data.get("preferred_certifications", []),
        "security_clearance": job_data.get("security_clearance"),
        "travel_requirements": job_data.get("travel_requirements"),
        "job_id": job_data.get("id"),
        "source": job_data.get("site"),
        "posting_date": job_data.get("date_posted"),
    }

def parse_job_data_dynamic(job_data: dict, fields: list[str]) -> dict:
    """
    Dynamically parses raw job data to match the fields extracted from the prompt schema.

    Args:
        job_data: A dictionary containing raw job data.
        fields: A list of field names to extract from the job data.

    Returns:
        A dictionary containing the extracted fields.
    """
    parsed_data = {}
    for field in fields:
        parsed_data[field] = job_data.get(field)
    return parsed_data

# Example usage
if __name__ == "__main__":
    prompt_path = os.path.join(os.path.dirname(__file__), '../analysis/prompts/job_extraction.prompt')
    fields = extract_fields_from_prompt(prompt_path)
    console.log(f"[green]Extracted fields from prompt: {fields}[/green]")

    # Example job data
    job_data = {
        "title": "Software Engineer",
        "description": "Develop and maintain software applications.",
        "location": "Remote",
        "salary": "$100k - $120k",
        "company": "TechCorp",
    }

    parsed_job = parse_job_data_dynamic(job_data, fields)
    console.log(f"[blue]Parsed job data: {parsed_job}[/blue]")