# Prompts

This directory contains prompt templates used by the MyJobSpy AI application for various natural language processing tasks.

## Structure

```
prompts/
├── job_extraction.prompt    # Prompt for extracting structured job information
├── resume_extraction.prompt  # Prompt for extracting structured resume information
└── suitability_analysis.prompt  # Prompt for analyzing job suitability
```

## Usage

Prompts are loaded using the `load_prompt` utility function from `myjobspyai.utils.files`:

```python
from myjobspyai.utils.files import load_prompt

# Load a prompt template
prompt_template = load_prompt("job_extraction.prompt")

# Format the prompt with variables
formatted_prompt = prompt_template.format(
    job_title="Software Engineer",
    company="Example Inc.",
    # ... other variables
)
```

## Creating New Prompts

1. Create a new `.prompt` file in this directory
2. Use `{variable_name}` syntax for dynamic content
3. Document the required variables in a comment at the top of the file
4. Add the filename to the `package_data` list in `setup.py`

## Best Practices

- Keep prompts focused on a single task
- Include clear instructions and examples
- Use consistent variable naming
- Document required and optional variables
- Test prompts with various inputs to ensure reliability
