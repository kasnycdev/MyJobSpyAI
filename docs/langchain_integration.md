# LangChain Integration

This module provides integration with LangChain for advanced resume analysis and candidate matching using LLMs.

## Features

- **Resume Analysis**: Extract structured data from resumes including skills, experience, and education
- **Candidate Matching**: Evaluate how well a candidate matches a job description
- **Semantic Search**: Find candidates based on semantic similarity to job requirements
- **Structured Outputs**: Type-safe data models using Pydantic
- **Configurable**: Fine-tune behavior through YAML configuration

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements-langchain.txt
   ```

2. Set up your Ollama server (or configure another LLM provider in `config/langchain_config.yaml`)

## Usage

### Basic Example

```python
from myjobspyai.analysis.components.factory import get_analyzer, get_matcher
import asyncio

async def main():
    # Get an analyzer instance
    analyzer = get_analyzer()
    
    # Analyze a resume
    resume_text = """..."""  # Your resume text here
    analysis = await analyzer.analyze(resume_text)
    
    print(f"Skills: {analysis.skills}")
    print(f"Experience: {[exp.title for exp in analysis.experience]}")
    
    # Match a candidate to a job
    matcher = get_matcher()
    job_description = """..."""  # Your job description here
    
    match_result = await matcher.match(
        job_description=job_description,
        candidate_profile=resume_text,
        required_skills=["Python", "AWS"]
    )
    
    print(f"Match Score: {match_result.overall_score}/10")

asyncio.run(main())
```

### Running the Demo

1. Start your Ollama server:
   ```bash
   ollama serve
   ```

2. Run the demo script:
   ```bash
   python examples/langchain_demo.py
   ```

## Configuration

Edit `config/langchain_config.yaml` to customize the behavior:

```yaml
langchain:
  model:
    name: "llama3"  # Model to use
    temperature: 0.1  # Lower for more deterministic outputs
    
  analyzer:
    chunk_size: 4000  # For processing large resumes
    
  matching:
    skill_weight: 0.5  # Weight for skill matching
    experience_weight: 0.3
    education_weight: 0.2
    
  provider:
    type: "ollama"  # or "openai"
    base_url: "http://localhost:11434"
```

## API Reference

### ResumeAnalyzer

Analyzes resumes and extracts structured data.

```python
class ResumeAnalyzer:
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1):
        ...
    
    async def analyze(self, resume_text: str) -> ResumeAnalysis:
        """Analyze a resume and return structured data."""
        ...
```

### CandidateMatcher

Matches candidates to job descriptions.

```python
class CandidateMatcher:
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1):
        ...
    
    async def match(
        self,
        job_description: str,
        candidate_profile: str,
        required_skills: Optional[List[str]] = None
    ) -> MatchScore:
        """Match a candidate to a job description."""
        ...
```

## Testing

Run the unit tests:

```bash
pytest tests/unit/test_langchain_integration.py -v
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
