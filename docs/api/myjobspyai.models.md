# myjobspyai.models

## Overview

The models module provides data models for job analysis, job listings, resumes, and analysis results. It includes base classes and specific models for different types of data.

## Classes

### Base

Base class for all models.

#### Methods

##### \_validate

```python
# Validate the model data
is_valid = model._validate()
```

Validates the model data against the schema.

##### to_dict

```python
# Convert model to dictionary
model_dict = model.to_dict()
```

Converts the model to a dictionary representation.

##### from_dict

```python
# Create model from dictionary
model = Base.from_dict(data_dict)
```

Creates a model instance from a dictionary.

### Analysis

Model for job analysis results.

#### Fields

- `job_title`: str - The job title
- `required_skills`: List[str] - Required skills
- `preferred_skills`: List[str] - Preferred skills
- `experience_level`: str - Experience level (e.g., "Senior", "Junior")
- `education_requirements`: List[str] - Required education levels
- `salary_range`: Tuple[float, float] - Salary range (min, max)
- `location`: str - Job location
- `company`: str - Company name
- `job_description`: str - Full job description
- `analysis_score`: float - Overall analysis score
- `confidence_score`: float - Confidence in analysis

#### Methods

##### \_calculate_score

```python
# Calculate analysis score internally
score = analysis._calculate_score()
```

Internal method to calculate the analysis score.

##### calculate_score

```python
# Calculate analysis score
score = analysis.calculate_score()
```

Public method to calculate and return the analysis score.

##### to_dict

```python
# Convert analysis to dictionary
analysis_dict = analysis.to_dict()
```

Converts the analysis model to a dictionary.

### Job

Model for job listings.

#### Fields

- `title`: str - Job title
- `company`: str - Company name
- `location`: str - Job location
- `description`: str - Job description
- `posting_date`: datetime - Date of posting
- `salary`: Optional[str] - Salary information
- `source`: str - Source of job posting
- `url`: str - URL to job posting
- `skills`: List[str] - Required skills
- `experience`: int - Years of experience required
- `education`: str - Required education level
- `job_type`: str - Job type (e.g., "Full-time", "Remote")
- `metadata`: Dict[str, Any] - Additional metadata

#### Methods

##### \_extract_skills

```python
# Extract skills from job description
skills = job._extract_skills()
```

Internal method to extract skills from the job description.

##### extract_skills

```python
# Extract skills from job description
skills = job.extract_skills()
```

Public method to extract and return skills from the job description.

##### to_dict

```python
# Convert job to dictionary
job_dict = job.to_dict()
```

Converts the job model to a dictionary.

### Resume

Model for resumes.

#### Fields

- `name`: str - Candidate name
- `email`: str - Contact email
- `phone`: str - Contact phone
- `summary`: str - Professional summary
- `experience`: List[Experience] - Work experience
- `education`: List[Education] - Education history
- `skills`: List[str] - Skills
- `certifications`: List[str] - Certifications
- `projects`: List[Project] - Projects
- `metadata`: Dict[str, Any] - Additional metadata

#### Methods

##### \_extract_skills

```python
# Extract skills from resume
skills = resume._extract_skills()
```

Internal method to extract skills from the resume.

##### extract_skills

```python
# Extract skills from resume
skills = resume.extract_skills()
```

Public method to extract and return skills from the resume.

##### to_dict

```python
# Convert resume to dictionary
resume_dict = resume.to_dict()
```

Converts the resume model to a dictionary.

## Usage Example

```python
from myjobspyai.models import Job, Resume, Analysis

# Create job model
job = Job(
    title="Software Engineer",
    company="TechCorp",
    location="New York",
    description="Senior software engineer position...",
    posting_date=datetime.now(),
    salary="$100,000 - $150,000",
    source="indeed",
    url="https://example.com/job/123"
)

# Create resume model
resume = Resume(
    name="John Doe",
    email="john@example.com",
    phone="555-1234",
    summary="Experienced software engineer...",
    experience=[
        {
            "company": "TechCorp",
            "title": "Senior Engineer",
            "start_date": datetime(2018, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "description": "Led development team..."
        }
    ]
)

# Create analysis model
analysis = Analysis(
    job_title=job.title,
    job_description=job.description,
    resume_data=resume.to_dict(),
    analysis_score=0.85,
    confidence_score=0.90
)

# Convert to dictionaries
job_dict = job.to_dict()
resume_dict = resume.to_dict()
analysis_dict = analysis.to_dict()

# Validate models
is_valid_job = job._validate()
is_valid_resume = resume._validate()
is_valid_analysis = analysis._validate()
```

## Best Practices

1. Always validate model data
2. Use appropriate field types
3. Handle missing data gracefully
4. Use proper error handling
5. Follow type hints
6. Use model conversion methods
7. Validate before processing
8. Handle nested data structures
9. Use metadata for additional info
10. Follow schema validation rules
