"""Job-related models for MyJobSpy AI."""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class JobSource(str, Enum):
    """Enumeration of possible job sources."""

    OTHER = "other"
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    JOBSPY = "jobspy"
    CUSTOM = "custom"


class JobType(str, Enum):
    """Enumeration of job types."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"
    OTHER = "other"


class JobMatch(BaseModel):
    """Model representing a job match analysis."""

    score: float = Field(..., ge=0.0, le=100.0, description="Match score (0-100)")
    skills_match: Dict[str, float] = Field(
        {},
        description="Matching skills with confidence scores",
    )
    missing_skills: List[str] = Field(
        [],
        description="List of skills mentioned in the job but not in the resume",
    )
    extra_skills: List[str] = Field(
        [],
        description="List of skills in the resume but not required by the job",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes about the match",
    )


class JobAnalysis(BaseModel):
    """Model representing the analysis of a job posting."""

    summary: str = Field(
        ...,
        description="Brief summary of the job posting",
    )
    required_skills: List[str] = Field(
        [],
        description="List of required skills",
    )
    preferred_skills: List[str] = Field(
        [],
        description="List of preferred skills",
    )
    experience_level: Optional[str] = Field(
        None,
        description="Required experience level",
    )
    education_requirements: List[str] = Field(
        [],
        description="List of education requirements",
    )
    salary_range: Optional[str] = Field(
        None,
        description="Salary range if mentioned",
    )
    is_remote: Optional[bool] = Field(
        None,
        description="Whether the job is remote",
    )
    location: Optional[str] = Field(
        None,
        description="Job location if not remote",
    )
    company_culture: List[str] = Field(
        [],
        description="Notable aspects of company culture",
    )


class Job(BaseModel):
    """Model representing a job posting."""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
        use_enum_values = True

    # Core fields
    id: str = Field(
        ...,
        description="Unique identifier for the job",
    )
    title: str = Field(
        ...,
        description="Job title",
    )
    company: str = Field(
        ...,
        description="Company name",
    )
    location: str = Field(
        ...,
        description="Job location",
    )
    description: str = Field(
        ...,
        description="Full job description",
    )
    url: Optional[HttpUrl] = Field(
        None,
        description="URL to the job posting",
    )
    job_type: Optional[JobType] = Field(
        None,
        description="Type of job (full-time, part-time, etc.)",
    )
    remote: bool = Field(
        False,
        description="Whether the job is remote",
    )
    salary: Optional[str] = Field(
        None,
        description="Salary information if available",
    )
    metadata: Dict[str, Any] = Field(
        {},
        description="Additional metadata about the job",
    )

    # Metadata
    source: JobSource = Field(
        JobSource.OTHER,
        description="Source of the job posting",
    )
    posted_date: Optional[datetime] = Field(
        None,
        description="When the job was posted",
    )
    application_deadline: Optional[datetime] = Field(
        None,
        description="Application deadline if specified",
    )

    # Analysis
    raw_data: Dict[str, Any] = Field(
        {},
        description="Raw data from the source",
    )
    analysis: Optional[JobAnalysis] = Field(
        None,
        description="Analysis of the job posting",
    )
    match: Optional[JobMatch] = Field(
        None,
        description="Matching information against a resume",
    )

    @field_validator("url", mode='before')
    @classmethod
    def validate_url(cls, v: Union[str, HttpUrl, None]) -> Optional[HttpUrl]:
        """Validate and convert URL strings to HttpUrl objects."""
        if v is None or isinstance(v, HttpUrl):
            return v
        return HttpUrl(v) if v else None
