"""Job-related models for MyJobSpy AI."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, HttpUrl, field_validator

from .base import BaseModel, TimestampMixin


class JobSource(str, Enum):
    """Enum for job sources."""

    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    JOBSPY = "jobspy"
    CUSTOM = "custom"

    def __str__(self):
        return self.value


class JobType(str, Enum):
    """Enum for job types."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    REMOTE = "remote"
    VOLUNTEER = "volunteer"
    HYBRID = "hybrid"
    OTHER = "other"

    def __str__(self):
        return self.value


class JobMatch(BaseModel):
    """Model representing a job match analysis."""

    score: float = Field(..., ge=0.0, le=100.0, description="Match score (0-100)")
    skills_match: dict[str, float] = Field(
        {}, description="Matching skills with confidence scores"
    )
    missing_skills: list[str] = Field(
        [], description="List of skills mentioned in the job but not in the resume"
    )
    extra_skills: list[str] = Field(
        [], description="List of skills in the resume but not required by the job"
    )
    notes: str | None = Field(None, description="Additional notes about the match")


class JobAnalysis(BaseModel):
    """Model representing the analysis of a job posting."""

    summary: str = Field(..., description="Brief summary of the job posting")
    required_skills: list[str] = Field([], description="List of required skills")
    preferred_skills: list[str] = Field([], description="List of preferred skills")
    experience_level: str | None = Field(None, description="Required experience level")
    education_requirements: list[str] = Field(
        [], description="List of education requirements"
    )
    salary_range: str | None = Field(None, description="Salary range if mentioned")
    is_remote: bool | None = Field(None, description="Whether the job is remote")
    location: str | None = Field(None, description="Job location if not remote")
    company_culture: list[str] = Field(
        [], description="Notable aspects of company culture"
    )


class Job(BaseModel, TimestampMixin):
    """Model representing a job posting."""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
        use_enum_values = True
        extra = "ignore"

    # Core fields
    id: str = Field(..., description="Unique identifier for the job")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    description: str = Field(..., description="Full job description")
    url: HttpUrl | None = Field(None, description="URL to the job posting")
    job_type: JobType | None = Field(
        None, description="Type of job (full-time, part-time, etc.)"
    )
    remote: bool = Field(False, description="Whether the job is remote")
    salary: str | None = Field(None, description="Salary information if available")
    metadata: dict[str, Any] = Field(
        {}, description="Additional metadata about the job"
    )

    # Metadata
    source: JobSource = Field(JobSource.CUSTOM, description="Source of the job posting")
    posted_date: datetime | None = Field(None, description="When the job was posted")
    application_deadline: datetime | None = Field(
        None, description="Application deadline if specified"
    )

    # Analysis
    raw_data: dict[str, Any] = Field({}, description="Raw data from the source")
    analysis: JobAnalysis | None = Field(
        None, description="Analysis of the job posting"
    )
    match: JobMatch | None = Field(
        None, description="Matching information against a resume"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary with proper serialization.

        Returns:
            dict[str, Any]: A dictionary representation of the job with proper
            serialization of dates, URLs, and nested models.
        """
        data = super().to_dict()
        # Handle URL serialization
        if self.url is not None:
            data["url"] = str(self.url)
        # Handle date serialization
        posted_date = getattr(self, 'posted_date', None)
        if isinstance(posted_date, datetime):
            data["posted_date"] = posted_date.isoformat()
        deadline = getattr(self, 'application_deadline', None)
        if isinstance(deadline, datetime):
            data["application_deadline"] = deadline.isoformat()
        # Handle nested model serialization
        analysis = getattr(self, 'analysis', None)
        if analysis is not None:
            if hasattr(analysis, 'model_dump'):
                data["analysis"] = analysis.model_dump()
            elif hasattr(analysis, 'dict'):
                data["analysis"] = analysis.dict()
        match = getattr(self, 'match', None)
        if match is not None:
            if hasattr(match, 'model_dump'):
                data["match"] = match.model_dump()
            elif hasattr(match, 'dict'):
                data["match"] = match.dict()
        return data

    @field_validator("url", mode='before')
    @classmethod
    def validate_url(cls, v: str | HttpUrl | None) -> HttpUrl | None:
        """Validate and convert URL strings to HttpUrl objects."""
        if v is None or isinstance(v, HttpUrl):
            return v
        return HttpUrl(v) if v else None
