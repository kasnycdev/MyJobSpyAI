"""Enhanced resume analysis models for MyJobSpy AI."""

from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, field_validator


class EducationLevel(str, Enum):
    """Enumeration of education levels."""
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    DOCTORATE = "doctorate"
    PROFESSIONAL = "professional"
    OTHER = "other"


class ExperienceLevel(str, Enum):
    """Enumeration of experience levels."""
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


class SkillCategory(str, Enum):
    """Enumeration of skill categories."""
    TECHNICAL = "technical"
    SOFT = "soft"
    TOOL = "tool"
    LANGUAGE = "language"
    CERTIFICATION = "certification"
    OTHER = "other"


class Education(BaseModel):
    """Model representing an education entry."""
    institution: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree or certification obtained")
    field_of_study: str = Field(..., description="Field of study")
    level: EducationLevel = Field(..., description="Education level")
    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date (None if current)")
    gpa: Optional[float] = Field(None, description="GPA if available", ge=0.0, le=4.0)
    description: Optional[str] = Field(None, description="Additional details")


class Experience(BaseModel):
    """Model representing a work experience entry."""
    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job title/position")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date (None if current)")
    current: bool = Field(False, description="Whether this is the current position")
    description: str = Field(..., description="Job description and responsibilities")
    skills_used: List[str] = Field(
        default_factory=list,
        description="List of skills used in this position"
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="Notable achievements in this role"
    )


class Skill(BaseModel):
    """Model representing a skill with proficiency level."""
    name: str = Field(..., description="Name of the skill")
    category: SkillCategory = Field(..., description="Category of the skill")
    proficiency: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Proficiency level from 0 to 1"
    )
    years_experience: Optional[float] = Field(
        None,
        ge=0.0,
        description="Years of experience with this skill"
    )
    last_used: Optional[int] = Field(
        None,
        ge=1900,
        le=2100,
        description="Last year the skill was used"
    )


class ResumeData(BaseModel):
    """Comprehensive resume data model with enhanced fields for analysis."""
    # Personal Information
    full_name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    linkedin_url: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio_url: Optional[HttpUrl] = Field(None, description="Portfolio or website URL")

    # Professional Summary
    summary: Optional[str] = Field(None, description="Professional summary/objective")

    # Core Sections
    education: List[Education] = Field(
        default_factory=list,
        description="List of education entries"
    )
    experience: List[Experience] = Field(
        default_factory=list,
        description="List of work experience entries"
    )
    skills: List[Skill] = Field(
        default_factory=list,
        description="List of skills with proficiency levels"
    )

    # Additional Sections
    certifications: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of certifications with name and issuer"
    )
    languages: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list,
        description="List of languages with proficiency level"
    )
    projects: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of projects with details"
    )

    # Metadata
    last_updated: Optional[date] = Field(
        None,
        description="When the resume was last updated"
    )
    source_file: Optional[str] = Field(
        None,
        description="Path to the original resume file"
    )

    # Validation
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v and '@' not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v: List[Skill]) -> List[Skill]:
        if not v:
            raise ValueError("At least one skill is required")
        return v
