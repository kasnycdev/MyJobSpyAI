from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class EducationLevel(str, Enum):
    """Enumeration of education levels."""

    HIGHSCHOOL = "highschool"
    ASSOCIATE = "associate"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    DOCTORATE = "doctorate"
    PROFESSIONAL = "professional"
    CERTIFICATION = "certification"
    DIPLOMA = "diploma"


class ExperienceLevel(str, Enum):
    """Enumeration of experience levels."""

    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class SkillCategory(str, Enum):
    """Enumeration of skill categories."""

    PROGRAMMING = "programming"
    FRAMEWORK = "framework"
    TOOL = "tool"
    LANGUAGE = "language"
    DATABASE = "database"
    CLOUD = "cloud"
    DEVOPS = "devops"
    TESTING = "testing"
    DESIGN = "design"
    SOFT_SKILL = "soft_skill"
    BUSINESS = "business"
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

    def to_dict(self) -> dict[str, Any]:
        """Convert education entry to dictionary with proper serialization.

        Returns:
            Dictionary containing the education data with proper serialization
            of dates and enums.
        """
        data = self.model_dump(exclude_none=True)
        # Convert dates to ISO format strings
        for date_field in ['start_date', 'end_date']:
            if date_field in data and data[date_field] is not None:
                data[date_field] = data[date_field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Education':
        """Create an Education instance from a dictionary.

        Args:
            data: Dictionary containing education data

        Returns:
            Education instance
        """
        return cls(**data)

    @model_validator(mode='after')
    def validate_education_dates(self) -> 'Education':
        """Validate that end date is after start date if both are provided."""
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        return self


class Experience(BaseModel):
    """Model representing a work experience entry.

    Attributes:
        company: Company name
        position: Job title/position
        location: Job location (optional)
        start_date: Start date of employment
        end_date: End date (None if current)
        current: Whether this is the current position
        description: Job description and responsibilities
        skills_used: List of skills used in this position
        achievements: Notable achievements in this role
    """

    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job title/position")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[date] = Field(None, description="Start date of employment")
    end_date: Optional[date] = Field(
        None, description="End date of employment (None if current)"
    )
    current: bool = Field(False, description="Whether this is the current position")
    description: str = Field(..., description="Job description and responsibilities")
    skills_used: list[str] = Field(
        default_factory=list, description="List of skills used in this position"
    )
    achievements: list[str] = Field(
        default_factory=list, description="Notable achievements in this role"
    )
    experience_level: Optional[ExperienceLevel] = Field(
        None, description="Experience level for this position"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert experience entry to dictionary with proper serialization.

        Returns:
            Dictionary containing the experience data with proper serialization
            of dates and enums.
        """
        data = self.model_dump(exclude_none=True)
        # Convert dates to ISO format strings
        for date_field in ['start_date', 'end_date']:
            if date_field in data and data[date_field] is not None:
                data[date_field] = data[date_field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Experience':
        """Create an Experience instance from a dictionary.

        Args:
            data: Dictionary containing experience data

        Returns:
            Experience instance
        """
        return cls(**data)

    @model_validator(mode='after')
    def validate_experience_dates(self) -> 'Experience':
        """Validate that end date is after start date if both are provided."""
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        return self

    def get_duration_years(self, end_date: Optional[date] = None) -> float:
        """Calculate duration of experience in years.

        Args:
            end_date: Optional end date to use (defaults to current date or end_date)

        Returns:
            Duration in years, rounded to 1 decimal place
        """
        if not self.start_date:
            return 0.0

        end = end_date or self.end_date or date.today()
        if not end:
            return 0.0

        delta = end - self.start_date
        return round(delta.days / 365.25, 1)


class Skill(BaseModel):
    """Model representing a skill with proficiency level.

    Attributes:
        name: Name of the skill
        category: Category of the skill from SkillCategory
        proficiency: Proficiency level from 0 to 1
        years_experience: Years of experience with this skill
        last_used: Last year the skill was used (YYYY)
    """

    name: str = Field(..., description="Name of the skill")
    category: SkillCategory = Field(..., description="Category of the skill")
    proficiency: float = Field(
        0.0, ge=0.0, le=1.0, description="Proficiency level from 0 to 1"
    )
    years_experience: Optional[float] = Field(
        None, ge=0.0, description="Years of experience with this skill"
    )
    last_used: Optional[int] = Field(
        None, ge=1900, le=2100, description="Last year the skill was used"
    )


class ResumeData(BaseModel):
    """Comprehensive resume data model with enhanced fields for analysis.

    This model represents a complete resume with personal information, education,
    work experience, skills, and additional sections. It includes validation and
    helper methods for common operations.

    Attributes:
        full_name: Full name of the candidate
        email: Email address
        phone: Phone number
        location: Current location
        linkedin_url: LinkedIn profile URL
        portfolio_url: Portfolio or website URL
        summary: Professional summary/objective
        education: List of education entries
        experience: List of work experience entries
        skills: List of skills with proficiency levels
        certifications: List of certifications with name and issuer
        languages: List of languages with proficiency level
        projects: List of projects with details
        last_updated: When the resume was last updated
        source_file: Path to the original resume file
    """

    # Personal Information
    full_name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    linkedin_url: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio_url: Optional[HttpUrl] = Field(
        None, description="Portfolio or website URL"
    )

    # Professional Summary
    summary: Optional[str] = Field(None, description="Professional summary/objective")

    # Core Sections
    education: list[Education] = Field(
        default_factory=list, description="List of education entries"
    )
    experience: list[Experience] = Field(
        default_factory=list, description="List of work experience entries"
    )
    skills: list[Skill] = Field(
        default_factory=list, description="List of skills with proficiency levels"
    )

    # Additional Sections
    certifications: list[dict[str, str]] = Field(
        default_factory=list, description="List of certifications with name and issuer"
    )
    languages: list[dict[str, str | float]] = Field(
        default_factory=list, description="List of languages with proficiency level"
    )
    projects: list[dict[str, str]] = Field(
        default_factory=list, description="List of projects with details"
    )

    # Metadata
    last_updated: Optional[date] = Field(
        None, description="When the resume was last updated"
    )
    source_file: Optional[str] = Field(
        None, description="Path to the original resume file"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert resume to dictionary with proper serialization.

        Returns:
            Dictionary containing all resume data with proper serialization
            of dates, URLs, and nested models.
        """
        data = self.model_dump(exclude_none=True)

        # Handle URL serialization
        for url_field in ['linkedin_url', 'portfolio_url']:
            if url_field in data and data[url_field] is not None:
                data[url_field] = str(data[url_field])

        # Handle date serialization
        if 'last_updated' in data and data['last_updated'] is not None:
            data['last_updated'] = data['last_updated'].isoformat()

        # Handle nested models
        if 'education' in data:
            data['education'] = [edu.to_dict() for edu in self.education]

        if 'experience' in data:
            data['experience'] = [exp.to_dict() for exp in self.experience]

        if 'skills' in data:
            data['skills'] = [skill.to_dict() for skill in self.skills]

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ResumeData':
        """Create a ResumeData instance from a dictionary.

        Args:
            data: Dictionary containing resume data

        Returns:
            ResumeData instance
        """
        # Handle nested models
        if 'education' in data and isinstance(data['education'], list):
            data['education'] = [
                Education.from_dict(edu) if isinstance(edu, dict) else edu
                for edu in data['education']
            ]

        if 'experience' in data and isinstance(data['experience'], list):
            data['experience'] = [
                Experience.from_dict(exp) if isinstance(exp, dict) else exp
                for exp in data['experience']
            ]

        if 'skills' in data and isinstance(data['skills'], list):
            data['skills'] = [
                Skill.from_dict(skill) if isinstance(skill, dict) else skill
                for skill in data['skills']
            ]

        return cls(**data)

    def get_total_experience_years(self) -> float:
        """Calculate total years of work experience.

        Returns:
            Total years of experience across all positions, rounded to 1 decimal place
        """
        if not self.experience:
            return 0.0

        total_days = 0
        for exp in self.experience:
            if exp.start_date:
                end_date = exp.end_date or date.today()
                if end_date and end_date >= exp.start_date:
                    total_days += (end_date - exp.start_date).days

        return round(total_days / 365.25, 1)

    def get_skills_by_category(self, category: SkillCategory) -> list['Skill']:
        """Get all skills in a specific category.

        Args:
            category: SkillCategory to filter by

        Returns:
            List of skills in the specified category
        """
        return [skill for skill in self.skills if skill.category == category]

    def get_highest_education(self) -> Optional['Education']:
        """Get the highest level of education from the resume.

        Returns:
            The highest Education instance, or None if no education entries exist.
        """
        if not self.education:
            return None

        # Define priority of education levels
        level_priority = {
            EducationLevel.DOCTORATE: 6,
            EducationLevel.PROFESSIONAL: 5,
            EducationLevel.MASTERS: 4,
            EducationLevel.BACHELORS: 3,
            EducationLevel.ASSOCIATE: 2,
            EducationLevel.HIGHSCHOOL: 1,
            EducationLevel.CERTIFICATION: 0,
            EducationLevel.DIPLOMA: 0,
        }

        return max(
            self.education,
            key=lambda x: level_priority.get(x.level, 0),
        )

    # Validation methods
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if v and '@' not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator('skills')
    @classmethod
    def validate_skills(cls, v: list['Skill']) -> list['Skill']:
        """Validate that at least one skill is provided."""
        if not v:
            raise ValueError("At least one skill is required")
        return v

    @field_validator('experience')
    @classmethod
    def validate_experience_dates(cls, v: list[Experience]) -> list[Experience]:
        """Validate that experience entries have valid date ranges."""
        for exp in v:
            if exp.start_date and exp.end_date and exp.end_date < exp.start_date:
                raise ValueError(
                    f"Experience end date ({exp.end_date}) cannot be before "
                    f"start date ({exp.start_date})"
                )
        return v
