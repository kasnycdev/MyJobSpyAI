"""
Data models for resume extraction and parsing.

This module defines Pydantic models for structured resume data.
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import date
from typing_extensions import Literal


class ContactInfo(BaseModel):
    """Contact information from a resume."""
    name: Optional[str] = Field(
        None,
        description="Full name of the candidate"
    )
    email: Optional[str] = Field(
        None,
        description="Primary email address",
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    phone: Optional[str] = Field(
        None,
        description="Primary phone number"
    )
    location: Optional[str] = Field(
        None,
        description="Current location (City, State/Country)"
    )
    linkedin: Optional[str] = Field(
        None,
        description="LinkedIn profile URL"
    )
    portfolio: Optional[str] = Field(
        None,
        description="Personal website or portfolio URL"
    )
    github: Optional[str] = Field(
        None,
        description="GitHub profile URL"
    )


class Education(BaseModel):
    """Education history entry."""
    institution: str = Field(
        ...,
        description="Name of the educational institution"
    )
    degree: Optional[str] = Field(
        None,
        description="Degree obtained (e.g., 'Bachelor of Science')"
    )
    field_of_study: Optional[str] = Field(
        None,
        description="Field of study or major"
    )
    start_date: Optional[Union[date, str]] = Field(
        None,
        description="Start date in YYYY-MM-DD format or just YYYY"
    )
    end_date: Optional[Union[date, str, Literal['Present']]] = Field(
        None,
        description="End date in YYYY-MM-DD format, just YYYY, or 'Present'"
    )
    gpa: Optional[float] = Field(
        None,
        description="Grade point average (e.g., 3.7)",
        ge=0.0,
        le=4.5
    )
    description: Optional[str] = Field(
        None,
        description="Additional details about the education"
    )


class Experience(BaseModel):
    """Work experience entry."""
    company: str = Field(
        ...,
        description="Name of the company or organization"
    )
    position: str = Field(
        ...,
        description="Job title or position held"
    )
    location: Optional[str] = Field(
        None,
        description="Location of the job (City, State/Country)"
    )
    start_date: Optional[Union[date, str]] = Field(
        None,
        description="Start date in YYYY-MM-DD format or just YYYY"
    )
    end_date: Optional[Union[date, str, Literal['Present']]] = Field(
        None,
        description="End date in YYYY-MM-DD format, just YYYY, or 'Present'"
    )
    description: List[str] = Field(
        default_factory=list,
        description="Bullet points describing responsibilities and achievements"
    )
    technologies: Optional[List[str]] = Field(
        None,
        description="Technologies or tools used in this role"
    )


class SkillCategory(BaseModel):
    """Category of skills (e.g., 'Programming', 'Languages')."""
    name: str = Field(..., description="Name of the skill category")
    skills: List[str] = Field(
        default_factory=list,
        description="List of skills in this category"
    )


class LanguageProficiency(BaseModel):
    """Language proficiency level."""
    language: str = Field(..., description="Language name")
    proficiency: Optional[str] = Field(
        None,
        description="Proficiency level (e.g., 'Native', 'Fluent', 'Intermediate')"
    )


class Certification(BaseModel):
    """Professional certification."""
    name: str = Field(..., description="Name of the certification")
    issuer: Optional[str] = Field(
        None,
        description="Issuing organization"
    )
    date_obtained: Optional[Union[date, str]] = Field(
        None,
        description="Date obtained in YYYY-MM-DD format or just YYYY"
    )
    expiration_date: Optional[Union[date, str]] = Field(
        None,
        description="Expiration date in YYYY-MM-DD format or just YYYY"
    )


class Project(BaseModel):
    """Project experience."""
    name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Description of the project")
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies used in the project"
    )
    url: Optional[str] = Field(
        None,
        description="URL to the project (if available)"
    )
    role: Optional[str] = Field(
        None,
        description="Role in the project"
    )
    start_date: Optional[Union[date, str]] = Field(
        None,
        description="Start date in YYYY-MM-DD format or just YYYY"
    )
    end_date: Optional[Union[date, str, Literal['Present']]] = Field(
        None,
        description="End date in YYYY-MM-DD format, just YYYY, or 'Present'"
    )


class ResumeData(BaseModel):
    """Complete resume data model."""
    contact_info: ContactInfo = Field(
        default_factory=ContactInfo,
        description="Contact information"
    )
    summary: Optional[str] = Field(
        None,
        description="Professional summary or objective statement"
    )
    experience: List[Experience] = Field(
        default_factory=list,
        description="Work experience entries"
    )
    education: List[Education] = Field(
        default_factory=list,
        description="Education history"
    )
    skills: List[SkillCategory] = Field(
        default_factory=list,
        description="Skills organized by category"
    )
    languages: List[LanguageProficiency] = Field(
        default_factory=list,
        description="Language proficiencies"
    )
    certifications: List[Certification] = Field(
        default_factory=list,
        description="Professional certifications"
    )
    projects: List[Project] = Field(
        default_factory=list,
        description="Notable projects"
    )
    raw_text: Optional[str] = Field(
        None,
        description="Original raw text of the resume"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "contact_info": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1 (555) 123-4567",
                    "location": "San Francisco, CA, USA",
                    "linkedin": "linkedin.com/in/johndoe"
                },
                "summary": "Experienced software engineer with 5+ years of experience...",
                "experience": [
                    {
                        "company": "Tech Corp",
                        "position": "Senior Software Engineer",
                        "location": "San Francisco, CA",
                        "start_date": "2020-01-01",
                        "end_date": "Present",
                        "description": [
                            "Led a team of 5 developers to deliver new features",
                            "Improved system performance by 40%"
                        ],
                        "technologies": ["Python", "Django", "React"]
                    }
                ],
                "education": [
                    {
                        "institution": "Stanford University",
                        "degree": "M.S. in Computer Science",
                        "field_of_study": "Artificial Intelligence",
                        "start_date": "2018-09-01",
                        "end_date": "2020-05-15",
                        "gpa": 3.9
                    }
                ],
                "skills": [
                    {
                        "name": "Programming Languages",
                        "skills": ["Python", "JavaScript", "TypeScript"]
                    }
                ],
                "languages": [
                    {"language": "English", "proficiency": "Native"},
                    {"language": "Spanish", "proficiency": "Intermediate"}
                ]
            }
        }

    @validator('experience', 'education', 'projects', pre=True)
    def ensure_list(cls, v):
        """Ensure these fields are always lists, even if None is provided."""
        return v or []
