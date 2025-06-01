"""Pydantic models for structured data extraction."""
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import date, datetime
from pydantic import BaseModel, Field, HttpUrl, EmailStr, validator, root_validator
from enum import Enum

class ContactInfo(BaseModel):
    """Contact information model."""
    name: str = Field(..., description="Full name of the candidate")
    email: EmailStr = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City, State/Province, Country")
    linkedin: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    github: Optional[HttpUrl] = Field(None, description="GitHub profile URL")
    portfolio: Optional[HttpUrl] = Field(None, description="Personal website or portfolio URL")

class Skill(BaseModel):
    """Skill model."""
    name: str = Field(..., description="Name of the skill")
    proficiency: Optional[str] = Field(None, description="Proficiency level (e.g., Beginner, Intermediate, Advanced, Expert)")
    years_of_experience: Optional[float] = Field(None, description="Years of experience with this skill")

class Experience(BaseModel):
    """Work experience model."""
    job_title: str = Field(..., description="Job title or position")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="City, State/Province, Country")
    start_date: str = Field(..., description="Start date (YYYY-MM or YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM or YYYY-MM-DD or 'Present' if current)")
    description: Optional[str] = Field(None, description="Job description")
    achievements: List[str] = Field(default_factory=list, description="Key achievements or responsibilities")
    is_current: bool = Field(False, description="Whether this is the current position")

class Education(BaseModel):
    """Education model."""
    degree: str = Field(..., description="Degree or certification name")
    institution: str = Field(..., description="Name of the educational institution")
    field_of_study: Optional[str] = Field(None, description="Field of study or major")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM or YYYY)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM or YYYY)")
    gpa: Optional[float] = Field(None, description="GPA (if applicable)")
    description: Optional[str] = Field(None, description="Additional details about the education")

class Project(BaseModel):
    """Project model."""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM or YYYY)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM, YYYY, or 'Present' if ongoing)")
    url: Optional[HttpUrl] = Field(None, description="URL to the project")

class Language(BaseModel):
    """Language proficiency model."""
    name: str = Field(..., description="Language name")
    proficiency: str = Field(..., description="Proficiency level (e.g., Native, Fluent, Intermediate, Basic)")

class Certification(BaseModel):
    """Certification model."""
    name: str = Field(..., description="Name of the certification")
    issuer: str = Field(..., description="Issuing organization")
    issue_date: Optional[str] = Field(None, description="Date issued (YYYY-MM or YYYY)")
    expiration_date: Optional[str] = Field(None, description="Expiration date (YYYY-MM or YYYY) if applicable")
    credential_id: Optional[str] = Field(None, description="Credential ID or license number")
    credential_url: Optional[HttpUrl] = Field(None, description="URL to verify the credential")

class Resume(BaseModel):
    """Complete resume model."""
    contact_info: ContactInfo = Field(..., description="Contact information")
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    skills: List[Skill] = Field(default_factory=list, description="List of skills")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education history")
    projects: List[Project] = Field(default_factory=list, description="Projects")
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")
    languages: List[Language] = Field(default_factory=list, description="Languages spoken")
    
    class Config:
        json_schema_extra = {
            "example": {
                "contact_info": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1-555-123-4567",
                    "location": "San Francisco, CA, USA"
                },
                "summary": "Experienced software engineer with 5+ years of experience...",
                "skills": [
                    {"name": "Python", "proficiency": "Expert", "years_of_experience": 5},
                    {"name": "Machine Learning", "proficiency": "Advanced", "years_of_experience": 3}
                ],
                "experience": [
                    {
                        "job_title": "Senior Software Engineer",
                        "company": "Tech Corp",
                        "start_date": "2020-01",
                        "end_date": "Present",
                        "is_current": True,
                        "achievements": ["Led a team of 5 developers..."]
                    }
                ],
                "education": [
                    {
                        "degree": "MSc in Computer Science",
                        "institution": "Stanford University",
                        "field_of_study": "Machine Learning",
                        "end_date": "2019"
                    }
                ]
            }
        }
