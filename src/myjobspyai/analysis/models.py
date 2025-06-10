from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic.v1 import BaseModel, Field, validator


class ExperienceDetail(BaseModel):
    job_title: Optional[str] = Field(None, description="Specific job title held.")
    company: Optional[str] = Field(None, description="Company name.")
    duration: Optional[str] = Field(
        None,
        description="Duration of employment (e.g., 'Jan 2020 - Dec 2022', '3 years').",
    )
    start_date: Optional[str] = Field(
        None, description="Start date of employment (YYYY-MM format)."
    )
    end_date: Optional[str] = Field(
        None,
        description="End date of employment (YYYY-MM format), or 'Present' for current role.",
    )
    responsibilities: List[str] = Field(
        [], description="Key responsibilities and achievements."
    )
    quantifiable_achievements: List[str] = Field(
        [],
        description="Specific, measurable achievements (e.g., 'Increased sales by 15%', 'Managed team of 5').",
    )
    skills_used: List[str] = Field(
        [],
        description="List of skills utilized in this role.",
    )


class EducationDetail(BaseModel):
    degree: Optional[str] = Field(
        None, description="Degree obtained (e.g., 'B.S. Computer Science')."
    )
    field_of_study: Optional[str] = Field(None, description="Field or major of study.")
    institution: Optional[str] = Field(
        None, description="Name of the educational institution."
    )
    graduation_year: Optional[str] = Field(
        None, description="Year of graduation or expected graduation (YYYY)."
    )
    gpa: Optional[str] = Field(None, description="Grade point average, if applicable.")
    honors: List[str] = Field(
        [], description="Any academic honors or distinctions received."
    )


class SkillCategory(str, Enum):
    PROGRAMMING = "programming"
    DATABASE = "database"
    FRAMEWORK = "framework"
    TOOL = "tool"
    LANGUAGE = "language"
    METHODOLOGY = "methodology"
    PLATFORM = "platform"
    CLOUD = "cloud"
    DEVOPS = "devops"
    TESTING = "testing"
    OTHER = "other"


class ProficiencyLevel(str, Enum):
    NOVICE = "Novice"
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class SkillDetail(BaseModel):
    name: str = Field(..., description="Name of the skill.")
    category: Optional[SkillCategory] = Field(
        SkillCategory.OTHER,
        description="Category of the skill for better organization.",
    )
    level: Optional[ProficiencyLevel] = Field(
        None,
        description="Proficiency level for this skill.",
    )
    years_experience: Optional[Union[float, str]] = Field(
        None,
        description="Years of experience with the skill. Can be a decimal or range string.",
    )
    last_used: Optional[str] = Field(
        None,
        description="When this skill was last used (YYYY or YYYY-MM format).",
    )
    relevance: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Relevance of this skill to the target role (0-1 scale).",
    )

    @validator('years_experience', pre=True)
    def parse_years_experience(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                # Handle ranges like "3-5" or "3 to 5"
                if '-' in v or 'to' in v.lower():
                    parts = re.split(r'[-\s]+', v.lower().replace('to', '').strip())
                    nums = [float(p) for p in parts if p.replace('.', '').isdigit()]
                    if nums:
                        return sum(nums) / len(nums)  # Return average for ranges
                return float(v)
            except (ValueError, TypeError):
                return None
        return None


class ResumeData(BaseModel):
    """Structured representation of resume data extracted by LLM with enhanced fields."""

    # Personal Information
    full_name: Optional[str] = Field(None, description="Candidate's full name.")
    contact_information: Dict[str, Optional[str]] = Field(
        {},
        description="Dictionary containing email, phone, LinkedIn URL, portfolio URL, etc.",
    )
    location: Optional[str] = Field(
        None, description="Current location or preferred work location."
    )

    # Professional Information
    summary: Optional[str] = Field(
        None, description="Professional summary or objective statement."
    )
    target_roles: List[str] = Field(
        [],
        description="List of job titles or roles the candidate is targeting.",
    )

    # Experience and Education
    work_experience: List[ExperienceDetail] = Field(
        [],
        description="List of professional experiences in reverse chronological order.",
    )
    education: List[EducationDetail] = Field(
        [],
        description="List of educational qualifications in reverse chronological order.",
    )

    # Skills and Competencies
    technical_skills: List[SkillDetail] = Field(
        [],
        description="List of technical skills with proficiency levels and experience.",
    )
    soft_skills: List[SkillDetail] = Field(
        [],
        description="List of soft skills with proficiency levels.",
    )

    # Additional Sections
    certifications: List[Dict[str, str]] = Field(
        [],
        description="List of certifications with name, issuer, and date obtained.",
    )
    projects: List[Dict[str, Any]] = Field(
        [],
        description="List of personal or academic projects with details like name, description, technologies used, and outcomes.",
    )
    languages: List[Dict[str, str]] = Field(
        [],
        description="List of languages with proficiency level (e.g., 'Native', 'Fluent', 'Intermediate').",
    )

    # Metadata
    last_updated: Optional[str] = Field(
        None,
        description="Date when the resume was last updated (ISO 8601 format).",
    )
    raw_text_hash: Optional[str] = Field(
        None,
        description="MD5 hash of the raw text used for extraction (for caching).",
    )

    # Analysis Metadata
    analysis_metadata: Dict[str, Any] = Field(
        {},
        description="Additional metadata generated during analysis.",
    )


class ParsedJobData(BaseModel):
    """Structured representation of key job mandate details extracted by LLM."""

    job_title_extracted: Optional[str] = Field(
        None, description="Job title as interpreted from the description."
    )
    key_responsibilities: List[Dict[str, Any]] = Field(
        [], description="Primary duties and tasks mentioned."
    )
    required_skills: List[SkillDetail] = Field(
        [], description="Skills explicitly listed as required or essential."
    )
    preferred_skills: List[SkillDetail] = Field(
        [], description="Skills listed as preferred, desired, or 'nice-to-have'."
    )
    required_experience_years: Optional[Any] = Field(  # Changed from Optional[int]
        None,
        description="Minimum years of experience required (numeric). LLM may return a range dict.",
    )
    preferred_experience_years: Optional[Any] = Field(  # Changed from Optional[int]
        None, description="Preferred years of experience. LLM may return a range dict."
    )
    required_education: List[Any] = Field(
        [],
        description="Minimum education level or degree specified (e.g., 'Bachelor's degree', 'Master's in CS').",
    )
    preferred_education: List[Any] = Field(
        [], description="Preferred education level or degree."
    )
    salary_range_extracted: Optional[str] = Field(
        None, description="Salary range found within the description text, if any."
    )
    work_model_extracted: Optional[str] = Field(
        None, description="Work model inferred (Remote, Hybrid, On-site)."
    )
    company_culture_hints: List[Dict[str, Any]] = Field(
        [],
        description="Keywords or phrases hinting at company culture (e.g., 'fast-paced', 'collaborative').",
    )
    tools_technologies: List[Dict[str, Any]] = Field(
        [],
        description="Specific tools or technologies mentioned (e.g., 'AWS', 'Jira', 'Salesforce').",
    )
    job_type: Optional[str] = Field(
        None,
        description="Type of job (e.g., 'Full-time', 'Part-time', 'Contract', 'Internship').",
    )
    industry: Optional[str] = Field(
        None,
        description="Industry or sector (e.g., 'Technology', 'Finance', 'Healthcare').",
    )
    required_certifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Certifications explicitly stated as REQUIRED.",
    )
    preferred_certifications: List[Dict[str, Any]] = Field(
        default_factory=list, description="Certifications stated as PREFERRED."
    )
    security_clearance: Optional[str] = Field(
        None,
        description="Security clearance required (e.g., 'Top Secret', 'Secret', 'None').",
    )
    travel_requirements: Optional[str] = Field(
        None,
        description="Travel requirements (e.g., 'Up to 10%', 'Up to 20%', 'None').",
    )


class SectionScores(BaseModel):
    """Detailed section scores for resume analysis."""

    # Core Sections
    title: float = Field(
        ..., ge=0, le=100, description="Match score for job title alignment."
    )
    summary: float = Field(
        ..., ge=0, le=100, description="Match score for professional summary."
    )

    # Skills and Experience
    skills: Dict[str, float] = Field(
        ..., description="Scores by skill category (e.g., 'programming', 'databases')."
    )
    experience: float = Field(
        ..., ge=0, le=100, description="Match score for work experience."
    )

    # Education and Certifications
    education: float = Field(
        ..., ge=0, le=100, description="Match score for education background."
    )
    certifications: float = Field(
        ..., ge=0, le=100, description="Match score for certifications."
    )

    # Additional Sections
    projects: float = Field(
        ..., ge=0, le=100, description="Match score for project experience."
    )
    achievements: float = Field(
        ..., ge=0, le=100, description="Match score for quantifiable achievements."
    )

    # Soft Factors
    cultural_fit: float = Field(
        ..., ge=0, le=100, description="Cultural fit assessment score."
    )
    growth_potential: float = Field(
        ..., ge=0, le=100, description="Growth potential score."
    )


class ImprovementRecommendation(BaseModel):
    """Structured recommendation for improving the resume."""

    category: str = Field(
        ..., description="Category of improvement (e.g., 'skills', 'experience')."
    )
    description: str = Field(..., description="Detailed recommendation description.")
    priority: Literal["high", "medium", "low"] = Field(
        ..., description="Priority level of the recommendation."
    )
    impact: float = Field(
        ..., ge=0, le=1, description="Estimated impact on overall score (0-1 scale)."
    )
    action_items: List[str] = Field(
        [],
        description="Specific action items to address this recommendation.",
    )


class TrainingResource(BaseModel):
    """Recommended training resource to address skill gaps."""

    title: str = Field(..., description="Name/title of the training resource.")
    url: str = Field(..., description="URL to access the resource.")
    provider: str = Field(
        ..., description="Organization or platform providing the resource."
    )
    resource_type: Literal["course", "book", "tutorial", "certification", "other"] = (
        Field(..., description="Type of resource.")
    )
    estimated_hours: Optional[float] = Field(
        None,
        description="Estimated time commitment in hours.",
    )
    cost: Optional[float] = Field(
        None,
        description="Cost in USD, if applicable.",
    )
    skills_covered: List[str] = Field(
        [],
        description="List of skills this resource helps develop.",
    )


class JobAnalysisResult(BaseModel):
    """
    Comprehensive analysis of resume suitability for a specific job.
    Enhanced with detailed section scores, recommendations, and resources.
    """

    # Core Metrics
    suitability_score: float = Field(
        default=0.0, ge=0, le=100, description="Overall suitability score (0-100)."
    )
    confidence_score: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Confidence level in the analysis (0-1 scale).",
    )

    # Detailed Analysis
    summary: str = Field(
        default="",
        description="Executive summary of the analysis.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths and strong matches with the job requirements.",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Key weaknesses or gaps in the candidate's profile.",
    )

    # Section-by-Section Analysis
    section_scores: Optional[SectionScores] = Field(
        None,
        description="Detailed scores for each resume section.",
    )

    # Improvement Recommendations
    improvement_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of recommendations to improve the resume.",
    )

    # Additional Resources
    cover_letter: Optional[str] = Field(
        None,
        description="Generated cover letter tailored to the job.",
    )
    training_resources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recommended training resources to address skill gaps.",
    )

    # Metadata
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of when the analysis was performed.",
    )
    model_version: str = Field(
        "1.0.0",
        description="Version of the analysis model used.",
    )

    # Backward Compatibility
    @property
    def justification(self) -> str:
        """Backward compatibility with legacy code."""
        return self.summary

    @property
    def pros(self) -> List[str]:
        """Backward compatibility with legacy code."""
        return self.strengths  # Fixed typo: was self.strengs

    @property
    def cons(self) -> List[str]:
        """Backward compatibility with legacy code."""
        return self.weaknesses

    @property
    def key_qualifications(self) -> List[Dict[str, str]]:
        """Backward compatibility with legacy code."""
        return [
            {
                "requirement": rec.get("title", ""),
                "match_strength": str(rec.get("priority", "medium")),
                "evidence": rec.get("description", ""),
            }
            for rec in self.improvement_recommendations
        ]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnalyzedJob(BaseModel):
    """
    Comprehensive analysis of a job posting combined with resume matching results.
    """

    # Core Job Data
    original_job_data: Dict[str, Any] = Field(
        ...,
        description="The raw job data dictionary as scraped/loaded.",
    )

    # Structured Job Details
    parsed_job_details: Optional[ParsedJobData] = Field(
        None,
        description="Structured details extracted from the job description by the LLM.",
    )

    # Analysis Results
    analysis: Optional[JobAnalysisResult] = Field(
        None,
        description="The enhanced suitability analysis result comparing resume to the job.",
    )

    # Metadata
    analyzed_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of when the analysis was performed.",
    )
    analysis_version: str = Field(
        "1.0.0",
        description="Version of the analysis model used.",
    )

    # Properties for backward compatibility
    @property
    def score(self) -> float:
        """Backward compatibility with legacy code."""
        return self.analysis.suitability_score if self.analysis else 0.0

    # Helper methods
    def get_high_priority_recommendations(self) -> List[Dict[str, Any]]:
        """Get high-priority improvement recommendations."""
        if not self.analysis:
            return []
        return [
            rec.dict()
            for rec in self.analysis.improvement_recommendations
            if rec.priority == "high"
        ]

    def get_free_resources(self) -> List[Dict[str, Any]]:
        """Get free training resources."""
        if not self.analysis:
            return []
        return [
            res.dict()
            for res in self.analysis.training_resources
            if res.cost == 0 or res.cost is None
        ]

    def get_skills_gap_analysis(self) -> Dict[str, Any]:
        """Get a summary of skill gaps and recommendations."""
        if not self.analysis or not self.parsed_job_details:
            return {}

        return {
            "missing_skills": [
                skill
                for skill in self.parsed_job_details.required_skills
                if skill.name.lower()
                not in [
                    s.name.lower() for s in self.analysis.section_scores.skills.values()
                ]
            ],
            "recommended_resources": [
                res.dict() for res in self.analysis.training_resources
            ][
                :5
            ],  # Limit to top 5
        }

    def model_dump(self, *args, **kwargs):
        """Custom model_dump to handle Pandas objects."""
        # Use dict() for Pydantic v1 compatibility
        dumped = self.dict(*args, **kwargs)

        # Convert Pandas objects to dictionaries
        if isinstance(dumped["original_job_data"], pd.Series):
            dumped["original_job_data"] = dumped["original_job_data"].to_dict()
        if dumped["parsed_job_details"] is not None and isinstance(
            dumped["parsed_job_details"], dict
        ):
            dumped["parsed_job_details"] = dumped["parsed_job_details"]
        if dumped["analysis"] is not None and isinstance(dumped["analysis"], dict):
            dumped["analysis"] = dumped["analysis"]

        return dumped
