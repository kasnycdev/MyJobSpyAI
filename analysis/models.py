from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Enhanced Resume Data Model ---
class ExperienceDetail(BaseModel):
    job_title: Optional[str] = Field(None, description="Specific job title held.")
    company: Optional[str] = Field(None, description="Company name.")
    duration: Optional[str] = Field(None, description="Duration of employment (e.g., 'Jan 2020 - Dec 2022', '3 years').")
    responsibilities: List[str] = Field([], description="Key responsibilities and achievements.")
    quantifiable_achievements: List[str] = Field([], description="Specific, measurable achievements (e.g., 'Increased sales by 15%', 'Managed team of 5').")

    import pandas as pd
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    class ExperienceDetail(BaseModel):
        # ... (Existing code)

    class EducationDetail(BaseModel):
        degree: Optional[str] = Field(None, description="Degree obtained (e.g., 'B.S. Computer Science').")
        institution: Optional[str] = Field(None, description="Name of the educational institution.")
        graduation_year: Optional[str] = Field(None, description="Year of graduation or expected graduation.") # Keep as string for flexibility


    class SkillDetail(BaseModel):
        name: str = Field(..., description="Name of the skill.")
        level: Optional[str] = Field(None, description="Proficiency level (e.g., 'Advanced', 'Intermediate', 'Familiar', 'Expert').")
        years_experience: Optional[int] = Field(None, description="Approximate years of experience with the skill.")


    class ResumeData(BaseModel):
        """Structured representation of resume data extracted by LLM."""
        full_name: Optional[str] = Field(None, description="Candidate's full name.")
        contact_information: Dict[str, Optional[str]] = Field({}, description="Dictionary containing email, phone, LinkedIn URL, portfolio URL etc.")
        summary: Optional[str] = Field(None, description="Professional summary or objective statement.")
        work_experience: List[ExperienceDetail] = Field([], description="List of professional experiences.")
        education: List[EducationDetail] = Field([], description="List of educational qualifications.")
        technical_skills: List[SkillDetail] = Field([], description="List of technical skills with optional proficiency/years.")
        soft_skills: List[str] = Field([], description="List of soft skills or competencies.")
        certifications: List[str] = Field([], description="List of relevant certifications.")
        projects: List[Dict[str, Any]] = Field([], description="List of personal or academic projects with details like name, description, technologies used.")
        languages: List[str] = Field([], description="List of spoken/written languages.")
        raw_text_hash: Optional[str] = Field(None, description="MD5 hash of the raw text used for extraction (for caching).")


    # --- Structured Job Data Model (Extracted from Description) ---
    class ParsedJobData(BaseModel):
        """Structured representation of key job mandate details extracted by LLM."""
        job_title_extracted: Optional[str] = Field(None, description="Job title as interpreted from the description.")
        key_responsibilities: List[str] = Field([], description="Primary duties and tasks mentioned.")
        required_skills: List[SkillDetail] = Field([], description="Skills explicitly listed as required or essential.")
        preferred_skills: List[SkillDetail] = Field([], description="Skills listed as preferred, desired, or 'nice-to-have'.")
        required_experience_years: Optional[int] = Field(None, description="Minimum years of experience required (numeric).")
        preferred_experience_years: Optional[int] = Field(None, description="Preferred years of experience.")
        required_education: Optional[str] = Field(None, description="Minimum education level or degree specified (e.g., 'Bachelor's degree', 'Master's in CS').")
        preferred_education: Optional[str] = Field(None, description="Preferred education level or degree.")
        salary_range_extracted: Optional[str] = Field(None, description="Salary range found within the description text, if any.")
        work_model_extracted: Optional[str] = Field(None, description="Work model inferred (Remote, Hybrid, On-site).")
        company_culture_hints: List[str] = Field([], description="Keywords or phrases hinting at company culture (e.g., 'fast-paced', 'collaborative').")
        tools_technologies: List[str] = Field([], description="Specific tools or technologies mentioned (e.g., 'AWS', 'Jira', 'Salesforce').")


    # --- Enhanced Analysis Result Model ---
    class JobAnalysisResult(BaseModel):
        """Detailed analysis comparing resume to job description."""
        suitability_score: int = Field(..., ge=0, le=100, description="Overall suitability score (0-100).")
        justification: str = Field(..., description="Detailed explanation for the score, citing resume/job details.")
        pros: List[str] = Field([], description="Specific points where the resume strongly matches the job requirements.")
        cons: List[str] = Field([], description="Specific points where the resume lacks alignment or requirements are not met.")
        skill_match_summary: Optional[str] = Field(None, description="Brief summary of skill alignment.")
        experience_match_summary: Optional[str] = Field(None, description="Brief summary of experience alignment.")
        education_match_summary: Optional[str] = Field(None, description="Brief summary of education alignment.")
        missing_keywords: List[str] = Field([], description="Keywords/skills from the job description potentially missing in the resume.")
        # Optional: Add sub-scores if the prompt is designed to output them
        # skill_score: Optional[int] = Field(None, ge=0, le=100)
        # experience_score: Optional[int] = Field(None, ge=0, le=100)
        # education_score: Optional[int] = Field(None, ge=0, le=100)


    class AnalyzedJob(BaseModel):
        original_job_data: Dict[str, Any]
        parsed_job_details: Optional[ParsedJobData]
        analysis: Optional[JobAnalysisResult]

        @property
        def score(self) -> int:
            return self.analysis.suitability_score if self.analysis else 0

        def model_dump(self, *args, **kwargs):
            """Custom model_dump to handle Pandas objects."""
            exclude_unset = kwargs.pop('exclude_unset', True)
            by_alias = kwargs.pop('by_alias', True)
            dumped = super().model_dump(*args, **kwargs)

            # Convert Pandas objects to dictionaries
            if isinstance(dumped['original_job_data'], pd.Series):
                dumped['original_job_data'] = dumped['original_job_data'].to_dict()
            if dumped['parsed_job_details'] is not None and isinstance(dumped['parsed_job_details'], dict):
                dumped['parsed_job_details'] = dumped['parsed_job_details']
            if dumped['analysis'] is not None and isinstance(dumped['analysis'], dict):
                dumped['analysis'] = dumped['analysis']

            return dumped

class SkillDetail(BaseModel):
    name: str = Field(..., description="Name of the skill.")
    level: Optional[str] = Field(None, description="Proficiency level (e.g., 'Advanced', 'Intermediate', 'Familiar', 'Expert').")
    years_experience: Optional[int] = Field(None, description="Approximate years of experience with the skill.")

class ResumeData(BaseModel):
    """Structured representation of resume data extracted by LLM."""
    full_name: Optional[str] = Field(None, description="Candidate's full name.")
    contact_information: Dict[str, Optional[str]] = Field({}, description="Dictionary containing email, phone, LinkedIn URL, portfolio URL etc.")
    summary: Optional[str] = Field(None, description="Professional summary or objective statement.")
    work_experience: List[ExperienceDetail] = Field([], description="List of professional experiences.")
    education: List[EducationDetail] = Field([], description="List of educational qualifications.")
    technical_skills: List[SkillDetail] = Field([], description="List of technical skills with optional proficiency/years.")
    soft_skills: List[str] = Field([], description="List of soft skills or competencies.")
    certifications: List[str] = Field([], description="List of relevant certifications.")
    projects: List[Dict[str, Any]] = Field([], description="List of personal or academic projects with details like name, description, technologies used.")
    languages: List[str] = Field([], description="List of spoken/written languages.")
    raw_text_hash: Optional[str] = Field(None, description="MD5 hash of the raw text used for extraction (for caching).")

# --- Structured Job Data Model (Extracted from Description) ---
class ParsedJobData(BaseModel):
    """Structured representation of key job mandate details extracted by LLM."""
    job_title_extracted: Optional[str] = Field(None, description="Job title as interpreted from the description.")
    key_responsibilities: List[str] = Field([], description="Primary duties and tasks mentioned.")
    required_skills: List[SkillDetail] = Field([], description="Skills explicitly listed as required or essential.")
    preferred_skills: List[SkillDetail] = Field([], description="Skills listed as preferred, desired, or 'nice-to-have'.")
    required_experience_years: Optional[int] = Field(None, description="Minimum years of experience required (numeric).")
    preferred_experience_years: Optional[int] = Field(None, description="Preferred years of experience.")
    required_education: Optional[str] = Field(None, description="Minimum education level or degree specified (e.g., 'Bachelor's degree', 'Master's in CS').")
    preferred_education: Optional[str] = Field(None, description="Preferred education level or degree.")
    salary_range_extracted: Optional[str] = Field(None, description="Salary range found within the description text, if any.")
    work_model_extracted: Optional[str] = Field(None, description="Work model inferred (Remote, Hybrid, On-site).")
    company_culture_hints: List[str] = Field([], description="Keywords or phrases hinting at company culture (e.g., 'fast-paced', 'collaborative').")
    tools_technologies: List[str] = Field([], description="Specific tools or technologies mentioned (e.g., 'AWS', 'Jira', 'Salesforce').")

# --- Enhanced Analysis Result Model ---
class JobAnalysisResult(BaseModel):
    """Detailed analysis comparing resume to job description."""
    suitability_score: int = Field(..., ge=0, le=100, description="Overall suitability score (0-100).")
    justification: str = Field(..., description="Detailed explanation for the score, citing resume/job details.")
    pros: List[str] = Field([], description="Specific points where the resume strongly matches the job requirements.")
    cons: List[str] = Field([], description="Specific points where the resume lacks alignment or requirements are not met.")
    skill_match_summary: Optional[str] = Field(None, description="Brief summary of skill alignment.")
    experience_match_summary: Optional[str] = Field(None, description="Brief summary of experience alignment.")
    education_match_summary: Optional[str] = Field(None, description="Brief summary of education alignment.")
    missing_keywords: List[str] = Field([], description="Keywords/skills from the job description potentially missing in the resume.")
    # Optional: Add sub-scores if the prompt is designed to output them
    # skill_score: Optional[int] = Field(None, ge=0, le=100)
    # experience_score: Optional[int] = Field(None, ge=0, le=100)
    # education_score: Optional[int] = Field(None, ge=0, le=100)

class AnalyzedJob(BaseModel):
    """Combines original job data with its analysis result and extracted details."""
    original_job_data: Dict[str, Any] = Field(..., description="The raw job data dictionary as scraped/loaded.")
    parsed_job_details: Optional[ParsedJobData] = Field(None, description="Structured details extracted from the job description by the LLM.")
    analysis: Optional[JobAnalysisResult] = Field(None, description="The suitability analysis result comparing resume to the job.")

    # Add a helper property for sorting/filtering convenience if needed    import pandas as pd
    from pydantic import BaseModel, Field
    from typing import List, Optional, Dict, Any

    class ExperienceDetail(BaseModel):
        # ... (Existing code)

    class EducationDetail(BaseModel):
        # ... (Existing code)

    class SkillDetail(BaseModel):
        # ... (Existing code)

    class ResumeData(BaseModel):
        # ... (Existing code)

    class ParsedJobData(BaseModel):
        # ... (Existing code)

    class JobAnalysisResult(BaseModel):
        # ... (Existing code)

    class AnalyzedJob(BaseModel):
        original_job_data: Dict[str, Any]
        parsed_job_details: Optional[ParsedJobData]
        analysis: Optional[JobAnalysisResult]

        @property
        def score(self) -> int:
            return self.analysis.suitability_score if self.analysis else 0

        def model_dump(self, *args, **kwargs):
            """Custom model_dump to handle Pandas objects."""
            exclude_unset = kwargs.pop('exclude_unset', True)
            by_alias = kwargs.pop('by_alias', True)
            dumped = super().model_dump(*args, **kwargs)

            # Convert Pandas objects to dictionaries
            if isinstance(dumped['original_job_data'], pd.Series):
                dumped['original_job_data'] = dumped['original_job_data'].to_dict()
            if dumped['parsed_job_details'] is not None and isinstance(dumped['parsed_job_details'], dict):
                dumped['parsed_job_details'] = dumped['parsed_job_details']
            if dumped['analysis'] is not None and isinstance(dumped['analysis'], dict):
                dumped['analysis'] = dumped['analysis']

            return dumped