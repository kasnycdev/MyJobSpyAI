"""LangChain integration for resume analysis and candidate matching."""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging

from pydantic import BaseModel, Field, HttpUrl, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing_extensions import Literal

logger = logging.getLogger(__name__)

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXPERT = "expert"

class EducationEntry(BaseModel):
    institution: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree obtained")
    field_of_study: Optional[str] = Field(None, description="Field of study")
    start_year: Optional[int] = Field(None, description="Start year")
    end_year: Optional[int] = Field(None, description="End year (or None if current)")
    gpa: Optional[float] = Field(None, description="GPA if available")

class ExperienceEntry(BaseModel):
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD), None if current")
    is_current: bool = Field(False, description="Whether this is the current position")
    description: Optional[str] = Field(None, description="Job description and achievements")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")

class ResumeAnalysis(BaseModel):
    """Structured resume analysis result."""
    skills: List[str] = Field(default_factory=list, description="List of skills")
    experience: List[ExperienceEntry] = Field(default_factory=list, description="Work experience")
    education: List[EducationEntry] = Field(default_factory=list, description="Education history")
    summary: str = Field(..., description="Professional summary")
    experience_level: Optional[ExperienceLevel] = Field(None, description="Experience level")
    years_experience: Optional[float] = Field(None, description="Total years of experience")

class MatchScore(BaseModel):
    """Candidate matching score and details."""
    overall_score: float = Field(..., ge=0, le=10, description="Overall match score (0-10)")
    skill_match: float = Field(..., ge=0, le=1, description="Skill match ratio (0-1)")
    experience_match: float = Field(..., ge=0, le=1, description="Experience match ratio (0-1)")
    education_match: float = Field(..., ge=0, le=1, description="Education match ratio (0-1)")
    missing_skills: List[str] = Field(default_factory=list, description="Missing required skills")
    matching_experience: List[Dict] = Field(default_factory=list, description="Matching experience details")
    explanation: str = Field(..., description="Detailed explanation of the score")

class ResumeAnalyzer:
    """Analyze resumes using LangChain and LLMs."""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1):
        """Initialize the analyzer.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for LLM generation (0-1)
        """
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            format="json",
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        self.parser = JsonOutputParser(pydantic_object=ResumeAnalysis)
        self.prompt = self._create_analysis_prompt()
        
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for resume analysis."""
        template = """You are an expert resume analyst. Extract structured information from the resume.
        
        Instructions:
        1. Extract all technical and professional skills
        2. List all work experiences with details
        3. Include all education history
        4. Write a concise professional summary
        5. Estimate experience level and total years
        
        {format_instructions}
        
        Resume:
        {resume_text}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{resume_text}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
    
    async def analyze(self, resume_text: str) -> ResumeAnalysis:
        """Analyze a resume and return structured data."""
        try:
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({"resume_text": resume_text})
            return ResumeAnalysis(**result)
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            raise AnalysisError(f"Failed to analyze resume: {str(e)}")

class CandidateMatcher:
    """Match candidates to job descriptions using semantic similarity."""
    
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1):
        """Initialize the matcher.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for LLM generation (0-1)
        """
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            format="json",
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        self.parser = JsonOutputParser(pydantic_object=MatchScore)
        self.prompt = self._create_matching_prompt()
        
    def _create_matching_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for candidate matching."""
        template = """You are an expert recruiter. Evaluate how well the candidate matches the job requirements.
        
        Job Requirements:
        {job_description}
        
        Candidate Profile:
        {candidate_profile}
        
        Instructions:
        1. Score the match from 0-10
        2. Calculate match ratios for skills, experience, and education
        3. List missing required skills
        4. Provide a detailed explanation
        
        {format_instructions}"""
        
        return ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", "{job_description}\n\n{candidate_profile}")
        ]).partial(format_instructions=self.parser.get_format_instructions())
    
    async def match(
        self,
        job_description: str,
        candidate_profile: str,
        required_skills: Optional[List[str]] = None
    ) -> MatchScore:
        """Match a candidate to a job description.
        
        Args:
            job_description: The job description text
            candidate_profile: The candidate's resume or profile text
            required_skills: List of required skills (if any)
            
        Returns:
            MatchScore with detailed matching results
        """
        try:
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "job_description": job_description,
                "candidate_profile": candidate_profile,
                "required_skills": required_skills or []
            })
            return MatchScore(**result)
        except Exception as e:
            logger.error(f"Error matching candidate: {str(e)}")
            raise AnalysisError(f"Failed to match candidate: {str(e)}")

class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass
