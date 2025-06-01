"""LangChain-based resume analyzer implementation."""
import logging
from typing import Dict, Any, Optional

from myjobspyai.analysis.components.analyzers.base import BaseAnalyzer, AnalysisError
from myjobspyai.analysis.components.analyzers.resume_analyzer import (
    ContactInfo, ExperienceItem, EducationItem, ResumeData
)
from myjobspyai.analysis.components.factory import get_analyzer as get_langchain_analyzer
from myjobspyai.analysis.components.langchain_integration import ResumeAnalysis
from myjobspyai.analysis.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class LangChainResumeAnalyzer(BaseAnalyzer):
    """Resume analyzer using LangChain and LLMs for extraction."""
    
    def __init__(self, provider: BaseProvider, model: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the LangChain resume analyzer.
        
        Args:
            provider: The LLM provider (not used directly, kept for compatibility)
            model: The model name to use
            config: Configuration dictionary
        """
        super().__init__(provider, model, config)
        self.analyzer = get_langchain_analyzer()
    
    async def analyze(self, resume_text: str) -> ResumeData:
        """Analyze resume text and extract structured data.
        
        Args:
            resume_text: The raw text content of the resume
            
        Returns:
            ResumeData object with structured information
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not resume_text or not isinstance(resume_text, str):
            raise ValueError("resume_text must be a non-empty string")
        
        try:
            # Use LangChain analyzer to process the resume
            analysis = await self.analyzer.analyze(resume_text)
            
            # Convert to our standard ResumeData format
            return self._convert_to_resume_data(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing resume with LangChain: {str(e)}")
            raise AnalysisError(f"Failed to analyze resume: {str(e)}")
    
    def _convert_to_resume_data(self, analysis: ResumeAnalysis) -> ResumeData:
        """Convert LangChain analysis to our standard ResumeData format."""
        # Create contact info (minimal, as we might not have this from the raw analysis)
        contact_info = ContactInfo(
            name="Unknown",  # Will be updated if we can extract it
            email=None,
            phone=None,
            location=None
        )
        
        # Convert experience
        experience = []
        for exp in analysis.experience:
            try:
                exp_item = ExperienceItem(
                    company=exp.company,
                    title=exp.title,
                    location=exp.location,
                    start_date=exp.start_date,
                    end_date=exp.end_date,
                    is_current=exp.is_current,
                    description=exp.description,
                )
                experience.append(exp_item)
            except Exception as e:
                logger.warning(f"Failed to parse experience item: {exp}. Error: {e}")
        
        # Convert education
        education = []
        for edu in analysis.education:
            try:
                edu_item = EducationItem(
                    institution=edu.institution,
                    degree=edu.degree,
                    field_of_study=edu.field_of_study,
                    start_date=edu.start_date,
                    end_date=edu.end_date,
                    gpa=edu.gpa
                )
                education.append(edu_item)
            except Exception as e:
                logger.warning(f"Failed to parse education item: {edu}. Error: {e}")
        
        # Create the resume data
        resume_data = ResumeData(
            contact_info=contact_info,
            summary=analysis.summary,
            skills=analysis.skills,
            experience=experience,
            education=education,
            certifications=[],  # Not currently in LangChain model
            projects=[],  # Not currently in LangChain model
            languages=[],  # Not currently in LangChain model
            social_profiles={}  # Not currently in LangChain model
        )
        
        return resume_data
