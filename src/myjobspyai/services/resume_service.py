"""
Resume Service for MyJobSpyAI.

This module provides services for parsing, analyzing, and processing resumes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from myjobspyai.analysis.analyzer import ResumeAnalyzer
from myjobspyai.models.resume import ResumeData

logger = logging.getLogger(__name__)


class ResumeService:
    """Service for resume parsing and analysis operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the resume service.

        Args:
            config: Configuration dictionary for the service
        """
        self.config = config or {}
        self.analyzer = ResumeAnalyzer()

    async def parse_resume(
        self, resume_path: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Parse a resume file and extract structured data.

        Args:
            resume_path: Path to the resume file
            **kwargs: Additional arguments for the parser

        Returns:
            Dictionary containing parsed resume data, or None if parsing fails
        """
        try:
            # Read the resume file
            with open(resume_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()

            # Parse the resume
            resume_data = await self.analyzer.extract_resume_data_async(resume_text)

            # Convert to Pydantic model for validation
            validated_data = ResumeData(**resume_data)

            # Return as dict for easier serialization
            return validated_data.model_dump()

        except Exception as e:
            logger.error(f"Failed to parse resume {resume_path}: {e}", exc_info=True)
            return None

    async def analyze_resume(
        self, resume_data: Dict[str, Any], job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a resume against job requirements.

        Args:
            resume_data: Parsed resume data
            job_data: Job data to analyze against

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert to Pydantic model if needed
            if not isinstance(resume_data, ResumeData):
                resume_data = ResumeData(**resume_data)

            # Perform analysis
            analysis = await self.analyzer.analyze_resume_suitability(
                resume_data=resume_data, job_data=job_data
            )

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze resume: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_resume_skills(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and categorize skills from resume data.

        Args:
            resume_data: Parsed resume data

        Returns:
            Dictionary of categorized skills
        """
        try:
            if not isinstance(resume_data, ResumeData):
                resume_data = ResumeData(**resume_data)

            skills = {
                "technical": [],
                "soft": [],
                "languages": [],
                "certifications": [],
            }

            # Categorize skills
            for skill in resume_data.skills:
                if skill.category == "technical":
                    skills["technical"].append(
                        {
                            "name": skill.name,
                            "proficiency": skill.proficiency,
                            "years": skill.years_experience,
                        }
                    )
                elif skill.category == "language":
                    skills["languages"].append(
                        {"name": skill.name, "proficiency": skill.proficiency}
                    )
                else:
                    skills["soft"].append(
                        {"name": skill.name, "proficiency": skill.proficiency}
                    )

            # Add certifications
            if hasattr(resume_data, 'certifications'):
                skills["certifications"] = [
                    {"name": cert["name"], "issuer": cert.get("issuer", "")}
                    for cert in resume_data.certifications
                ]

            return skills

        except Exception as e:
            logger.error(f"Failed to extract skills from resume: {e}", exc_info=True)
            return {"error": str(e)}
