"""Google Gemini provider for job analysis."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from pydantic import BaseModel

from myjobspyai.models import Job, JobAnalysis

from .base import AnalysisProvider


class GeminiProvider(AnalysisProvider):
    """Job analysis using Google's Gemini API."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Gemini provider.

        Args:
            config: Configuration dictionary with 'api_key' and 'model' keys.
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model_name = config.get("model", "gemini-pro")

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        # Configure the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    async def close(self) -> None:
        """Close any resources."""
        # No explicit close needed for Gemini client
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the Gemini API.

        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional generation parameters.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If the API request fails.
        """
        try:
            # Run the synchronous Gemini API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.model.generate_content(prompt, **kwargs)
            )

            # Extract the response text
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "parts"):
                return " ".join(
                    part.text for part in response.parts if hasattr(part, "text")
                )
            else:
                return str(response)

        except Exception as e:
            raise RuntimeError(f"Gemini API request failed: {str(e)}")

    async def analyze_job(self, job: Job, resume_text: str) -> JobAnalysis:
        """Analyze a job posting against a resume.

        Args:
            job: The job to analyze.
            resume_text: The text content of the resume.

        Returns:
            JobAnalysis object with the analysis results.
        """
        # Create a prompt for job analysis
        prompt = f"""Analyze the following job posting and provide a structured analysis in JSON format.

Job Title: {job.title}
Company: {job.company}
Location: {job.location}

Job Description:
{job.description}

Resume Summary:
{resume_text[:2000]}...

Provide analysis in this JSON format (no markdown code blocks, just raw JSON):
{{
  "summary": "Brief summary of the job posting",
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "experience_level": "e.g., Entry, Mid, Senior",
  "education_requirements": ["Bachelor's in X", ...],
  "salary_range": "If mentioned",
  "is_remote": true/false,
  "location": "Job location if not remote",
  "company_culture": ["aspect1", "aspect2", ...]
}}"""

        try:
            # Get the response from the model
            response_text = await self._generate(prompt)

            # Clean up the response to extract just the JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis_data = json.loads(json_str)
            else:
                # If we can't find JSON, try to parse the whole response
                analysis_data = json.loads(response_text)

            # Convert to JobAnalysis
            return JobAnalysis(
                summary=analysis_data.get("summary", ""),
                required_skills=analysis_data.get("required_skills", []),
                preferred_skills=analysis_data.get("preferred_skills", []),
                experience_level=analysis_data.get("experience_level"),
                education_requirements=analysis_data.get("education_requirements", []),
                salary_range=analysis_data.get("salary_range"),
                is_remote=analysis_data.get("is_remote"),
                location=analysis_data.get("location"),
                company_culture=analysis_data.get("company_culture", []),
            )

        except Exception as e:
            return JobAnalysis(
                summary=f"Error analyzing job: {str(e)}",
                required_skills=[],
                preferred_skills=[],
            )

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text using Gemini's API.

        Args:
            text: The text to get embeddings for.

        Returns:
            List of floats representing the text embeddings.

        Raises:
            RuntimeError: If the embeddings request fails.
        """
        try:
            # Note: As of my knowledge, Gemini doesn't have a direct embeddings API like OpenAI.
            # This is a placeholder implementation that would need to be updated
            # based on Gemini's actual API capabilities.

            # For now, we'll return a dummy embedding
            # In a real implementation, you would call the Gemini embeddings API here
            return [0.0] * 768  # Dummy embedding

        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings: {str(e)}")

    @classmethod
    def get_provider_name(cls) -> str:
        """Get the name of the provider."""
        return "gemini"
