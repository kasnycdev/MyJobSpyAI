"""Ollama provider for job analysis using local LLMs."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, HttpUrl

from myjobspyai.models import Job, JobAnalysis, JobMatch

from .base import AnalysisProvider


class OllamaProvider(AnalysisProvider):
    """Job analysis using Ollama's local LLM API."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama provider.

        Args:
            config: Configuration dictionary with 'base_url' and 'model' keys.
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = config.get("model", "llama3")
        self.timeout = config.get("timeout", 300)  # 5 minutes default
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the Ollama API.

        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional generation parameters.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If the API request fails.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/generate"

        payload = {"model": self.model, "prompt": prompt, "stream": False, **kwargs}

        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("response", "")

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Ollama API request failed: {str(e)}")

    async def analyze_job(self, job: Job, resume_text: str) -> JobAnalysis:
        """Analyze a job posting against a resume.

        Args:
            job: The job to analyze.
            resume_text: The text content of the resume.

        Returns:
            JobAnalysis object with the analysis results.
        """
        # Create a prompt for job analysis
        prompt = f"""Analyze the following job posting and provide a structured analysis.

Job Title: {job.title}
Company: {job.company}
Location: {job.location}

Job Description:
{job.description}

Resume Summary:
{resume_text[:2000]}...

Please provide analysis in the following JSON format:
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
            # Get the raw response from the model
            response_text = await self._generate(prompt, format="json")

            # Try to parse the JSON response
            try:
                analysis_data = json.loads(response_text.strip())
            except json.JSONDecodeError:
                # If parsing fails, try to extract JSON from the response
                analysis_data = self._extract_json(response_text)

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

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from a text response.

        Args:
            text: The text potentially containing JSON.

        Returns:
            The parsed JSON as a dictionary.

        Raises:
            ValueError: If no valid JSON is found.
        """
        # Simple implementation - in practice, you might want something more robust
        import json
        import re

        # Look for JSON-like content between ```json ... ```
        match = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError("No valid JSON found in response")

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text.

        Args:
            text: The text to get embeddings for.

        Returns:
            List of floats representing the text embeddings.

        Raises:
            RuntimeError: If the embeddings request fails.
        """
        session = await self._get_session()
        url = f"{self.base_url}/api/embeddings"

        payload = {
            "model": self.model,
            "prompt": text,
        }

        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("embedding", [])

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to get embeddings: {str(e)}")

    @classmethod
    def get_provider_name(cls) -> str:
        """Get the name of the provider."""
        return "ollama"
