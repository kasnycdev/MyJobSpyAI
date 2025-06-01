"""Unit tests for LangChain integration."""
import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from myjobspyai.analysis.components.langchain_integration import (
    ResumeAnalyzer,
    CandidateMatcher,
    ResumeAnalysis,
    MatchScore,
    ExperienceEntry,
    EducationEntry,
    AnalysisError
)

# Sample test data
SAMPLE_RESUME = """
John Doe
Senior Software Engineer

SUMMARY
Experienced software engineer with 5+ years of experience in Python and cloud technologies.

SKILLS
- Python, JavaScript, AWS, Docker, Kubernetes
- Machine Learning, Data Analysis
- Agile Methodologies

EXPERIENCE
Senior Software Engineer
Tech Corp Inc. | 2020 - Present
- Led a team of 5 developers
- Designed and implemented microservices

Software Developer
Dev Solutions | 2018 - 2020
- Developed RESTful APIs
- Worked on CI/CD pipelines

EDUCATION
MS in Computer Science
State University | 2016 - 2018
GPA: 3.8

BS in Computer Science
City College | 2012 - 2016
GPA: 3.6
"""

SAMPLE_JOB_DESCRIPTION = """
We are looking for a Senior Software Engineer with:
- 5+ years of Python experience
- Cloud platform experience (AWS, GCP, or Azure)
- Containerization (Docker, Kubernetes)
- Experience with microservices architecture
- Strong problem-solving skills

Nice to have:
- Machine learning experience
- Team leadership experience
- CI/CD pipeline experience
"""

@patch('myjobspyai.analysis.components.langchain_integration.Ollama')
class TestResumeAnalyzer:
    """Test the ResumeAnalyzer class."""
    
    async def test_analyze_success(self, mock_ollama):
        """Test successful resume analysis."""
        # Setup mock
        mock_llm = AsyncMock()
        mock_ollama.return_value = mock_llm
        
        # Mock LLM response
        expected_result = {
            "skills": ["Python", "AWS", "Docker"],
            "experience": [
                {
                    "company": "Tech Corp Inc.",
                    "title": "Senior Software Engineer",
                    "start_date": "2020-01-01",
                    "is_current": True,
                    "description": "Led team of developers"
                }
            ],
            "education": [
                {
                    "institution": "State University",
                    "degree": "MS in Computer Science",
                    "end_year": 2018
                }
            ],
            "summary": "Experienced engineer with cloud expertise",
            "experience_level": "senior",
            "years_experience": 5.5
        }
        mock_llm.ainvoke.return_value = json.dumps(expected_result)
        
        # Test
        analyzer = ResumeAnalyzer()
        result = await analyzer.analyze(SAMPLE_RESUME)
        
        # Assertions
        assert result.skills == ["Python", "AWS", "Docker"]
        assert len(result.experience) == 1
        assert result.experience[0].company == "Tech Corp Inc."
        assert result.summary == "Experienced engineer with cloud expertise"
        assert result.years_experience == 5.5
    
    async def test_analyze_invalid_response(self, mock_ollama):
        """Test handling of invalid LLM response."""
        mock_llm = AsyncMock()
        mock_ollama.return_value = mock_llm
        mock_llm.ainvoke.return_value = "invalid json"
        
        analyzer = ResumeAnalyzer()
        with pytest.raises(AnalysisError):
            await analyzer.analyze(SAMPLE_RESUME)

@patch('myjobspyai.analysis.components.langchain_integration.Ollama')
class TestCandidateMatcher:
    """Test the CandidateMatcher class."""
    
    async def test_match_success(self, mock_ollama):
        """Test successful candidate matching."""
        mock_llm = AsyncMock()
        mock_ollama.return_value = mock_llm
        
        # Mock LLM response
        expected_result = {
            "overall_score": 8.5,
            "skill_match": 0.9,
            "experience_match": 0.85,
            "education_match": 1.0,
            "missing_skills": ["GCP"],
            "matching_experience": [
                {"skill": "Python", "match": "5+ years"},
                {"skill": "AWS", "match": "3 years"}
            ],
            "explanation": "Strong match with required skills and experience."
        }
        mock_llm.ainvoke.return_value = json.dumps(expected_result)
        
        # Test
        matcher = CandidateMatcher()
        result = await matcher.match(
            job_description=SAMPLE_JOB_DESCRIPTION,
            candidate_profile=SAMPLE_RESUME,
            required_skills=["Python", "AWS", "Docker"]
        )
        
        # Assertions
        assert result.overall_score == 8.5
        assert result.skill_match == 0.9
        assert "GCP" in result.missing_skills
        assert len(result.matching_experience) == 2
    
    async def test_match_with_missing_requirements(self, mock_ollama):
        """Test matching with missing requirements."""
        mock_llm = AsyncMock()
        mock_ollama.return_value = mock_llm
        
        # Mock LLM response with low scores
        expected_result = {
            "overall_score": 3.0,
            "skill_match": 0.3,
            "experience_match": 0.2,
            "education_match": 0.8,
            "missing_skills": ["Python", "AWS", "Docker"],
            "matching_experience": [],
            "explanation": "Missing key required skills."
        }
        mock_llm.ainvoke.return_value = json.dumps(expected_result)
        
        matcher = CandidateMatcher()
        result = await matcher.match(
            job_description="Looking for Python developers with AWS experience.",
            candidate_profile="I have experience with JavaScript and React.",
            required_skills=["Python", "AWS", "Docker"]
        )
        
        assert result.overall_score == 3.0
        assert len(result.missing_skills) == 3

class TestModels:
    """Test the Pydantic models."""
    
    def test_resume_analysis_validation(self):
        """Test ResumeAnalysis model validation."""
        valid_data = {
            "skills": ["Python", "AWS"],
            "experience": [{
                "company": "Test",
                "title": "Developer",
                "start_date": "2020-01-01"
            }],
            "education": [{
                "institution": "University",
                "degree": "BS"
            }],
            "summary": "Test summary"
        }
        result = ResumeAnalysis(**valid_data)
        assert result.summary == "Test summary"
    
    def test_match_score_validation(self):
        """Test MatchScore model validation."""
        valid_data = {
            "overall_score": 7.5,
            "skill_match": 0.8,
            "experience_match": 0.7,
            "education_match": 0.9,
            "missing_skills": ["Docker"],
            "matching_experience": [{"skill": "Python", "years": 5}],
            "explanation": "Good match"
        }
        result = MatchScore(**valid_data)
        assert result.overall_score == 7.5
        assert "Docker" in result.missing_skills

# Run tests with: pytest -v tests/unit/test_langchain_integration.py -s
