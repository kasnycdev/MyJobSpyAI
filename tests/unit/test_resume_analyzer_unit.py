"""
Unit tests for the ResumeAnalyzer class.

These tests focus on testing the ResumeAnalyzer class in isolation using mocks
for external dependencies.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from pydantic import ValidationError

from myjobspyai.analysis.components.analyzers.resume_analyzer import ResumeAnalyzer
from myjobspyai.analysis.components.analyzers.exceptions import AnalysisError

# Set up test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample test data
SAMPLE_RESUME = """
John Doe
Senior Software Engineer

Contact:
- Email: john.doe@example.com
- Phone: (123) 456-7890
- LinkedIn: linkedin.com/in/johndoe

Skills:
- Python, JavaScript, TypeScript
- Machine Learning, Deep Learning
- AWS, Docker, Kubernetes

Experience:
- Senior Software Engineer at Tech Corp (2020-Present)
  - Led a team of 5 developers
  - Developed microservices in Python

- Software Engineer at Web Solutions (2018-2020)
  - Built RESTful APIs
  - Implemented CI/CD pipelines

Education:
- MS in Computer Science, University of Technology (2018)
- BS in Computer Science, State University (2016)
"""

SAMPLE_SCHEMA = {
    "title": "ResumeData",
    "type": "object",
    "properties": {
        "contact_info": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "linkedin": {"type": "string"}
            },
            "required": ["name", "email"]
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "period": {"type": "string"},
                    "description": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "company"]
            }
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": "integer"}
                },
                "required": ["degree", "institution"]
            }
        }
    },
    "required": ["contact_info", "skills", "experience", "education"]
}

SAMPLE_EXTRACTED_DATA = {
    "contact_info": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "(123) 456-7890",
        "linkedin": "linkedin.com/in/johndoe"
    },
    "skills": ["Python", "JavaScript", "TypeScript", "Machine Learning", "AWS"],
    "experience": [
        {
            "title": "Senior Software Engineer",
            "company": "Tech Corp",
            "period": "2020-Present",
            "description": ["Led a team of 5 developers", "Developed microservices in Python"]
        },
        {
            "title": "Software Engineer",
            "company": "Web Solutions",
            "period": "2018-2020",
            "description": ["Built RESTful APIs", "Implemented CI/CD pipelines"]
        }
    ],
    "education": [
        {
            "degree": "MS in Computer Science",
            "institution": "University of Technology",
            "year": 2018
        },
        {
            "degree": "BS in Computer Science",
            "institution": "State University",
            "year": 2016
        }
    ]
}

@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=json.dumps(SAMPLE_EXTRACTED_DATA))
    return provider

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "provider": "mock",
        "model": "test-model",
        "generation_params": {
            "temperature": 0.2,
            "max_tokens": 2000
        }
    }

@pytest.fixture
def analyzer(mock_provider, mock_config, tmp_path):
    """Create a ResumeAnalyzer instance for testing."""
    # Create temporary template and schema files
    template_dir = tmp_path / "templates" / "resume"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_file = template_dir / "extraction_instructions.prompt"
    template_file.write_text("Extract information from: {resume_text}")
    
    schema_dir = tmp_path / "schemas" / "resume"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_file = schema_dir / "resume_schema.json"
    schema_file.write_text(json.dumps(SAMPLE_SCHEMA))
    
    # Update config with test paths
    mock_config.update({
        "analysis": {
            "templates": {
                "base_dir": str(tmp_path / "templates"),
                "resume": {
                    "extraction": str(template_file)
                }
            },
            "schemas": {
                "base_dir": str(tmp_path / "schemas"),
                "resume": str(schema_file)
            }
        }
    })
    
    # Create analyzer with mock provider and config
    analyzer = ResumeAnalyzer(provider=mock_provider, config=mock_config)
    return analyzer

class TestResumeAnalyzer:
    """Test suite for the ResumeAnalyzer class."""
    
    @pytest.mark.asyncio
    async def test_analyze_resume_success(self, analyzer, mock_provider):
        """Test successful resume analysis."""
        # Mock the provider's generate method
        mock_provider.generate.return_value = json.dumps(SAMPLE_EXTRACTED_DATA)
        
        # Call the method under test
        result = await analyzer.analyze_resume(SAMPLE_RESUME)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "contact_info" in result
        assert "skills" in result
        assert "experience" in result
        assert "education" in result
        
        # Verify the provider was called with the correct arguments
        mock_provider.generate.assert_awaited_once()
        
        # Verify the prompt template was used
        args, kwargs = mock_provider.generate.await_args
        assert "prompt" in kwargs
        assert SAMPLE_RESUME in kwargs["prompt"]
    
    @pytest.mark.asyncio
    async def test_analyze_resume_with_instructor(self, analyzer, mock_provider):
        """Test resume analysis with Instructor provider."""
        # Mock the provider to return a Pydantic model
        from pydantic import BaseModel
        
        class MockResumeData(BaseModel):
            contact_info: dict
            skills: list
            experience: list
            education: list
        
        # Mock the provider to return a Pydantic model
        mock_provider.generate.return_value = MockResumeData(
            contact_info=SAMPLE_EXTRACTED_DATA["contact_info"],
            skills=SAMPLE_EXTRACTED_DATA["skills"],
            experience=SAMPLE_EXTRACTED_DATA["experience"],
            education=SAMPLE_EXTRACTED_DATA["education"]
        )
        
        # Call the method with use_instructor=True
        result = await analyzer.analyze_resume(SAMPLE_RESUME, use_instructor=True)
        
        # Verify the result
        assert isinstance(result, dict)
        assert result["contact_info"]["name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_analyze_resume_retry_on_failure(self, analyzer, mock_provider):
        """Test that the analyzer retries on transient failures."""
        # Make the mock fail twice then succeed
        mock_provider.generate.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            json.dumps(SAMPLE_EXTRACTED_DATA)
        ]
        
        # Call with max_retries=3
        result = await analyzer.analyze_resume(SAMPLE_RESUME, max_retries=3)
        
        # Should eventually succeed after retries
        assert result is not None
        assert mock_provider.generate.await_count == 3
    
    @pytest.mark.asyncio
    async def test_analyze_resume_validation_error(self, analyzer, mock_provider):
        """Test that invalid data raises a validation error."""
        # Mock the provider to return invalid data (missing required fields)
        invalid_data = {"contact_info": {"name": "John"}}  # Missing required fields
        mock_provider.generate.return_value = json.dumps(invalid_data)
        
        # Should raise ValidationError
        with pytest.raises(ValidationError):
            await analyzer.analyze_resume(SAMPLE_RESUME)
    
    @pytest.mark.asyncio
    async def test_load_template_from_config(self, analyzer):
        """Test that templates are loaded from the configured path."""
        assert analyzer.prompt_template is not None
        assert "Extract information from:" in analyzer.prompt_template
    
    @pytest.mark.asyncio
    async def test_load_schema_from_config(self, analyzer):
        """Test that schemas are loaded from the configured path."""
        assert analyzer.resume_schema is not None
        assert "properties" in analyzer.resume_schema
    
    @pytest.mark.asyncio
    async def test_invalid_template_path(self, mock_provider, mock_config, tmp_path):
        """Test that missing template files raise an appropriate error."""
        # Create config with non-existent template path
        mock_config["analysis"]["templates"]["resume"]["extraction"] = str(tmp_path / "nonexistent.prompt")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ResumeAnalyzer(provider=mock_provider, config=mock_config)
    
    @pytest.mark.asyncio
    async def test_invalid_schema_path(self, mock_provider, mock_config, tmp_path):
        """Test that missing schema files raise an appropriate error."""
        # Create template file but no schema file
        template_dir = tmp_path / "templates" / "resume"
        template_dir.mkdir(parents=True, exist_ok=True)
        template_file = template_dir / "extraction_instructions.prompt"
        template_file.write_text("Test template")
        
        # Update config with non-existent schema path
        mock_config.update({
            "analysis": {
                "templates": {
                    "base_dir": str(tmp_path / "templates"),
                    "resume": {"extraction": str(template_file)}
                },
                "schemas": {
                    "base_dir": str(tmp_path / "schemas"),
                    "resume": str(tmp_path / "nonexistent_schema.json")
                }
            }
        })
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            ResumeAnalyzer(provider=mock_provider, config=mock_config)

    @pytest.mark.asyncio
    async def test_analyze_resume_with_custom_generation_params(self, analyzer, mock_provider):
        """Test that custom generation parameters are passed to the provider."""
        custom_params = {
            "temperature": 0.8,
            "max_tokens": 1000,
            "top_p": 0.95
        }
        
        # Call with custom parameters
        await analyzer.analyze_resume(SAMPLE_RESUME, generation_params=custom_params)
        
        # Verify the provider was called with the custom parameters
        _, kwargs = mock_provider.generate.await_args
        for key, value in custom_params.items():
            assert kwargs.get(key) == value

    @pytest.mark.asyncio
    async def test_analyze_resume_with_invalid_json_response(self, analyzer, mock_provider):
        """Test handling of invalid JSON response from the provider."""
        # Mock the provider to return invalid JSON
        mock_provider.generate.return_value = "This is not valid JSON"
        
        # Should raise JSONDecodeError wrapped in AnalysisError
        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.analyze_resume(SAMPLE_RESUME)
        
        assert "Failed to parse response as JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_resume_with_missing_required_fields(self, analyzer, mock_provider):
        """Test handling of response missing required fields."""
        # Mock the provider to return data missing required fields
        invalid_data = {
            "contact_info": {"name": "John Doe"},  # Missing other required fields
            "skills": [],
            "experience": [],
            "education": []
        }
        mock_provider.generate.return_value = json.dumps(invalid_data)
        
        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            await analyzer.analyze_resume(SAMPLE_RESUME)
        
        assert "Missing required fields" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
