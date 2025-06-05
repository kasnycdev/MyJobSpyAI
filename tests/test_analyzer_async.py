"""Test script to verify async analyzer initialization and basic functionality."""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myjobspyai.analysis.analyzer import create_analyzer, ResumeAnalyzer, JobAnalyzer
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


async def test_analyzer_initialization():
    """Test that analyzers can be initialized asynchronously."""
    print("Testing analyzer initialization...")

    try:
        # Test ResumeAnalyzer
        print("\nTesting ResumeAnalyzer...")
        resume_analyzer = await create_analyzer(ResumeAnalyzer)
        print(
            f"ResumeAnalyzer initialized with provider: {resume_analyzer.provider_name}"
        )

        # Test JobAnalyzer
        print("\nTesting JobAnalyzer...")
        job_analyzer = await create_analyzer(JobAnalyzer)
        print(f"JobAnalyzer initialized with provider: {job_analyzer.provider_name}")

        return True
    except Exception as e:
        print(f"Error initializing analyzers: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_resume_analysis():
    """Test resume analysis with a sample resume."""
    print("\nTesting resume analysis...")

    sample_resume = """
    John Doe
    Senior Software Engineer
    
    SUMMARY
    Experienced software engineer with 5+ years of experience in Python and web development.
    
    EXPERIENCE
    - Senior Software Engineer at Tech Corp (2020-Present)
      * Led a team of 5 developers
      * Developed microservices using Python and FastAPI
    
    SKILLS
    - Python, FastAPI, Django, JavaScript, React, AWS
    """

    try:
        analyzer = await create_analyzer(ResumeAnalyzer)
        print("Analyzing resume...")
        resume_data = await analyzer.extract_resume_data_async(sample_resume)
        print("Resume analysis successful!")
        print(f"Extracted name: {getattr(resume_data, 'name', 'Not found')}")
        print(f"Extracted skills: {getattr(resume_data, 'skills', [])}")
        return True
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_job_analysis():
    """Test job analysis with a sample job description."""
    print("\nTesting job analysis...")

    sample_job = """
    Job Title: Senior Python Developer
    
    We are looking for a Senior Python Developer with experience in web development.
    
    Requirements:
    - 5+ years of Python experience
    - Experience with FastAPI or Django
    - Knowledge of AWS services
    - Strong problem-solving skills
    
    Benefits:
    - Competitive salary
    - Remote work options
    - Health insurance
    """

    try:
        analyzer = await create_analyzer(JobAnalyzer)
        print("Analyzing job description...")
        job_data = await analyzer.extract_job_details_async(
            sample_job, "Senior Python Developer"
        )
        print("Job analysis successful!")
        print(f"Extracted requirements: {getattr(job_data, 'requirements', [])}")
        return True
    except Exception as e:
        print(f"Error analyzing job: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("Starting analyzer tests...")

    # Test initialization
    init_success = await test_analyzer_initialization()
    resume_success = await test_resume_analysis()
    job_success = await test_job_analysis()

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Analyzer Initialization: {'PASSED' if init_success else 'FAILED'}")
    print(f"Resume Analysis: {'PASSED' if resume_success else 'FAILED'}")
    print(f"Job Analysis: {'PASSED' if job_success else 'FAILED'}")

    return all([init_success, resume_success, job_success])


if __name__ == "__main__":
    asyncio.run(main())
