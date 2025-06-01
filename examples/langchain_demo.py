"""
LangChain Integration Demo

This script demonstrates how to use the LangChain integration for resume analysis
and candidate matching.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from myjobspyai.analysis.components.factory import get_analyzer, get_matcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data
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

async def analyze_resume():
    """Demonstrate resume analysis."""
    logger.info("Starting resume analysis...")
    
    # Get an analyzer instance
    analyzer = get_analyzer()
    
    # Analyze the resume
    try:
        result = await analyzer.analyze(SAMPLE_RESUME)
        
        # Print results
        print("\n=== Resume Analysis Results ===")
        print(f"Summary: {result.summary}")
        print(f"\nSkills: {', '.join(result.skills[:5])}...")
        print(f"\nExperience Level: {result.experience_level or 'Not specified'}")
        print(f"Years of Experience: {result.years_experience or 'Not specified'}")
        
        if result.experience:
            print("\nMost Recent Experience:")
            exp = result.experience[0]
            print(f"  {exp.title} at {exp.company}")
            print(f"  {exp.start_date} - {exp.end_date or 'Present'}")
            
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        raise

async def match_candidate():
    """Demonstrate candidate matching."""
    logger.info("Starting candidate matching...")
    
    # Get a matcher instance
    matcher = get_matcher()
    
    # Match the candidate to the job
    try:
        result = await matcher.match(
            job_description=SAMPLE_JOB_DESCRIPTION,
            candidate_profile=SAMPLE_RESUME,
            required_skills=["Python", "AWS", "Docker"]
        )
        
        # Print results
        print("\n=== Candidate Matching Results ===")
        print(f"Overall Match Score: {result.overall_score}/10")
        print(f"Skill Match: {result.skill_match*100:.1f}%")
        print(f"Experience Match: {result.experience_match*100:.1f}%")
        
        if result.missing_skills:
            print("\nMissing Skills:")
            for skill in result.missing_skills:
                print(f"- {skill}")
                
        print(f"\nAnalysis: {result.explanation}")
        
    except Exception as e:
        logger.error(f"Error matching candidate: {e}")
        raise

async def main():
    """Run the demo."""
    try:
        # Run resume analysis
        await analyze_resume()
        
        # Run candidate matching
        await match_candidate()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
