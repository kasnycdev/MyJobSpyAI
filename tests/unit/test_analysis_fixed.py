"""
Test script for job analysis functionality.

This script tests the job analysis pipeline with sample resume and job data,
verifying the integration with the LLM provider and the analysis results.
"""
# Standard library imports
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local application imports
from myjobspyai.analysis.main_matcher import analyze_jobs_async
from myjobspyai.analysis.models import ResumeData
from myjobspyai.config import LLMConfig, settings
from myjobspyai.utils.logging_utils import setup_logging

# Configure LLM settings with enhanced configuration
settings.llm = LLMConfig(
    provider="ollama",
    model="llama3:instruct",
    base_url="http://localhost:11434",
    temperature=0.7,
    max_tokens=2000,
    timeout=300,
    streaming=True,
    stream_chunk_size=1000,
)

# Initialize logging with enhanced configuration
setup_logging()
logger = logging.getLogger(__name__)

# Ensure logs directory exists
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Clear existing handlers to avoid duplicate logs
logger.handlers.clear()

# Configure file handler for test logs with timestamped filename
log_file = log_dir / f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Configure console handler with color support
class ColorFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColorFormatter())

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Suppress noisy library logs
for lib in ["urllib3", "httpx", "httpcore", "openai", "asyncio"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

def create_sample_resume() -> ResumeData:
    """Create a sample resume for testing with comprehensive data."""
    return ResumeData(
        contact_info={
            "name": "John A. Doe",
            "email": "john.doe@example.com",
            "phone": "+1 (555) 123-4567",
            "linkedin": "linkedin.com/in/johndoe",
            "location": "San Francisco, CA",
            "portfolio": "johndoe.dev"
        },
        summary=(
            "Senior Software Engineer with 6+ years of experience in building scalable "
            "web applications using Python and modern JavaScript frameworks. "
            "Specialized in backend development, API design, and cloud infrastructure. "
            "Proven track record of leading engineering teams and delivering high-quality software."
        ),
        experience=[
            {
                "job_title": "Senior Software Engineer",
                "company": "TechCorp Inc.",
                "location": "San Francisco, CA",
                "duration": "2020 - Present",
                "responsibilities": [
                    "Lead a cross-functional team of 5 developers in designing and implementing new features",
                    "Architected and deployed scalable microservices using Python, FastAPI, and Docker",
                    "Mentored junior developers and conducted code reviews to maintain code quality"
                ],
                "quantifiable_achievements": [
                    "Reduced API response time by 40% through query optimization and caching",
                    "Increased test coverage from 70% to 95% by implementing comprehensive test suites",
                    "Led migration from monolith to microservices, improving deployment frequency by 3x"
                ]
            },
            {
                "job_title": "Software Engineer",
                "company": "WebSolutions LLC",
                "location": "Austin, TX",
                "duration": "2018 - 2020",
                "responsibilities": [
                    "Developed and maintained RESTful APIs using Django REST Framework",
                    "Implemented frontend components using React and Redux",
                    "Collaborated with product team to define requirements and technical specifications"
                ],
                "quantifiable_achievements": [
                    "Improved application performance by 25% through database optimization",
                    "Reduced deployment time by 40% by implementing CI/CD pipelines"
                ]
            }
        ],
        education=[
            {
                "degree": "Master of Science in Computer Science",
                "institution": "Stanford University",
                "location": "Stanford, CA",
                "graduation_year": "2018",
                "gpa": "3.8/4.0",
                "honors": ["Summa Cum Laude", "Dean's List"]
            },
            {
                "degree": "Bachelor of Science in Computer Engineering",
                "institution": "University of California, Berkeley",
                "location": "Berkeley, CA",
                "graduation_year": "2016",
                "gpa": "3.7/4.0"
            }
        ],
        skills=[
            {"name": "Python", "level": "Expert", "years_experience": 6, "category": "Programming Languages"},
            {"name": "JavaScript/TypeScript", "level": "Advanced", "years_experience": 5, "category": "Programming Languages"},
            {"name": "Django/FastAPI", "level": "Expert", "years_experience": 5, "category": "Frameworks"},
            {"name": "React/Redux", "level": "Advanced", "years_experience": 4, "category": "Frontend"},
            {"name": "PostgreSQL", "level": "Advanced", "years_experience": 5, "category": "Databases"},
            {"name": "Docker/Kubernetes", "level": "Intermediate", "years_experience": 3, "category": "DevOps"},
            {"name": "AWS/GCP", "level": "Intermediate", "years_experience": 3, "category": "Cloud"},
            {"name": "CI/CD", "level": "Advanced", "years_experience": 4, "category": "DevOps"},
            {"name": "RESTful APIs", "level": "Expert", "years_experience": 5, "category": "APIs"},
            {"name": "Microservices", "level": "Advanced", "years_experience": 3, "category": "Architecture"}
        ],
        projects=[
            {
                "name": "E-commerce Platform",
                "description": "A full-stack e-commerce platform with product catalog, shopping cart, and payment integration.",
                "technologies_used": ["Python", "Django", "React", "PostgreSQL", "Stripe"],
                "role": "Lead Developer",
                "duration": "6 months",
                "impact": "Processed over 10,000+ orders in the first quarter"
            },
            {
                "name": "Task Management System",
                "description": "A collaborative task management tool with real-time updates and team collaboration features.",
                "technologies_used": ["FastAPI", "WebSockets", "Vue.js", "MongoDB"],
                "role": "Full-stack Developer",
                "duration": "3 months",
                "impact": "Adopted by 20+ teams across the organization"
            }
        ],
        certifications=[
            "AWS Certified Solutions Architect - Associate",
            "Google Cloud Professional Cloud Architect",
            "Certified Kubernetes Administrator (CKA)",
            "Python Institute Certified Professional"
        ],
        languages=[
            {"language": "English", "proficiency": "Native"}, 
            {"language": "Spanish", "proficiency": "Professional Working"},
            {"language": "French", "proficiency": "Elementary"}
        ]
    )

def create_sample_jobs() -> List[Dict[str, Any]]:
    """Create a list of sample job listings for testing."""
    return [
        {
            "job_id": "job123",
            "title": "Senior Python Developer",
            "company": "InnovateTech",
            "location": "San Francisco, CA (Hybrid)",
            "job_type": "Full-time",
            "salary": {
                "min": 140000,
                "max": 180000,
                "currency": "USD",
                "period": "year"
            },
            "description": (
                "We are looking for an experienced Python Developer to join our team. "
                "The ideal candidate will have 5+ years of experience with Python, Django, "
                "and cloud technologies. Experience with microservices and containerization "
                "is a plus. You will be responsible for designing and implementing "
                "scalable backend services and APIs."
            ),
            "requirements": {
                "required_skills": ["Python", "Django", "REST APIs", "PostgreSQL"],
                "preferred_skills": ["Docker", "Kubernetes", "AWS", "React"],
                "education": "Bachelor's degree in Computer Science or related field",
                "experience_years": 5,
                "certifications": ["AWS Certified Developer"]
            },
            "posted_date": "2023-05-15",
            "application_deadline": "2023-06-15",
            "application_url": "https://careers.innovatetech.com/senior-python-dev"
        },
        {
            "job_id": "job124",
            "title": "Full Stack Developer",
            "company": "WebCrafters",
            "location": "Remote",
            "job_type": "Full-time",
            "salary": {
                "min": 120000,
                "max": 160000,
                "currency": "USD",
                "period": "year"
            },
            "description": (
                "Join our team as a Full Stack Developer to build and maintain web applications. "
                "You'll work with a team of talented engineers to design and implement new features, "
                "optimize performance, and ensure code quality. Strong experience with React, Node.js, "
                "and modern JavaScript is required."
            ),
            "requirements": {
                "required_skills": ["JavaScript", "React", "Node.js", "Express"],
                "preferred_skills": ["TypeScript", "MongoDB", "GraphQL", "Docker"],
                "education": "Bachelor's degree or equivalent experience",
                "experience_years": 3,
                "certifications": []
            },
            "posted_date": "2023-05-20",
            "application_deadline": "2023-06-20",
            "application_url": "https://webcrafters.io/careers/full-stack-dev"
        }
    ]

# Create sample data instances
SAMPLE_RESUME = create_sample_resume()
SAMPLE_JOBS = create_sample_jobs()

def format_analysis_result(result: dict) -> str:
    """Format an analysis result for better readability."""
    try:
        if not isinstance(result, dict):
            return str(result)
            
        output = []
        
        # Basic job info
        if 'job_title' in result and 'company' in result:
            output.append(f"\n{'='*50}")
            output.append(f"Job: {result['job_title']} at {result.get('company', 'N/A')}")
            output.append(f"Match Score: {result.get('match_score', 0) * 100:.1f}%")
            output.append('-'*50)
        
        # Matching skills
        if 'matching_skills' in result and result['matching_skills']:
            output.append("\n✅ Matching Skills:")
            for skill in result['matching_skills'][:10]:  # Limit to top 10
                if isinstance(skill, dict):
                    skill_str = skill.get('name', str(skill))
                    if 'match_strength' in skill:
                        skill_str += f" (Match: {skill['match_strength']:.1%})"
                else:
                    skill_str = str(skill)
                output.append(f"  • {skill_str}")
        
        # Missing skills
        if 'missing_skills' in result and result['missing_skills']:
            output.append("\n❌ Missing Skills:")
            for skill in result['missing_skills'][:10]:  # Limit to top 10
                if isinstance(skill, dict):
                    skill_str = skill.get('name', str(skill))
                    if 'importance' in skill:
                        skill_str += f" (Importance: {skill['importance']})"
                else:
                    skill_str = str(skill)
                output.append(f"  • {skill_str}")
        
        # Experience match
        if 'experience_match' in result:
            exp = result['experience_match']
            if isinstance(exp, dict) and 'years' in exp and 'required' in exp:
                output.append(f"\n📊 Experience: {exp['years']} years (Required: {exp['required']} years)")
        
        # Additional analysis
        for key in ['summary', 'analysis', 'recommendations']:
            if key in result and result[key]:
                output.append(f"\n🔍 {key.capitalize()}:")
                output.append(f"   {result[key]}")
        
        return '\n'.join(output)
    except Exception as e:
        logger.warning(f"Error formatting analysis result: {e}")
        return str(result)

async def test_job_analysis():
    """Test job analysis with sample data."""
    try:
        logger.info("\n" + "="*50)
        logger.info("STARTING JOB ANALYSIS TEST")
        logger.info("="*50)
        
        # Log resume summary
        logger.info("\n📄 RESUME SUMMARY:")
        logger.info(f"Name: {SAMPLE_RESUME.contact_info.get('name', 'N/A')}")
        logger.info(f"Title: {SAMPLE_RESUME.summary.split('.')[0] if SAMPLE_RESUME.summary else 'N/A'}")
        logger.info(f"Experience: {len(SAMPLE_RESUME.experience or [])} positions")
        logger.info(f"Skills: {len(SAMPLE_RESUME.skills or [])} skills listed")
        logger.info(f"Education: {len(SAMPLE_RESUME.education or [])} degrees")
        
        # Run analysis
        logger.info("\n🔍 ANALYZING JOB MATCHES...")
        results = await analyze_jobs_async(SAMPLE_RESUME, SAMPLE_JOBS)
        
        # Process and log results
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS RESULTS")
        logger.info("="*50)
        
        all_results = []
        for job, result in zip(SAMPLE_JOBS, results):
            try:
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    result = {"analysis": str(result)}
                
                # Add job info to result
                result.update({
                    'job_title': job.get('title', 'Unknown Position'),
                    'company': job.get('company', 'Unknown Company'),
                })
                
                # Format and log the result
                formatted_result = format_analysis_result(result)
                logger.info(formatted_result)
                
                # Save the full result
                all_results.append({
                    'job_id': job.get('job_id'),
                    'job_title': job.get('title'),
                    'company': job.get('company'),
                    'match_score': result.get('match_score', 0),
                    'analysis': result
                })
                
            except Exception as e:
                logger.error(f"Error processing result for job {job.get('job_id', 'unknown')}: {e}", exc_info=True)
        
        # Save results to file
        test_results_dir = Path("test_results")
        test_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = test_results_dir / f"job_analysis_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'resume': SAMPLE_RESUME.dict(),
                'jobs': SAMPLE_JOBS,
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Analysis complete! Results saved to: {results_file}")
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Error during job analysis: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Run the test
        asyncio.run(test_job_analysis())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user.")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        sys.exit(1)
