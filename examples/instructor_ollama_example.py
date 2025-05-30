"""
Example script demonstrating how to use the InstructorOllamaClient for structured outputs.

This example shows how to extract structured data from text using Ollama with Instructor.
"""
import asyncio
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the InstructorOllamaClient
from myjobspyai.analysis.providers.instructor_ollama import InstructorOllamaClient

# Define your Pydantic model for structured output
class ResumeData(BaseModel):
    """Structured resume data model."""
    name: str = Field(..., description="The person's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    experience: List[dict] = Field(
        default_factory=list, 
        description="List of work experiences, each with company, title, and duration"
    )
    education: List[dict] = Field(
        default_factory=list,
        description="List of education entries, each with institution, degree, and year"
    )

async def main():
    """Main function to demonstrate InstructorOllamaClient usage."""
    # Example resume text (in a real application, this would come from a file or input)
    resume_text = """
    John Doe
    john.doe@example.com | (123) 456-7890 | New York, NY
    
    SKILLS:
    - Python, JavaScript, SQL
    - Machine Learning, Data Analysis
    - AWS, Docker, Kubernetes
    
    EXPERIENCE:
    
    Senior Software Engineer
    Tech Corp, New York, NY
    June 2020 - Present
    - Led a team of 5 developers to build a new microservice architecture
    - Improved system performance by 40% through optimization
    
    Software Developer
    Web Solutions Inc, Boston, MA
    May 2018 - May 2020
    - Developed and maintained web applications using React and Node.js
    - Implemented CI/CD pipelines using GitHub Actions
    
    EDUCATION:
    
    Bachelor of Science in Computer Science
    University of Technology, Boston, MA
    Graduated: May 2018
    GPA: 3.8/4.0
    """
    
    # Configure the client
    config = {
        "model": "llama3:instruct",  # or any other Ollama model
        "base_url": "http://localhost:11434/v1"
    }
    
    # Create the client
    client = InstructorOllamaClient[ResumeData](config)
    
    try:
        # Generate structured output
        logger.info("Extracting structured data from resume...")
        result = await client.generate(
            prompt=f"Extract the following information from this resume in JSON format:\n\n{resume_text}",
            response_model=ResumeData,
            temperature=0.1  # Lower temperature for more deterministic output
        )
        
        # Print the structured result
        print("\nStructured Resume Data:")
        print("=" * 50)
        print(f"Name: {result.name}")
        print(f"Email: {result.email}")
        print(f"Phone: {result.phone}")
        
        print("\nSkills:")
        for skill in result.skills:
            print(f"- {skill}")
            
        print("\nExperience:")
        for exp in result.experience:
            print(f"- {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})")
        
        print("\nEducation:")
        for edu in result.education:
            print(f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})")
            
    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
    finally:
        # Clean up resources
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
