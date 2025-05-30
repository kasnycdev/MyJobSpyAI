import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Set up basic logging immediately
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger.info(f"Python path: {sys.path}")

# Import MyJobSpyAI components
try:
    from myjobspyai.analysis.resume_analyzer import ResumeAnalyzer
    from myjobspyai.utils.logging_utils import setup_logging
    
    # Configure application logging
    setup_logging()
    logger.info("Successfully imported MyJobSpyAI components")
except ImportError as e:
    logger.exception("Failed to import MyJobSpyAI components")
    raise

# Log the current configuration
def print_config(config):
    print("\n=== Current Configuration ===")
    for key, value in config.items():
        if key != 'ollama':
            print(f"{key}: {value}")
        else:
            print("ollama: {" + ", ".join(f"{k}={v}" for k, v in value.items()) + "}")
    print("===========================\n")

async def test_resume_analyzer():
    """Test the ResumeAnalyzer with a sample resume."""
    logger.info("Starting ResumeAnalyzer test")
    
    try:
        # Configuration for the analyzer
        config = {
            'provider': 'ollama',
            'debug': True
        }
        
        # Get model from environment or use a default
        ollama_model = os.getenv('OLLAMA_MODEL')
        if not ollama_model:
            raise ValueError("OLLAMA_MODEL environment variable not set. Please set it to the model you want to use.")
            
        # Add Ollama-specific configuration
        config.update({
            'model': ollama_model,
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'timeout': int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300')),
                'max_retries': int(os.getenv('OLLAMA_MAX_RETRIES', '3')),
                'model': ollama_model
            }
        })
        
        # Ensure the Ollama service is running
        logger.info("Verifying Ollama service is running...")
        
        # Print the configuration
        print_config(config)
        
        # Log configuration
        logger.info("Configuration:")
        for key, value in config.items():
            if key != 'ollama':
                logger.info(f"  {key}: {value}")
            else:
                logger.info("  ollama: {base_url=}, {timeout=}, {max_retries=}".format(**value))
        
        # Create the analyzer
        logger.info("Creating ResumeAnalyzer instance...")
        analyzer = ResumeAnalyzer(config=config)
        logger.info("ResumeAnalyzer created successfully")
        
        # Log analyzer details
        logger.info("ResumeAnalyzer details:")
        logger.info(f"  Model name: {getattr(analyzer, 'model_name', 'N/A')}")
        logger.info(f"  Config: {getattr(analyzer, 'config', 'N/A')}")
        logger.info(f"  Response model: {getattr(analyzer, 'RESPONSE_MODEL', 'N/A')}")
        logger.info(f"  Required fields: {getattr(analyzer, 'REQUIRED_FIELDS', 'N/A')}")
        
        # Test with a simple resume text
        test_resume = """
        John Doe
        Senior Software Engineer
        
        Contact:
        - Email: john.doe@example.com
        - Phone: (555) 123-4567
        - LinkedIn: linkedin.com/in/johndoe
        
        Summary:
        Experienced software engineer with 5+ years of experience in building scalable 
        web applications using Python, Docker, and cloud technologies.
        
        Skills:
        - Programming: Python, JavaScript, TypeScript
        - Cloud: AWS, Docker, Kubernetes
        - Databases: PostgreSQL, MongoDB, Redis
        - Frameworks: FastAPI, Django, React
        - Machine Learning: Scikit-learn, TensorFlow, PyTorch
        
        Experience:
        
        Senior Software Engineer
        Tech Company Inc. | 2020 - Present
        - Led a team of 5 developers to build a scalable microservices architecture
        - Implemented CI/CD pipelines reducing deployment time by 60%
        - Designed and developed RESTful APIs using FastAPI and PostgreSQL
        - Mentored junior developers and conducted code reviews
        
        Software Engineer
        Startup Co. | 2018 - 2020
        - Developed and maintained web applications using Django and React
        - Optimized database queries, improving performance by 40%
        - Implemented automated testing with 90% code coverage
        
        Education:
        
        B.S. in Computer Science
        University of Technology | 2014 - 2018
        - GPA: 3.8/4.0
        - Relevant Coursework: Algorithms, Data Structures, Machine Learning, 
          Software Engineering, Database Systems
        """
        
        logger.info("Starting resume analysis...")
        logger.debug(f"Resume text length: {len(test_resume)} characters")
        
        # Call the analyze method
        logger.info("Calling analyzer.analyze()...")
        result = await analyzer.analyze(test_resume)
        
        # Log the result
        logger.info("Analysis completed successfully")
        
        # Convert result to dict for pretty printing
        if hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = dict(result)
            
        print("\n=== Analysis Result ===")
        print(json.dumps(result_dict, indent=2, default=str))
        
        return result
        
    except Exception as e:
        logger.error("Error during resume analysis", exc_info=True)
        print(f"\nError: {type(e).__name__}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting test script")
        asyncio.run(test_resume_analyzer())
        logger.info("Test script completed successfully")
    except Exception as e:
        logger.exception("Test script failed with error")
        print(f"Error: {e}")
        sys.exit(1)
