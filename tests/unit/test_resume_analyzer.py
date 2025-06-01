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
    logger.info("=== Starting ResumeAnalyzer Test ===")
    
    # Enable debug logging for all loggers
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Enable debug logging for specific loggers
    for logger_name in ['myjobspyai', 'httpx', 'httpcore', 'urllib3']:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    # Log environment information
    logger.info("\n=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Log environment variables
    logger.info("\n=== Environment Variables ===")
    for key, value in os.environ.items():
        if key.startswith(('OLLAMA_', 'PYTHON', 'PATH', 'VIRTUAL_ENV')):
            logger.info(f"{key}: {value}")
    
    try:
        # Configuration for the analyzer
        config = {
            'provider': 'ollama',
            'debug': True,
            'log_level': 'DEBUG'  # Ensure debug logging is enabled
        }
        
        # Log environment variables for debugging
        logger.debug("=== Environment Variables ===")
        for key, value in os.environ.items():
            if key.startswith(('OLLAMA_', 'PYTHON', 'PATH')):
                logger.debug(f"  {key}: {value}")
        
        # Check Python path
        logger.debug("\n=== Python Path ===")
        for i, path in enumerate(sys.path):
            logger.debug(f"  {i}: {path}")
                
        # Check if Ollama is running
        try:
            import requests
            ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            logger.debug(f"Checking Ollama service at {ollama_url}")
            
            # Check if Ollama is running
            try:
                response = requests.get(f"{ollama_url}/api/version", timeout=5)
                response.raise_for_status()
                logger.info(f"Ollama is running at {ollama_url}")
                logger.debug(f"Ollama version: {response.text}")
                
                # List available models
                models_response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                models_response.raise_for_status()
                models = models_response.json().get('models', [])
                logger.debug("\n=== Available Ollama Models ===")
                for model in models[:10]:  # Show first 10 models to avoid huge logs
                    logger.debug(f"  - {model.get('name', 'N/A')} (size: {model.get('size', 0)/1e9:.1f}GB)")
                if len(models) > 10:
                    logger.debug(f"  ... and {len(models) - 10} more models")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to Ollama at {ollama_url}")
                logger.error(f"Error details: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                raise
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise RuntimeError("Ollama is not running or not accessible. Please start Ollama and try again.") from e
        
        # Use llama3:instruct model which is optimized for instruction following
        ollama_model = 'llama3:instruct'
        logger.info(f"Using Ollama model: {ollama_model}")
        
        # Check if the model is available
        try:
            import requests
            ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            # First, check if model exists
            logger.debug(f"Checking if model {ollama_model} exists...")
            response = requests.post(
                f"{ollama_url}/api/show",
                json={"name": ollama_model},
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"Model {ollama_model} is available")
                logger.debug(f"Model details: {json.dumps(model_info, indent=2)}")
            else:
                logger.warning(f"Model {ollama_model} not found (status: {response.status_code})")
                logger.debug(f"Response: {response.text}")
                
                # Try to pull the model if not found
                logger.info(f"Attempting to pull model {ollama_model}...")
                pull_response = requests.post(
                    f"{ollama_url}/api/pull",
                    json={"name": ollama_model},
                    stream=True,
                    timeout=300  # 5 minutes timeout for pulling
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model {ollama_model}")
                    # Verify the model is now available
                    response = requests.post(
                        f"{ollama_url}/api/show",
                        json={"name": ollama_model},
                        timeout=10
                    )
                    response.raise_for_status()
                    logger.info(f"Model {ollama_model} is now available")
                else:
                    raise RuntimeError(f"Failed to pull model {ollama_model}: {pull_response.status_code} {pull_response.text}")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking model {ollama_model}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise RuntimeError(f"Failed to verify model {ollama_model}") from e
            
        # Add Ollama-specific configuration
        config.update({
            'model': ollama_model,
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'timeout': int(os.getenv('OLLAMA_REQUEST_TIMEOUT', '300')),
                'max_retries': int(os.getenv('OLLAMA_MAX_RETRIES', '3')),
                'model': ollama_model
            },
            'model_kwargs': {
                'temperature': 0.1,  # Lower temperature for more deterministic output
                'top_p': 0.9,
                'max_tokens': 2000,
                'stop': ['<|eot_id|>', '<|end_of_text|>']  # Stop tokens for llama3
            },
            'max_retries': 3,
            'request_timeout': 300
        })
        
        # Log the full configuration
        logger.info("=== Analyzer Configuration ===")
        for key, value in config.items():
            if key != 'ollama':
                logger.info(f"  {key}: {value}")
            else:
                logger.info("  ollama: {" + ", ".join(f"{k}={v}" for k, v in value.items()) + "}")
        
        # Ensure the Ollama service is running
        logger.info("Verifying Ollama service is running...")
        
        # Print the configuration
        print_config(config)
        
        # Log configuration
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
            
        logger.info("Creating ResumeAnalyzer instance...")
        
        # Log the configuration that will be used
        logger.debug("=== ANALYZER CONFIGURATION ===")
        logger.debug(f"Provider: ollama")
        logger.debug(f"Model: {ollama_model}")
        logger.debug(f"Config: {json.dumps(config, indent=2, default=str)}")
        
        try:
            # Create the analyzer
            analyzer = ResumeAnalyzer(provider="ollama", model=ollama_model, config=config)
            
            # Log the analyzer instance details
            logger.info("Successfully created ResumeAnalyzer instance")
            logger.debug(f"Analyzer class: {analyzer.__class__.__name__}")
            logger.debug(f"Analyzer provider: {getattr(analyzer, 'provider', 'N/A')}")
            logger.debug(f"Analyzer model: {getattr(analyzer, 'model', 'N/A')}")
            
            # Check if the provider was properly initialized
            if hasattr(analyzer, '_provider_instance'):
                provider = analyzer._provider_instance
                logger.debug(f"Provider instance: {provider}")
                logger.debug(f"Provider class: {provider.__class__.__name__}")
                
                # Check if the provider has the required methods
                required_methods = ['generate', 'async_generate']
                for method in required_methods:
                    has_method = hasattr(provider, method)
                    logger.debug(f"Provider has method '{method}': {has_method}")
                    if not has_method:
                        logger.warning(f"Provider is missing required method: {method}")
            else:
                logger.warning("Analyzer does not have a provider instance")
                
        except Exception as e:
            logger.error(f"Failed to create ResumeAnalyzer: {str(e)}", exc_info=True)
            raise
        
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
        
        # Log the test resume for debugging
        logger.debug("=== TEST RESUME ===")
        logger.debug(test_resume)
        logger.debug("===================")
        
        # Log the prompt that will be sent to the LLM
        logger.debug("=== ANALYZER CONFIGURATION ===")
        try:
            # Log analyzer configuration
            logger.debug(f"Analyzer class: {analyzer.__class__.__name__}")
            logger.debug(f"Analyzer provider: {getattr(analyzer, 'provider', 'N/A')}")
            logger.debug(f"Analyzer model: {getattr(analyzer, 'model', 'N/A')}")
            
            # Access the prompt template directly for debugging
            prompt_template = getattr(analyzer, 'prompt_template', None)
            if prompt_template:
                logger.debug("\n=== PROMPT TEMPLATE ===")
                # Log just the first 500 chars to avoid huge logs
                logger.debug(f"Prompt template preview:\n{prompt_template[:500]}...")
                
                # Check for required placeholders
                required_placeholders = ['resume_text']
                missing_placeholders = [p for p in required_placeholders if f'{{{p}}}' not in prompt_template]
                if missing_placeholders:
                    logger.warning(f"Prompt template is missing required placeholders: {', '.join(missing_placeholders)}")
                else:
                    logger.debug("All required placeholders found in prompt template")
            else:
                logger.warning("No prompt template found in analyzer")
                
            # Log model configuration
            logger.debug("\n=== MODEL CONFIGURATION ===")
            model_config = getattr(analyzer, 'config', {})
            logger.debug(f"Model config: {json.dumps(model_config, indent=2, default=str)}")
            
            # Check if the model is callable
            if hasattr(analyzer, '_provider_instance'):
                logger.debug("Provider instance is available")
                provider = analyzer._provider_instance
                logger.debug(f"Provider class: {provider.__class__.__name__}")
                
                # Try to get model info if available
                if hasattr(provider, 'get_model_info'):
                    try:
                        model_info = provider.get_model_info()
                        logger.debug(f"Model info: {json.dumps(model_info, indent=2, default=str)}")
                    except Exception as e:
                        logger.warning(f"Could not get model info: {str(e)}")
            else:
                logger.warning("No provider instance found in analyzer")
                
        except Exception as e:
            logger.error(f"Error analyzing analyzer configuration: {str(e)}", exc_info=True)
        
        logger.info("Starting resume analysis...")
        logger.debug(f"Resume text length: {len(test_resume)} characters")
        
        try:
            # Run the analysis
            logger.debug("Calling analyzer.analyze()...")
            result = await analyzer.analyze(test_resume)
            
            # Log the result
            logger.info("Analysis completed successfully")
            logger.info("\n=== Analysis Result ===")
            
            # Convert result to dict for logging
            try:
                result_dict = result.model_dump()
                logger.info(json.dumps(result_dict, indent=2))
                
                # Log each section for better debugging
                logger.debug("=== Result Details ===")
                for section in ['contact_info', 'summary', 'skills', 'experience', 'education']:
                    if hasattr(result, section):
                        logger.debug(f"{section}: {json.dumps(getattr(result, section), indent=2, default=str)}")
                    else:
                        logger.warning(f"Missing section in result: {section}")
                
            except Exception as e:
                logger.error(f"Error formatting result: {str(e)}", exc_info=True)
                logger.debug(f"Raw result object: {result}")
                raise
                
            logger.info("======================")
            
            # Basic validation of the result
            assert result is not None, "Result should not be None"
            
            # Check for required fields
            required_fields = ['contact_info', 'summary', 'skills', 'experience', 'education']
            missing_fields = [field for field in required_fields if not hasattr(result, field)]
            
            if missing_fields:
                error_msg = f"Result is missing required fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                raise AssertionError(error_msg)
                
            # Log field details for debugging
            logger.debug("=== Field Details ===")
            for field in required_fields:
                value = getattr(result, field, None)
                logger.debug(f"{field}: {value is not None}")
            
            logger.info("All assertions passed")
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error("Error during resume analysis", exc_info=True)
        print(f"\nError: {type(e).__name__}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise

def main():
    """Main entry point for the test script."""
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Enable debug logging for specific loggers
    for logger_name in ['myjobspyai', 'httpx', 'httpcore', 'urllib3', 'asyncio']:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    try:
        # Log environment information
        logger.info("\n=== Environment Information ===")
        
        logger = logging.getLogger(__name__)
        
        logger.info("=== Starting Resume Analyzer Test ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Log environment variables
        logger.debug("Environment Variables:")
        for key, value in os.environ.items():
            if key.startswith(('OLLAMA_', 'PYTHON', 'PATH')):
                logger.debug(f"  {key}: {value}")
        
        # Ensure the event loop is properly configured for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Create and configure event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the test
            logger.info("Starting test execution...")
            result = loop.run_until_complete(test_resume_analyzer())
            
            if result is not None:
                logger.info("Test completed successfully with result")
                # Print a summary of the result
                try:
                    result_dict = result.model_dump()
                    print("\n=== Test Result Summary ===")
                    print(f"Contact Info: {result_dict.get('contact_info', {}).get('name', 'N/A')} ({result_dict.get('contact_info', {}).get('email', 'N/A')})")
                    print(f"Summary: {result_dict.get('summary', 'N/A')[:100]}..." if result_dict.get('summary') else "No summary")
                    print(f"Skills: {', '.join(skill['name'] for skill in result_dict.get('skills', [])[:5])}...")
                    print(f"Experience: {len(result_dict.get('experience', []))} positions")
                    print(f"Education: {len(result_dict.get('education', []))} entries")
                except Exception as e:
                    logger.error(f"Error formatting result summary: {str(e)}")
                    logger.debug("Full result object: %s", result, exc_info=True)
            else:
                logger.warning("Test completed but returned None result")
                
            logger.info("=== Test Completed Successfully ===")
            return 0
            
        except asyncio.CancelledError:
            logger.warning("Test was cancelled")
            return 1
            
        except Exception as e:
            logger.exception("Test failed with unexpected error")
            return 1
            
        finally:
            # Clean up the event loop
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                
            except Exception as e:
                logger.error("Error during cleanup: %s", str(e), exc_info=True)
                
            finally:
                if not loop.is_closed():
                    loop.close()
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Test failed with unexpected error")
        return 1

if __name__ == "__main__":
    sys.exit(main())
