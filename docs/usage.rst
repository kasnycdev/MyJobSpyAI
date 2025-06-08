Usage Guide
===========

This guide will help you get started with using MyJobSpyAI for analyzing job postings and resumes.

Basic Usage
-----------

Importing the Library
-------------------

.. code-block:: python

   from myjobspyai import JobAnalyzer, ResumeAnalyzer
   from myjobspyai.models import JobPosting, Resume

Analyzing a Job Posting
----------------------

.. code-block:: python

   # Create a job posting
   job_posting = JobPosting(
       title="Senior Python Developer",
       description="""
       We are looking for a Senior Python Developer with 5+ years of experience.
       Required skills: Python, Django, REST APIs, PostgreSQL.
       Nice to have: AWS, Docker, Kubernetes.
       """,
       company="Tech Corp Inc.",
       location="Remote",
   )

   # Analyze the job posting
   analyzer = JobAnalyzer()
   analysis = analyzer.analyze(job_posting)

   # View the analysis results
   print(f"Key Skills: {analysis.required_skills}")
   print(f"Experience Level: {analysis.experience_level}")
   print(f"Technical Requirements: {analysis.technical_requirements}")

Analyzing a Resume
------------------

.. code-block:: python

   # Create a resume
   resume = Resume(
       name="John Doe",
       title="Senior Python Developer",
       summary="5+ years of experience in Python development...",
       skills=["Python", "Django", "REST APIs", "PostgreSQL", "AWS"],
       experience=[
           {
               "title": "Senior Developer",
               "company": "Previous Corp",
               "duration": "3 years",
           },
           {"title": "Python Developer", "company": "Startup Inc.", "duration": "2 years"},
       ],
   )

   # Analyze the resume
   analyzer = ResumeAnalyzer()
   analysis = analyzer.analyze(resume)

   # View the analysis results
   print(f"Strengths: {analysis.strengths}")
   print(f"Areas for Improvement: {analysis.areas_for_improvement}")

Matching Resume to Job
----------------------

.. code-block:: python

   # Match resume to job
   match_analysis = analyzer.match_resume_to_job(resume, job_posting)

   # View matching results
   print(f"Match Score: {match_analysis.match_score}%")
   print(f"Matching Skills: {match_analysis.matching_skills}")
   print(f"Missing Skills: {match_analysis.missing_skills}")
   print(f"Recommendations: {match_analysis.recommendations}")

Advanced Usage
-------------

Customizing Analysis
-------------------

You can customize the analysis by passing configuration options:

.. code-block:: python

   from myjobspyai import AnalysisConfig

   config = AnalysisConfig(
       include_technical_skills=True,
       include_soft_skills=True,
       include_salary_estimates=False,
       detailed_analysis=True,
   )

   analyzer = JobAnalyzer(config=config)
   analysis = analyzer.analyze(job_posting)

Batch Processing
---------------

Analyze multiple job postings or resumes in batch:

.. code-block:: python

   # Batch analyze job postings
   job_postings = [job1, job2, job3]
   analyses = [analyzer.analyze(job) for job in job_postings]

   # Or use the batch method
   batch_analyses = analyzer.batch_analyze(job_postings)

Error Handling
--------------

Switch between different LLM providers:

.. code-block:: python

   from myjobspyai.providers import OllamaProvider, OpenAIConfig

   # Use OpenAI
   openai_config = OpenAIConfig(model="gpt-4", api_key="your-api-key")
   analyzer = JobAnalyzer(llm_provider=OpenAIProvider(openai_config))

   # Or use Ollama
   ollama_provider = OllamaProvider(
       model="llama3:instruct", base_url="http://localhost:11434"
   )
   analyzer = JobAnalyzer(llm_provider=ollama_provider)

Troubleshooting
--------------

### Common Issues

1. **LLM Provider Not Responding**
   - Check if the provider service is running
   - Verify API keys and base URLs
   - Check network connectivity

2. **Analysis Taking Too Long**
   - Try a smaller batch size
   - Reduce the complexity of the analysis
   - Check server load if using a remote provider

3. **Incorrect Analysis**
   - Verify the input data format
   - Check the model's context window size
   - Try adjusting temperature and other generation parameters

### Getting Help

For additional help, please refer to:
- :doc:`configuration` for setting up providers
- :doc:`examples` for more usage examples
- Open an issue on our `GitHub repository <https://github.com/kasnycdev/MyJobSpyAI>`_

Next Steps
----------
- :doc:`examples`: See more examples of using MyJobSpyAI
- :doc:`API reference <api/modules>`: Explore the full API reference

Command Line Interface
---------------------

.. code-block:: bash

   # Search for jobs
   myjobspyai search --search-term "Software Engineer" --location "Remote" --is-remote

   # Analyze jobs with a resume
   myjobspyai analyze --resume path/to/your/resume.pdf --jobs jobs.json

   # Get help
   myjobspyai --help

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   # Initialize with default settings
   client = MyJobSpyAI()

   # Search for jobs
   jobs = client.search_jobs(
       search_term="Software Engineer", location="Remote", is_remote=True
   )

   # Analyze jobs with a resume
   analysis = client.analyze_jobs_with_resume(
       resume_path="path/to/your/resume.pdf", jobs=jobs
   )

   # Save results
   client.save_results(analysis, "job_analysis_results.json")


Command Line Interface
---------------------

.. code-block:: bash

   # Search for jobs
   myjobspyai search --search-term "Software Engineer" --location "Remote" --is-remote

   # Analyze jobs with a resume
   myjobspyai analyze --resume path/to/your/resume.pdf --jobs jobs.json

   # Get help
   myjobspyai --help
