Examples
========

This page provides practical examples of how to use MyJobSpyAI for various job search and analysis tasks.

Basic Examples
-------------

1. Basic Job Analysis
-------------------

Analyze a single job posting to extract key information:

.. code-block:: python

   from myjobspyai import JobAnalyzer
   from myjobspyai.models import JobPosting

   # Create a job posting
   job = JobPosting(
       title="Senior Data Scientist",
       description="""
       We are looking for a Senior Data Scientist with expertise in machine learning.
       Requirements:
       - 5+ years of experience in data science
       - Strong Python and SQL skills
       - Experience with TensorFlow or PyTorch
       - Advanced degree in Computer Science or related field
       """,
       company="Data Insights Inc.",
       location="New York, NY",
   )

   # Analyze the job
   analyzer = JobAnalyzer()
   analysis = analyzer.analyze(job)

   print(f"Required Skills: {analysis.required_skills}")
   print(f"Experience Level: {analysis.experience_level}")
   print(f"Education Requirements: {analysis.education_requirements}")

2. Resume Analysis
----------------

Analyze a resume to identify strengths and areas for improvement:

.. code-block:: python

   from myjobspyai import ResumeAnalyzer
   from myjobspyai.models import Resume

   resume = Resume(
       name="Jane Smith",
       title="Data Scientist",
       summary="""
       4 years of experience in data science and machine learning.
       Strong background in Python, SQL, and data visualization.
       """,
       skills=["Python", "SQL", "Machine Learning", "Pandas", "NumPy", "Scikit-learn"],
       experience=[
           {
               "title": "Data Scientist",
               "company": "Tech Solutions Inc.",
               "duration": "2 years",
               "description": "Developed ML models for customer segmentation.",
           },
           {
               "title": "Junior Data Analyst",
               "company": "Data Insights Co.",
               "duration": "2 years",
               "description": "Performed data analysis and created dashboards.",
           },
       ],
       education=[
           {
               "degree": "MSc in Computer Science",
               "institution": "University of Technology",
               "year": 2019,
           }
       ],
   )

   analyzer = ResumeAnalyzer()
   analysis = analyzer.analyze(resume)

   print(f"Strengths: {analysis.strengths}")
   print(f"Areas for Improvement: {analysis.areas_for_improvement}")
   print(f"Skill Gaps: {analysis.skill_gaps}")

Intermediate Examples
-------------------

3. Job-Readiness Check
---------------------

Check how well a resume matches a specific job description:

.. code-block:: python

   from myjobspyai import JobAnalyzer, ResumeAnalyzer
   from myjobspyai.models import JobPosting, Resume

   # Job posting
   job = JobPosting(
       title="Machine Learning Engineer",
       description="""
       Looking for a Machine Learning Engineer with:
       - Strong Python programming skills
       - Experience with deep learning frameworks
       - Knowledge of cloud platforms (AWS/GCP)
       - Experience with MLOps tools
       """,
   )

   # Resume
   resume = Resume(
       skills=["Python", "Machine Learning", "AWS", "Docker"],
       experience=[
           {"title": "ML Engineer", "duration": "2 years"},
           {"title": "Data Scientist", "duration": "1 year"},
       ],
   )

   # Analyze and match
   job_analyzer = JobAnalyzer()
   resume_analyzer = ResumeAnalyzer()

   job_analysis = job_analyzer.analyze(job)
   resume_analysis = resume_analyzer.analyze(resume)

   # Get match score
   match = resume_analyzer.match_resume_to_job(resume, job)
   print(f"Match Score: {match.match_score}%")
   print(f"Missing Skills: {match.missing_skills}")
   print(f"Recommendations: {match.recommendations}")

### 4. Batch Processing

Analyze multiple job postings at once:

.. code-block:: python

   from myjobspyai import JobAnalyzer
   from myjobspyai.models import JobPosting

   # Create multiple job postings
   jobs = [
       JobPosting(
           title="Data Engineer",
           description="Looking for a data engineer with SQL and Python...",
       ),
       JobPosting(
           title="ML Engineer",
           description="Seeking ML engineer with TensorFlow experience...",
       ),
       JobPosting(title="Data Analyst", description="Junior data analyst position..."),
   ]

   # Batch analyze
   analyzer = JobAnalyzer()
   analyses = analyzer.batch_analyze(jobs)

   for job, analysis in zip(jobs, analyses):
       print(f"Job: {job.title}")
       print(f"  Required Skills: {analysis.required_skills[:3]}...")
       print(f"  Experience Level: {analysis.experience_level}")

Advanced Examples
----------------

### 4. Custom Analysis with Configuration
--------------------------------------

Customize the analysis with specific parameters:

.. code-block:: python

   from myjobspyai import JobAnalyzer, AnalysisConfig
   from myjobspyai.models import JobPosting

   config = AnalysisConfig(
       include_salary_estimates=True,
       include_company_analysis=True,
       detailed_technical_analysis=True,
       language="en",
   )

   analyzer = JobAnalyzer(config=config)
   job = JobPosting(title="Senior Developer", description="...")
   analysis = analyzer.analyze(job)

   print(f"Salary Estimate: {analysis.salary_estimate}")
   print(f"Technical Stack: {analysis.technical_stack}")

### 6. Integration with Job Boards

Fetch and analyze jobs from various sources:

.. code-block:: python

   from myjobspyai import JobBoardScraper, JobAnalyzer

   # Initialize scraper (example with LinkedIn)
   scraper = JobBoardScraper(source="linkedin")
   jobs = scraper.search_jobs("machine learning", location="Remote")

   # Analyze jobs
   analyzer = JobAnalyzer()
   for job in jobs[:3]:  # Analyze first 3 jobs
       analysis = analyzer.analyze(job)
       print(f"\nJob: {job.title}")
       print(f"Company: {job.company}")
       print(f"Key Skills: {analysis.required_skills[:5]}...")

Troubleshooting
--------------

### Common Issues

1. **No Results Returned**
   - Verify your internet connection
   - Check if the job board requires authentication
   - Ensure your search parameters are correct

2. **Analysis Errors**
   - Check the input format of your job postings/resumes
   - Verify that required fields are provided
   - Check the logs for detailed error messages

3. **Performance Issues**
   - For large batches, consider processing in smaller chunks
   - Use async/await for better performance with remote APIs

Next Steps
----------
- :doc:`usage`: Learn more about using MyJobSpyAI
- :doc:`configuration`: Configure advanced settings
- :doc:`api/modules`: Explore the full API reference

Basic Job Search
----------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   client = MyJobSpyAI()
   jobs = client.search_jobs(
       search_term="Data Scientist",
       location="New York, NY",
       is_remote=True,
       results_wanted=10,
   )
   print(f"Found {len(jobs)} jobs")

Resume Analysis
--------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   client = MyJobSpyAI()
   jobs = client.search_jobs("Machine Learning Engineer", "Remote")

   # Analyze jobs with your resume
   analysis = client.analyze_jobs_with_resume(
       resume_path="path/to/your/resume.pdf", jobs=jobs
   )

   # Print analysis results
   for job_analysis in analysis:
       print(f"Job: {job_analysis['job_title']}")
       print(f"Match Score: {job_analysis['match_score']}%")
       print("-" * 50)

Custom Configuration
-------------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI
   from pathlib import Path

   # Load custom config
   config_path = Path.home() / ".config" / "myjobspyai" / "config.yaml"

   client = MyJobSpyAI(config_path=config_path)
   # Use the client as usual...
