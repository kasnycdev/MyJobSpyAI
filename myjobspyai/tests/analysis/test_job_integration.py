"""Tests for the job integration module."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from myjobspyai.analysis.job_integration import JobIntegration

class TestJobIntegration(unittest.TestCase):
    """Test cases for JobIntegration class."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = [
            {
                "title": "Software Engineer",
                "description": "Develop and maintain software applications.",
                "company": "Tech Corp",
                "location": "Remote",
                "required_skills": ["Python", "Django"],
                "preferred_skills": ["AWS", "Docker"],
                "min_experience": 3,
                "salary": "$100,000 - $150,000"
            },
            {
                "title": "Data Scientist",
                "description": "Analyze data and build ML models.",
                "company": "Data Insights",
                "location": "New York, NY",
                "required_skills": ["Python", "Pandas", "Scikit-learn"],
                "preferred_skills": ["TensorFlow", "PyTorch"],
                "min_experience": 5,
                "salary": "$120,000 - $160,000"
            }
        ]
        
        # Create a temporary file with test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test_jobs.json"
        with open(self.test_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f)
    
    def tearDown(self):
        """Clean up test data."""
        self.temp_dir.cleanup()
    
    def test_load_jobs(self):
        """Test loading jobs from a file."""
        job_integration = JobIntegration()
        self.assertTrue(job_integration.load_jobs(str(self.test_file)))
        self.assertEqual(len(job_integration), 2)
    
    def test_get_job(self):
        """Test getting a job by ID."""
        job_integration = JobIntegration()
        job_integration.load_jobs(str(self.test_file))
        
        # Get the first job
        jobs = job_integration.get_jobs()
        if jobs:  # Check if jobs were loaded
            job_id = jobs[0].get('job_id')
            if job_id:  # Check if job_id exists
                job = job_integration.get_job(job_id)
                self.assertIsNotNone(job)
                self.assertEqual(job.get('job_title_extracted'), "Software Engineer")
    
    def test_get_jobs_by_title(self):
        """Test getting jobs by title."""
        job_integration = JobIntegration()
        job_integration.load_jobs(str(self.test_file))
        
        jobs = job_integration.get_jobs_by_title("Software")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].get('job_title_extracted'), "Software Engineer")
    
    def test_get_jobs_by_company(self):
        """Test getting jobs by company."""
        job_integration = JobIntegration()
        job_integration.load_jobs(str(self.test_file))
        
        jobs = job_integration.get_jobs_by_company("Data")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].get('company_name'), "Data Insights")

if __name__ == "__main__":
    unittest.main()
