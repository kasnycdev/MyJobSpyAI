"""
Model for job listings.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .job import JobType


@dataclass
class JobListing:
    """Data class representing a job listing."""

    title: str
    company: str
    location: str
    description: str
    job_type: JobType
    url: str
    posted_date: Optional[str] = None
    salary: Optional[str] = None
    remote: bool = False
    source: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the job listing to a dictionary."""
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "job_type": self.job_type.value,
            "url": self.url,
            "posted_date": self.posted_date,
            "salary": self.salary,
            "remote": self.remote,
            "source": self.source,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobListing":
        """Create a JobListing from a dictionary."""
        return cls(
            title=data.get("title", ""),
            company=data.get("company", ""),
            location=data.get("location", ""),
            description=data.get("description", ""),
            job_type=JobType(data.get("job_type", "other")),
            url=data.get("url", ""),
            posted_date=data.get("posted_date"),
            salary=data.get("salary"),
            remote=data.get("remote", False),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
        )
