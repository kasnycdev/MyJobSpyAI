from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from .pipeline import RAGPipeline
from ..analysis.analyzer import JobAnalyzer

logger = logging.getLogger(__name__)

class JobRAGProcessor:
    """Processor for job and resume data using RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the job RAG processor."""
        self.config = config
        self.pipeline = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('TEXT_CHUNK_SIZE', 1000),
            chunk_overlap=config.get('TEXT_CHUNK_OVERLAP', 200)
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )
        self.analyzer = JobAnalyzer(config)
        
    async def initialize(self) -> None:
        """Initialize the RAG pipeline."""
        self.pipeline = await RAGPipeline.initialize(self.config)
        
    async def process_job_data(self, job_data: List[Dict[str, Any]]) -> None:
        """Process and store job data in RAG pipeline."""
        try:
            # Convert job data to documents
            documents = []
            for job in job_data:
                # Create document with structured metadata
                doc = Document(
                    page_content=f"""
                    Job Title: {job['title']}
                    Company: {job['company']}
                    Location: {job['location']}
                    Description: {job['description']}
                    Requirements: {job['requirements']}
                    """,
                    metadata={
                        'type': 'job',
                        'id': job['id'],
                        'title': job['title'],
                        'company': job['company'],
                        'location': job['location'],
                        'timestamp': datetime.now().isoformat(),
                        'source': job.get('source', 'unknown')
                    }
                )
                documents.append(doc)
            
            # Process documents through pipeline
            await self.pipeline.process_documents(documents)
            
        except Exception as e:
            logger.error(f"Error processing job data: {e}")
            raise
    
    async def process_resume_data(self, resume_data: Dict[str, Any]) -> None:
        """Process and store resume data in RAG pipeline."""
        try:
            # Extract key sections from resume
            sections = [
                'education', 'experience', 'skills', 'projects',
                'certifications', 'achievements'
            ]
            
            # Create documents for each section
            documents = []
            for section in sections:
                if section in resume_data:
                    doc = Document(
                        page_content=f"""
                        Resume Section: {section}
                        Content: {resume_data[section]}
                        """,
                        metadata={
                            'type': 'resume',
                            'section': section,
                            'candidate_id': resume_data['candidate_id'],
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            # Process documents through pipeline
            await self.pipeline.process_documents(documents)
            
        except Exception as e:
            logger.error(f"Error processing resume data: {e}")
            raise
    
    async def analyze_suitability(self, job_id: str, resume_id: str) -> Dict[str, Any]:
        """Analyze suitability between job and resume using RAG."""
        try:
            # Query relevant job data
            job_query = f"Job with id {job_id}"
            async for job_doc in self.pipeline.query(job_query, k=1):
                job_data = job_doc.metadata
                
            # Query relevant resume data
            resume_query = f"Resume sections for candidate {resume_id}"
            async for resume_doc in self.pipeline.query(resume_query, k=5):
                resume_data = resume_doc.metadata
                
            # Perform analysis using both RAG context and LLM
            analysis = await self.analyzer.analyze_suitability(
                job_data=job_data,
                resume_data=resume_data
            )
            
            return {
                'job': job_data,
                'resume': resume_data,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing suitability: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.pipeline:
            await self.pipeline.cleanup()
