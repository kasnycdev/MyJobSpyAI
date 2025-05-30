from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
from datetime import datetime
import logging
from contextlib import asynccontextmanager
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .milvus import pipeline_context, MilvusVectorStore
from .rag_processor import RAGProcessor
from .text_processor import TextProcessor
from .job_rag_processor import JobRAGProcessor

logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG pipeline implementation with Milvus vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG pipeline."""
        self.config = config
        self.pipeline_id = config.get('PIPELINE_ID', f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._initialize_components()
        self.job_processor = None
        
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Initialize text processor
        self.text_processor = TextProcessor(
            chunk_size=self.config.get('TEXT_CHUNK_SIZE', 1000),
            chunk_overlap=self.config.get('TEXT_CHUNK_OVERLAP', 200)
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('TEXT_CHUNK_SIZE', 1000),
            chunk_overlap=self.config.get('TEXT_CHUNK_OVERLAP', 200)
        )
        
        # Initialize job-specific processor
        self.job_processor = JobRAGProcessor(self.config)
    
    async def process_documents(self, documents: List[Document]) -> None:
        """Process and store documents in vector store."""
        async with pipeline_context(self.config) as vector_store:
            try:
                # Process documents in chunks
                chunk_size = 100
                for i in range(0, len(documents), chunk_size):
                    chunk = documents[i:i + chunk_size]
                    
                    # Get embeddings
                    texts = [doc.page_content for doc in chunk]
                    embeddings = await self.embeddings.aembed_documents(texts)
                    
                    # Add to vector store with optimized job-specific metadata
                    await vector_store.aadd_documents(
                        chunk,
                        embeddings,
                        metadata={
                            'pipeline_id': self.pipeline_id,
                            'timestamp': datetime.now().isoformat(),
                            'type': 'job' if any(doc.metadata.get('type') == 'job' for doc in chunk) else 'general'
                        }
                    )
                    
                logger.info(f"Successfully processed {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                raise
    
    async def query(self, query: str, k: int = 5, document_type: Optional[str] = None) -> AsyncGenerator[Document, None]:
        """Query the RAG pipeline with optional document type filtering."""
        async with pipeline_context(self.config) as vector_store:
            try:
                # Get query embedding
                query_embedding = await self.embeddings.aembed_query(query)
                
                # Perform similarity search with optional type filtering
                results = await vector_store.asimilarity_search(
                    query_embedding,
                    k=k,
                    pipeline_id=self.pipeline_id,
                    filter={'type': document_type} if document_type else None
                )
                
                for doc in results:
                    yield doc
                    
            except Exception as e:
                logger.error(f"Error querying pipeline: {e}")
                raise
    
    async def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        async with pipeline_context(self.config) as vector_store:
            try:
                await vector_store.cleanup_old_data()
                logger.info("Pipeline cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                raise
    
    @staticmethod
    async def initialize(config: Dict[str, Any]) -> "RAGPipeline":
        """Initialize the RAG pipeline with proper setup."""
        try:
            # Initialize Milvus
            await initialize_milvus(config)
            
            # Create pipeline instance
            pipeline = RAGPipeline(config)
            
            # Perform health check
            health = await health_check()
            if health['status'] != 'healthy':
                raise Exception(f"Milvus health check failed: {health['error']}")
                
            logger.info("RAG pipeline initialized successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise

async def process_ingress_data(
    documents: List[Document],
    config: Dict[str, Any]
) -> None:
    """Process incoming data and store in vector store."""
    try:
        pipeline = await RAGPipeline.initialize(config)
        await pipeline.process_documents(documents)
        await pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Error processing ingress data: {e}")
        raise

async def query_egress_data(
    query: str,
    config: Dict[str, Any],
    k: int = 5
) -> AsyncGenerator[Document, None]:
    """Query stored data for egress."""
    try:
        pipeline = await RAGPipeline.initialize(config)
        async for doc in pipeline.query(query, k):
            yield doc
        await pipeline.cleanup()
        
    except Exception as e:
        logger.error(f"Error querying egress data: {e}")
        raise

async def health_check() -> Dict[str, Any]:
    """Perform health check on pipeline components."""
    try:
        # Check Milvus connection
        milvus_health = await milvus.health_check()
        
        # Check embeddings model
        embeddings_health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "overall_status": "healthy",
            "components": {
                "milvus": milvus_health,
                "embeddings": embeddings_health
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
