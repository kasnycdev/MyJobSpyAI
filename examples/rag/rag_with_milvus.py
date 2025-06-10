"""
Example of using the RAG pipeline with Milvus vector store.

This script demonstrates how to:
1. Set up a RAG pipeline with Milvus as the vector store
2. Add documents to the vector store
3. Query the RAG pipeline
4. Generate responses using the retrieved context
"""

import logging
import os

from dotenv import load_dotenv

from myjobspyai.rag import RAGConfig, RAGPipeline
from myjobspyai.rag.embeddings import EmbeddingConfig
from myjobspyai.rag.loader import DocumentLoaderConfig
from myjobspyai.rag.splitter import TextSplitterConfig
from myjobspyai.rag.vector_store import MilvusVectorStoreConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the RAG pipeline example with Milvus."""
    # Get Milvus connection details from environment variables
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    milvus_db = os.getenv("MILVUS_DB", "default")

    if not milvus_uri or not milvus_token:
        raise ValueError(
            "MILVUS_URI and MILVUS_TOKEN environment variables must be set"
        )

    # Configure the RAG pipeline
    config = RAGConfig(
        # Document loading configuration
        document_loader=DocumentLoaderConfig(
            file_path="",  # Not needed since we're using inline documents
            loader_type="text",
            metadata={"source": "inline"},
        ),
        # Text splitting configuration
        text_splitter=TextSplitterConfig(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n\n",
        ),
        # Embedding configuration
        embedding=EmbeddingConfig(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        ),
        # Vector store configuration
        vector_store=MilvusVectorStoreConfig(
            type="milvus",
            collection_name="myjobspyai_rag_demo",
            embedding_dimension=768,  # Dimension of the embeddings
            distance_metric="cosine",
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_db=milvus_db,
            milvus_secure=True,  # Assuming we're using a secure connection
            milvus_drop_old=True,  # For demo purposes, drop existing collection
        ),
        # Retrieval configuration
        retrieval={
            "top_k": 3,
            "search_kwargs": {"k": 3},
        },
    )

    # Initialize the RAG pipeline
    logger.info("Initializing RAG pipeline with Milvus...")
    rag = RAGPipeline(config)

    # Example documents to add
    example_docs = [
        {
            "content": "MyJobSpyAI is a tool for job search and resume analysis. It helps job seekers find relevant positions and optimize their resumes.",
            "metadata": {"source": "example", "page": 1},
        },
        {
            "content": "The platform uses AI to match job seekers with relevant job postings based on their skills, experience, and preferences.",
            "metadata": {"source": "example", "page": 2},
        },
        {
            "content": "MyJobSpyAI supports multiple job boards and resume formats, making it easy to apply to multiple positions quickly.",
            "metadata": {"source": "example", "page": 3},
        },
        {
            "content": "The tool provides personalized job recommendations and resume feedback to help users stand out to employers.",
            "metadata": {"source": "example", "page": 4},
        },
        {
            "content": "With advanced search filters and AI-powered matching, MyJobSpyAI streamlines the job search process.",
            "metadata": {"source": "example", "page": 5},
        },
    ]

    # Add documents to the vector store
    logger.info("Adding documents to the vector store...")
    doc_ids = rag.vector_store.add_documents(example_docs)
    logger.info("Added %d documents with IDs: %s", len(doc_ids), doc_ids)

    # Example queries
    queries = [
        "What is MyJobSpyAI?",
        "How does the AI matching work?",
        "What features does the platform offer?",
    ]

    for query in queries:
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing query: {query}")
        logger.info("=" * 80)
        logger.info("Querying: %s", query)

        # Retrieve relevant documents
        logger.info("\nRetrieving relevant documents...")
        results = rag.retrieve(query)
        logger.info("Retrieved %d relevant documents:", len(results))

        for i, doc in enumerate(results, 1):
            logger.info("\nDocument %d (Score: %.4f):", i, doc.get("score", 0.0))
            logger.info("Content: %s", doc["content"])
            logger.info("Metadata: %s", doc["metadata"])

        # Generate a response using the RAG pipeline
        logger.info("\nGenerating response...")
        try:
            response = rag.generate(
                query=query,
                max_tokens=150,
                temperature=0.7,
            )

            logger.info("\nGenerated response:")
            print("\n" + "=" * 70)
            print(f"QUERY: {query}")
            print("-" * 70)
            print(response.get("response", "No response generated"))
            print("=" * 70 + "\n")

            if "sources" in response:
                logger.info("Sources used:")
                for i, source in enumerate(response["sources"], 1):
                    logger.info(f"  {i}. {source}")

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

    logger.info("RAG pipeline with Milvus example completed successfully!")


if __name__ == "__main__":
    main()
