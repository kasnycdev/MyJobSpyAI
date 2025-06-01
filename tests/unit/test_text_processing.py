"""Test script for the text processing pipeline with chunking and parallel processing."""
import os
import time
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from myjobspyai.rag.text_processor import TextProcessor
from myjobspyai.config import settings

# Sample documents for testing
SAMPLE_DOCS = [
    Document(
        page_content="""
        Software Engineer with 5+ years of experience in Python and machine learning.
        Proficient in building scalable web applications and deploying ML models.
        Strong background in natural language processing and computer vision.
        """,
        metadata={"id": "doc1", "type": "resume"}
    ),
    Document(
        page_content="""
        Senior Data Scientist with expertise in deep learning and big data.
        Experience with PyTorch, TensorFlow, and cloud platforms.
        Published research in top AI conferences.
        """,
        metadata={"id": "doc2", "type": "resume"}
    ),
    # Add more sample documents as needed
]

# Initialize test vector store and embeddings
TEST_EMBEDDINGS = HuggingFaceEmbeddings()
TEST_INDEX_DIR = Path("test_index")

def setup_module():
    """Set up test environment."""
    TEST_INDEX_DIR.mkdir(exist_ok=True)

def teardown_module():
    """Clean up after tests."""
    # Clean up test index directory
    for f in TEST_INDEX_DIR.glob("*"):
        f.unlink()
    TEST_INDEX_DIR.rmdir()

def create_test_processor(**kwargs) -> TextProcessor:
    """Create a TextProcessor instance with test defaults and overrides."""
    defaults = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "use_semantic_chunking": True,
        "max_parallel_chunks": 3
    }
    defaults.update(kwargs)
    return TextProcessor(**defaults)

def test_text_processor_initialization():
    """Test TextProcessor initialization with different configurations."""
    # Test with defaults
    processor = create_test_processor()
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200
    assert processor.use_semantic_chunking is True
    assert processor.max_parallel_chunks == 3

    # Test with custom values
    processor = create_test_processor(
        chunk_size=2000,
        chunk_overlap=500,
        use_semantic_chunking=False,
        max_parallel_chunks=5
    )
    assert processor.chunk_size == 2000
    assert processor.chunk_overlap == 500
    assert processor.use_semantic_chunking is False
    assert processor.max_parallel_chunks == 5

def test_text_splitting():
    """Test text splitting functionality."""
    processor = create_test_processor(chunk_size=100, chunk_overlap=20)
    
    # Test with a simple document
    doc = Document(
        page_content="This is a test document. " * 10,  # ~200 chars
        metadata={"id": "test1"}
    )
    
    chunks = processor.split_text(doc.page_content)
    assert len(chunks) > 0
    
    # Verify chunk sizes are within limits
    for chunk in chunks:
        assert len(chunk.page_content) <= 100  # chunk_size
    
    # Verify overlap is maintained
    if len(chunks) > 1:
        chunk1_end = chunks[0].page_content[-20:]
        chunk2_start = chunks[1].page_content[:20]
        assert chunk1_end in chunks[1].page_content or chunk2_start in chunks[0].page_content

def test_parallel_processing():
    """Test parallel document processing."""
    processor = create_test_processor(
        chunk_size=500,
        chunk_overlap=100,
        max_parallel_chunks=2
    )
    
    # Create a test vector store
    vectorstore = FAISS.from_documents(
        documents=[],
        embedding=TEST_EMBEDDINGS
    )
    
    # Process documents in parallel
    start_time = time.time()
    processor.process_documents(
        documents=SAMPLE_DOCS,
        vectorstore=vectorstore,
        embeddings=TEST_EMBEDDINGS,
        batch_size=2
    )
    processing_time = time.time() - start_time
    
    # Verify documents were processed
    assert len(vectorstore.docstore._dict) >= len(SAMPLE_DOCS)
    
    # Log performance metrics
    print(f"\nProcessed {len(SAMPLE_DOCS)} documents in {processing_time:.2f} seconds")
    print(f"Chunk size: {processor.chunk_size}, Overlap: {processor.chunk_overlap}")
    print(f"Max parallel chunks: {processor.max_parallel_chunks}")

def test_semantic_chunking():
    """Test semantic chunking functionality."""
    try:
        processor = create_test_processor(
            use_semantic_chunking=True,
            chunk_size=300,
            chunk_overlap=50
        )
        
        # Test with a document containing multiple paragraphs
        doc = Document(
            page_content="""
            First paragraph. This is the first section of the document.
            
            Second paragraph. This is a different topic that should be in a separate chunk.
            
            Third paragraph. This continues the second topic but might be combined with the second paragraph.
            """,
            metadata={"id": "semantic_test"}
        )
        
        chunks = processor.split_text(doc.page_content)
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Verify metadata is preserved
        for chunk in chunks:
            assert chunk.metadata["id"] == "semantic_test"
            
    except ImportError:
        pytest.skip("Semantic chunking dependencies not available")

if __name__ == "__main__":
    # Run tests and print results
    import pytest
    import sys
    
    # Run tests with detailed output
    sys.exit(pytest.main(["-v", __file__]))
