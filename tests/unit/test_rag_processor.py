"""
Tests for the RAG (Retrieval-Augmented Generation) processor.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from myjobspyai.rag.rag_processor import RAGProcessor

# Test data
TEST_DOCUMENTS = [
    Document(
        page_content="Test document 1",
        metadata={"source": "test", "id": "doc1"}
    ),
    Document(
        page_content="Test document 2",
        metadata={"source": "test", "id": "doc2"}
    )
]

TEST_QUERY = "Test query"
TEST_RESPONSE = "Test response"

# Fixtures

@pytest.fixture
def mock_vectorstore() -> VectorStore:
    """Create a mock vector store."""
    mock = AsyncMock(spec=VectorStore)
    mock.aadd_documents.return_value = ["doc1", "doc2"]
    mock.aupdate_document.return_value = True
    mock.adelete.return_value = True
    mock.asimilarity_search.return_value = TEST_DOCUMENTS
    return mock

@pytest.fixture
def mock_retriever() -> BaseRetriever:
    """Create a mock retriever."""
    mock = AsyncMock(spec=BaseRetriever)
    mock.aget_relevant_documents.return_value = TEST_DOCUMENTS
    return mock

@pytest.fixture
def mock_embeddings() -> Embeddings:
    """Create a mock embeddings model."""
    mock = MagicMock(spec=Embeddings)
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock

@pytest.fixture
def mock_llm() -> Runnable:
    """Create a mock LLM."""
    mock = AsyncMock()
    mock.ainvoke.return_value = AIMessage(content="Test response")
    return mock

@pytest.fixture
def rag_processor(
    mock_vectorstore: VectorStore,
    mock_retriever: BaseRetriever,
    mock_embeddings: Embeddings,
    mock_llm: Runnable
) -> RAGProcessor:
    """Create a RAG processor with mock dependencies."""
    prompt_template = PromptTemplate(
        template="Context: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    
    # Configure the mock LLM to return a test response
    mock_llm.ainvoke.return_value = AIMessage(content=TEST_RESPONSE)
    
    return RAGProcessor(
        vectorstore=mock_vectorstore,
        retriever=mock_retriever,
        embeddings=mock_embeddings,
        prompt_template=prompt_template,
        llm=mock_llm,
        config={"test_config": "value"}  # Test config
    )

# Tests

@pytest.mark.asyncio
async def test_process_query(rag_processor: RAGProcessor):
    """Test processing a query through the RAG pipeline."""
    response = await rag_processor.process_query("Test query")
    
    assert "answer" in response
    assert "sources" in response
    assert "context" in response
    assert isinstance(response["answer"], str)
    assert len(response["sources"]) > 0

@pytest.mark.asyncio
async def test_add_documents(rag_processor: RAGProcessor):
    """Test adding documents to the vector store."""
    documents = [
        Document(page_content="Test document 1"),
        Document(page_content="Test document 2")
    ]
    
    doc_ids = await rag_processor.add_documents(documents)
    assert len(doc_ids) == 2
    rag_processor.vectorstore.aadd_documents.assert_awaited_once()

@pytest.mark.asyncio
async def test_update_document(rag_processor: RAGProcessor):
    """Test updating a document in the vector store."""
    doc = Document(page_content="Updated document")
    success = await rag_processor.update_document("doc1", doc)
    
    assert success is True
    rag_processor.vectorstore.aupdate_document.assert_awaited_once()

@pytest.mark.asyncio
async def test_delete_documents(rag_processor: RAGProcessor):
    """Test deleting documents from the vector store."""
    success = await rag_processor.delete_documents(["doc1", "doc2"])
    
    assert success is True
    rag_processor.vectorstore.adelete.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_document(rag_processor: RAGProcessor):
    """Test retrieving a document by ID."""
    document = await rag_processor.get_document("doc1")
    
    assert document is not None
    assert hasattr(document, "page_content")
    rag_processor.vectorstore.asimilarity_search.assert_awaited_once()

@pytest.mark.asyncio
async def test_search_documents(rag_processor: RAGProcessor):
    """Test searching for documents."""
    results = await rag_processor.search_documents("test query", k=2)
    
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)
    rag_processor.vectorstore.asimilarity_search.assert_awaited_once()

@pytest.mark.asyncio
async def test_process_query_error_handling(rag_processor: RAGProcessor):
    """Test error handling in process_query."""
    rag_processor.retriever.aget_relevant_documents.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        await rag_processor.process_query("test query")
