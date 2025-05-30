"""
RAG (Retrieval-Augmented Generation) Processor for document retrieval and generation.

This module provides a comprehensive RAG implementation with support for CRUD operations
on documents and their vector embeddings, using Milvus as the vector store.
"""
import logging
from typing import Dict, Any, Optional, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.chains import Runnable
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    RAG processor that handles document storage, retrieval, and generation.
    
    This class provides a complete implementation of the RAG pipeline with support for:
    - Document ingestion and embedding
    - Semantic search and retrieval
    - Context-augmented generation
    - Document management (CRUD operations)
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        retriever: BaseRetriever,
        embeddings: Embeddings,
        prompt_template: PromptTemplate,
        llm: Runnable,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the RAG processor.
        
        Args:
            vectorstore: Vector store for document storage and retrieval
            retriever: Retriever for fetching relevant documents
            embeddings: Embeddings model for text vectorization
            prompt_template: Template for generating prompts
            llm: Language model for generation
            config: Additional configuration parameters
        """
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.embeddings = embeddings
        self.prompt_template = prompt_template
        self.llm = llm
        self.config = config or {}
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize additional components."""
        # Initialize default prompt template if not provided
        if not hasattr(self, 'prompt_template') or not self.prompt_template:
            self.prompt_template = self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> PromptTemplate:
        """Get default prompt template for RAG."""
        template = """Answer the question based on the following context:
        
        {context}
        
        Question: {question}
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query using the RAG pipeline.
        
        Args:
            query: The input query
            **kwargs: Additional parameters for retrieval and generation
            
        Returns:
            Dictionary containing the generated response and source documents
        """
        try:
            # 1. Retrieve relevant documents
            docs = await self.retriever.aget_relevant_documents(query, **kwargs)
            
            # 2. Format context from documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}"
                for i, doc in enumerate(docs)
            ])
            
            # 3. Create prompt with context and query
            formatted_prompt = self.prompt_template.format(
                context=context,
                question=query,
                **kwargs
            )
            
            # 4. Get LLM response
            response = await self.llm.ainvoke(
                HumanMessage(content=formatted_prompt)
            )
            
            # 5. Format response
            return {
                "answer": response.content if hasattr(response, 'content') else str(response),
                "sources": [doc.metadata for doc in docs],
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise
            
    # Document CRUD Operations
    
    async def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            **kwargs: Additional parameters for the vector store
            
        Returns:
            List of document IDs for the added documents
        """
        if not documents:
            return []
            
        try:
            # Add documents to vector store
            doc_ids = await self.vectorstore.aadd_documents(documents, **kwargs)
            logger.info(f"Added {len(doc_ids)} documents to the vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            raise
            
    async def update_document(self, document_id: str, document: Document, **kwargs) -> bool:
        """Update a document in the vector store.
        
        Args:
            document_id: ID of the document to update
            document: Updated Document object
            **kwargs: Additional parameters for the vector store
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Update document in vector store
            success = await self.vectorstore.aupdate_document(document_id, document, **kwargs)
            if success:
                logger.info(f"Updated document {document_id} in the vector store")
            else:
                logger.warning(f"Failed to update document {document_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}", exc_info=True)
            return False
            
    async def delete_documents(self, document_ids: List[str], **kwargs) -> bool:
        """Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            **kwargs: Additional parameters for the vector store
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not document_ids:
            return False
            
        try:
            # Delete documents from vector store
            success = await self.vectorstore.adelete(document_ids, **kwargs)
            if success:
                logger.info(f"Deleted {len(document_ids)} documents from the vector store")
            else:
                logger.warning("Failed to delete some or all documents")
            return success
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}", exc_info=True)
            return False
            
    async def get_document(self, document_id: str, **kwargs) -> Optional[Document]:
        """Retrieve a document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Document object if found, None otherwise
        """
        try:
            # Search for document by ID in metadata
            results = await self.vectorstore.asimilarity_search(
                "",  # Empty query to match all
                filter={"document_id": document_id},
                k=1,
                **kwargs
            )
            
            if results:
                return results[0]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}", exc_info=True)
            return None
            
    async def search_documents(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for documents similar to the query.
        
        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of matching Document objects
        """
        try:
            # Perform similarity search
            results = await self.vectorstore.asimilarity_search(query, k=k, **kwargs)
            logger.debug(f"Found {len(results)} documents matching query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return []

    async def add_document(self, document: Document) -> None:
        """Add a document to the vector store."""
        try:
            # Get embeddings for document
            embeddings = self.embeddings.embed_documents([document.page_content])
            
            # Add to vector store
            await self.vectorstore.aadd_documents([document], embeddings)
            
        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            raise
