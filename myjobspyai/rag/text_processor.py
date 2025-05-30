import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        use_semantic_chunking: bool = True,
        max_parallel_chunks: int = 5,
    ):
        """Initialize the TextProcessor.
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to use for splitting text
            use_semantic_chunking: Whether to use semantic chunking (more accurate but slower)
            max_parallel_chunks: Maximum number of chunks to process in parallel
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " "]
        self.use_semantic_chunking = use_semantic_chunking
        self.max_parallel_chunks = max_parallel_chunks
        
        # Initialize the appropriate text splitter based on settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
        )
        
        # Initialize semantic chunking components if enabled
        self.semantic_splitter = None
        if self.use_semantic_chunking:
            try:
                # Lazy import to avoid dependency if not using semantic chunking
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain.embeddings import HuggingFaceEmbeddings
                
                embeddings = HuggingFaceEmbeddings()
                self.semantic_splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95,
                )
                logger.info("Semantic chunking enabled")
            except ImportError:
                logger.warning(
                    "Semantic chunking dependencies not found. Falling back to RecursiveCharacterTextSplitter. "
                    "Install with 'pip install langchain-experimental sentence-transformers' to enable semantic chunking."
                )
                self.use_semantic_chunking = False

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        return text.strip()

    def process_documents(
        self,
        documents: List[Document],
        vectorstore: VectorStore,
        embeddings: Optional[Embeddings] = None,
        batch_size: int = 10,
    ) -> None:
        """Process a list of documents in parallel and add them to the vector store.
        
        Args:
            documents: List of Document objects to process
            vectorstore: Vector store to add documents to
            embeddings: Embeddings model to use
            batch_size: Number of documents to process in parallel
        """
        # Process documents in batches to avoid memory issues
        
        total_batches = (len(documents) - 1) // batch_size + 1
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info("Processing batch %d/%d", batch_num, total_batches)
            
            with ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
                future_to_doc = {
                    executor.submit(self.split_text, doc.page_content): doc
                    for doc in batch
                }
                
                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    doc_id = doc.metadata.get("id", "unknown")
                    try:
                        chunks = future.result()
                        if not chunks:
                            continue
                            
                        # Add document metadata to each chunk
                        for chunk in chunks:
                            chunk.metadata.update({
                                "document_id": doc_id,
                                **{k: v for k, v in doc.metadata.items() 
                                   if k not in chunk.metadata or not chunk.metadata[k]}
                            })
                        
                        # Add chunks to vector store
                        vectorstore.add_documents(chunks)
                        logger.info(
                            "Added %d chunks from document %s", 
                            len(chunks), doc_id
                        )
                    except Exception as e:
                        logger.error(
                            "Error processing document %s: %s", 
                            doc_id, str(e), exc_info=True
                        )
    
    def split_text(self, text: str) -> List[Document]:
        """Split text into chunks using the configured text splitter.
        
        If semantic chunking is enabled and available, it will be used.
        Otherwise, falls back to RecursiveCharacterTextSplitter.
        """
        try:
            # Clean the text first
            cleaned_text = self.clean_text(text)
            
            # Use semantic chunking if available and enabled
            if self.use_semantic_chunking and self.semantic_splitter is not None:
                try:
                    documents = self.semantic_splitter.create_documents([cleaned_text])
                    # Add metadata to each document
                    for i, doc in enumerate(documents):
                        doc.metadata.update({
                            "source": "semantic_split",
                            "chunk_position": i,
                            "chunk_size": len(doc.page_content),
                        })
                    return documents
                except Exception as e:
                    logger.warning(
                        f"Error in semantic chunking: {str(e)}. Falling back to RecursiveCharacterTextSplitter."
                    )
            
            # Fall back to RecursiveCharacterTextSplitter
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create Document objects
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": "recursive_split",
                        "chunk_position": i,
                        "total_chunks": len(chunks)
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error splitting text: {e}", exc_info=True)
            raise
