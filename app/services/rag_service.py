"""RAG (Retrieval-Augmented Generation) service for document search."""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.config import settings
from app.utils.document_loader import Document, load_documents_from_directory


class RAGService:
    """Service for document retrieval using vector similarity search."""

    def __init__(self):
        """Initialize RAG service."""
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialized = False

    def initialize(self):
        """Initialize ChromaDB and embedding function."""
        try:
            logger.info("Initializing RAG Service...")

            # Create persist directory if it doesn't exist
            persist_directory = settings.chroma_persist_directory
            Path(persist_directory).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Initialize embedding function
            if settings.embedding_provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key is required for embeddings")

                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.openai_api_key,
                    model_name=settings.embedding_model
                )
                logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
            else:
                # Default to sentence-transformers
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("Using SentenceTransformer embeddings: all-MiniLM-L6-v2")

            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name="documents",
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection with {self.collection.count()} documents")
            except Exception:
                self.collection = self.client.create_collection(
                    name="documents",
                    embedding_function=self.embedding_function,
                    metadata={"description": "PT Products documentation"}
                )
                logger.info("Created new collection")

            self._initialized = True
            logger.success("RAG Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise

    def load_documents(self, directory: str, force_reload: bool = False):
        """
        Load documents from directory into vector store.

        Args:
            directory: Directory containing documents
            force_reload: If True, clear existing documents and reload
        """
        if not self._initialized:
            self.initialize()

        try:
            # Check if documents already loaded
            if self.collection.count() > 0 and not force_reload:
                logger.info(f"Collection already contains {self.collection.count()} documents. Skipping load.")
                logger.info("Use force_reload=True to reload documents")
                return

            # Clear collection if force reload
            if force_reload and self.collection.count() > 0:
                logger.warning("Force reload: clearing existing documents")
                self.client.delete_collection("documents")
                self.collection = self.client.create_collection(
                    name="documents",
                    embedding_function=self.embedding_function,
                    metadata={"description": "PT Products documentation"}
                )

            # Load and chunk documents
            logger.info(f"Loading documents from {directory}...")
            documents = load_documents_from_directory(
                directory,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

            if not documents:
                logger.warning("No documents found to load")
                return

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, doc in enumerate(documents):
                ids.append(f"doc_{i}")
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

            # Add documents to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
                logger.debug(f"Added batch {i // batch_size + 1}: {i}-{batch_end}/{len(documents)}")

            logger.success(f"Successfully loaded {len(documents)} document chunks into vector store")

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant documents with scores
        """
        if not self._initialized:
            self.initialize()

        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata
            )

            # Format results
            documents = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    documents.append(doc)

            logger.debug(f"Search query: '{query}' returned {len(documents)} results")

            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a single document to the vector store.

        Args:
            content: Document content
            metadata: Optional metadata

        Returns:
            Document ID
        """
        if not self._initialized:
            self.initialize()

        try:
            # Generate ID
            doc_id = f"doc_{self.collection.count()}"

            # Add document
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata or {}]
            )

            logger.debug(f"Added document with ID: {doc_id}")

            return doc_id

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def delete_document(self, doc_id: str):
        """
        Delete a document from the vector store.

        Args:
            doc_id: Document ID to delete
        """
        if not self._initialized:
            self.initialize()

        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self._initialized:
            self.initialize()

        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_function": str(type(self.embedding_function).__name__)
        }

    def clear_collection(self):
        """Clear all documents from the collection."""
        if not self._initialized:
            self.initialize()

        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_function,
                metadata={"description": "PT Products documentation"}
            )
            logger.warning("Cleared all documents from collection")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise


# Global RAG service instance
rag_service = RAGService()
