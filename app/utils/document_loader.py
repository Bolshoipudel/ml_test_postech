"""Document loader utilities for RAG system."""
import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


class Document:
    """Simple document class to hold text and metadata."""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        """
        Initialize document.

        Args:
            page_content: The text content
            metadata: Optional metadata dictionary
        """
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class DocumentLoader:
    """Loader for various document types."""

    def __init__(self):
        """Initialize document loader."""
        self.supported_extensions = ['.md', '.txt', '.text']

    def load_file(self, file_path: str) -> Document:
        """
        Load a single document from file.

        Args:
            file_path: Path to the file

        Returns:
            Document object
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                'source': str(path),
                'filename': path.name,
                'extension': path.suffix,
                'size': path.stat().st_size
            }

            logger.debug(f"Loaded document: {path.name} ({len(content)} chars)")

            return Document(page_content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects
        """
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        documents = []

        # Get all files with supported extensions
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_file(str(file_path))
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")

        return documents


class TextSplitter:
    """Split documents into chunks for better retrieval."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text splitter.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If not at the end, try to break at a good point
            if end < len(text):
                # Try to break at paragraph
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break
                else:
                    # Try to break at newline
                    newline_break = text.rfind('\n', start, end)
                    if newline_break != -1 and newline_break > start:
                        end = newline_break
                    else:
                        # Try to break at sentence
                        sentence_break = max(
                            text.rfind('. ', start, end),
                            text.rfind('! ', start, end),
                            text.rfind('? ', start, end)
                        )
                        if sentence_break != -1 and sentence_break > start:
                            end = sentence_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else end

        logger.debug(f"Split text into {len(chunks)} chunks")

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents
        """
        chunked_documents = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk'] = i
                metadata['total_chunks'] = len(chunks)

                chunked_documents.append(
                    Document(page_content=chunk, metadata=metadata)
                )

        logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")

        return chunked_documents


# Helper functions
def load_documents_from_directory(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load and chunk documents from a directory.

    Args:
        directory: Directory path
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked documents
    """
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load documents
    documents = loader.load_directory(directory, recursive=True)

    if not documents:
        logger.warning(f"No documents found in {directory}")
        return []

    # Split into chunks
    chunked_docs = splitter.split_documents(documents)

    return chunked_docs
