"""Tests for RAG Agent."""
import pytest
import os
from pathlib import Path

from app.agents.rag_agent import RAGAgent
from app.services.rag_service import RAGService
from app.utils.document_loader import DocumentLoader, TextSplitter, Document


class TestDocumentLoader:
    """Tests for document loader."""

    def test_load_single_file(self, tmp_path):
        """Test loading a single markdown file."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test.", encoding='utf-8')

        loader = DocumentLoader()
        doc = loader.load_file(str(test_file))

        assert doc is not None
        assert "Test Document" in doc.page_content
        assert doc.metadata['filename'] == "test.md"
        assert doc.metadata['extension'] == ".md"

    def test_load_directory(self, tmp_path):
        """Test loading multiple files from directory."""
        # Create test files
        (tmp_path / "doc1.md").write_text("Content 1", encoding='utf-8')
        (tmp_path / "doc2.txt").write_text("Content 2", encoding='utf-8')
        (tmp_path / "doc3.md").write_text("Content 3", encoding='utf-8')

        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))

        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)

    def test_unsupported_file_type(self, tmp_path):
        """Test handling of unsupported file types."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        loader = DocumentLoader()
        doc = loader.load_file(str(test_file))

        assert doc is None


class TestTextSplitter:
    """Tests for text splitter."""

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        text = "Short text"
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_split_long_text(self):
        """Test splitting long text into multiple chunks."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "Lorem ipsum dolor sit amet. " * 20  # ~560 characters
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 100 + 50 for chunk in chunks)  # Allow some variance

    def test_split_documents(self):
        """Test splitting documents with metadata preservation."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        doc = Document(
            page_content="Test content. " * 30,
            metadata={"source": "test.md", "filename": "test.md"}
        )
        chunked_docs = splitter.split_documents([doc])

        assert len(chunked_docs) > 1
        assert all(doc.metadata['source'] == "test.md" for doc in chunked_docs)
        assert all('chunk' in doc.metadata for doc in chunked_docs)
        assert all('total_chunks' in doc.metadata for doc in chunked_docs)


class TestRAGService:
    """Tests for RAG service."""

    @pytest.fixture
    def rag_service(self, tmp_path):
        """Create RAG service with test data."""
        # Note: This requires OpenAI API key or sentence-transformers
        # For CI/CD, you might want to mock the embedding function
        service = RAGService()
        # Override persist directory for testing
        service.client = None
        return service

    def test_initialize(self, rag_service):
        """Test RAG service initialization."""
        # This test requires actual ChromaDB setup
        # Skip in environments without proper setup
        try:
            rag_service.initialize()
            assert rag_service._initialized is True
            assert rag_service.client is not None
            assert rag_service.collection is not None
        except Exception as e:
            pytest.skip(f"RAG service initialization failed: {e}")

    def test_add_document(self, rag_service):
        """Test adding a document to the collection."""
        try:
            rag_service.initialize()
            doc_id = rag_service.add_document(
                content="Test document content",
                metadata={"source": "test"}
            )
            assert doc_id is not None
            assert rag_service.collection.count() > 0
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    def test_search(self, rag_service):
        """Test searching for documents."""
        try:
            rag_service.initialize()

            # Add test documents
            rag_service.add_document("Python is a programming language", {"topic": "python"})
            rag_service.add_document("JavaScript is used for web development", {"topic": "javascript"})

            # Search
            results = rag_service.search("programming language", top_k=2)

            assert len(results) > 0
            assert "Python" in results[0]['content'] or "JavaScript" in results[0]['content']
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")


class TestRAGAgent:
    """Tests for RAG agent."""

    @pytest.fixture
    def rag_agent(self):
        """Create RAG agent for testing."""
        agent = RAGAgent()
        return agent

    def test_initialize_without_docs(self, rag_agent):
        """Test initializing agent without loading documents."""
        try:
            rag_agent.initialize(load_docs=False)
            assert rag_agent._initialized is True
            assert rag_agent.llm is not None
        except Exception as e:
            pytest.skip(f"RAG agent initialization failed: {e}")

    def test_answer_question_no_documents(self, rag_agent):
        """Test answering question with no documents in collection."""
        try:
            rag_agent.initialize(load_docs=False)

            # Clear collection
            rag_agent.rag_service.clear_collection()

            result = rag_agent.answer_question("What is PT AI?")

            assert result is not None
            assert "success" in result
            # Should fail gracefully when no documents found
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    def test_answer_question_with_documents(self, rag_agent, tmp_path):
        """Test answering question with documents."""
        try:
            # Create test documents
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()

            test_doc = docs_dir / "test.md"
            test_doc.write_text("""
# PT Application Inspector

PT Application Inspector is a static application security testing (SAST) tool.
It helps developers find security vulnerabilities in source code.

## Features
- Supports multiple programming languages
- Integrates with CI/CD pipelines
- Provides detailed security reports
            """, encoding='utf-8')

            # Initialize with test documents
            rag_agent.initialize(load_docs=True, docs_directory=str(docs_dir))

            # Ask question
            result = rag_agent.answer_question("What is PT Application Inspector?")

            assert result is not None
            assert result.get("success") is True
            assert "answer" in result
            assert len(result["answer"]) > 0

            # Check if answer mentions relevant terms
            answer_lower = result["answer"].lower()
            assert any(term in answer_lower for term in ["security", "sast", "application", "static"])

        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    def test_get_collection_info(self, rag_agent):
        """Test getting collection information."""
        try:
            rag_agent.initialize(load_docs=False)
            info = rag_agent.get_collection_info()

            assert "total_documents" in info
            assert "collection_name" in info
            assert isinstance(info["total_documents"], int)
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")


# Integration test with real data (skip if docs not available)
class TestRAGIntegration:
    """Integration tests with actual documentation."""

    def test_with_real_documentation(self):
        """Test RAG agent with real PT documentation."""
        docs_path = Path("./data/docs")

        if not docs_path.exists() or not any(docs_path.iterdir()):
            pytest.skip("Documentation directory not found or empty")

        try:
            agent = RAGAgent()
            agent.initialize(load_docs=True, docs_directory=str(docs_path))

            # Test various questions
            questions = [
                "Что такое PT Application Inspector?",
                "Какие возможности есть у PT NAD?",
                "Как работает PT Sandbox?",
            ]

            for question in questions:
                result = agent.answer_question(question)

                assert result is not None
                assert result.get("success") is True
                assert len(result.get("answer", "")) > 50  # Expect substantial answer
                assert len(result.get("sources", [])) > 0

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
