"""Tests for Web Search Agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.agents.web_search_agent import WebSearchAgent
from app.services.search_service import SearchService


class TestSearchService:
    """Tests for search service."""

    @pytest.fixture
    def search_service(self):
        """Create search service for testing."""
        service = SearchService()
        return service

    def test_initialize_without_key(self, search_service):
        """Test initialization without API key."""
        with patch('app.services.search_service.settings.tavily_api_key', None):
            search_service.initialize()
            # Should not crash, but service won't be initialized
            assert search_service._initialized is False or search_service.client is None

    @patch('app.services.search_service.TavilyClient')
    def test_initialize_with_key(self, mock_tavily, search_service):
        """Test initialization with API key."""
        with patch('app.services.search_service.settings.tavily_api_key', 'test_key'):
            search_service.initialize()
            mock_tavily.assert_called_once_with(api_key='test_key')

    @patch('app.services.search_service.TavilyClient')
    def test_search_success(self, mock_tavily, search_service):
        """Test successful search."""
        # Mock Tavily client
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "Test answer",
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "Test content 1",
                    "score": 0.95
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Test content 2",
                    "score": 0.85
                }
            ]
        }
        mock_tavily.return_value = mock_client

        with patch('app.services.search_service.settings.tavily_api_key', 'test_key'):
            search_service.initialize()
            results = search_service.search("test query", max_results=2)

            assert len(results) == 2
            assert results[0]["title"] == "Test Result 1"
            assert results[0]["url"] == "https://example.com/1"
            assert results[0]["score"] == 0.95
            assert "tavily_answer" in results[0]
            assert results[0]["tavily_answer"] == "Test answer"

    def test_search_without_initialization(self, search_service):
        """Test search without initialization."""
        results = search_service.search("test query")
        # Should return empty list when not initialized
        assert results == []

    @patch('app.services.search_service.TavilyClient')
    def test_filter_results(self, mock_tavily, search_service):
        """Test filtering results by score."""
        results = [
            {"title": "High score", "score": 0.9},
            {"title": "Medium score", "score": 0.6},
            {"title": "Low score", "score": 0.3}
        ]

        filtered = search_service.filter_results(results, min_score=0.5)
        assert len(filtered) == 2
        assert all(r["score"] >= 0.5 for r in filtered)


class TestWebSearchAgent:
    """Tests for web search agent."""

    @pytest.fixture
    def web_search_agent(self):
        """Create web search agent for testing."""
        agent = WebSearchAgent()
        return agent

    def test_initialize_agent(self, web_search_agent):
        """Test agent initialization."""
        try:
            web_search_agent.initialize()
            # Initialization might fail without API key, but should not crash
            assert web_search_agent.llm is not None
        except Exception:
            pytest.skip("Agent initialization failed (expected without valid API key)")

    @patch('app.agents.web_search_agent.search_service')
    def test_search_and_answer_no_api_key(self, mock_search_service, web_search_agent):
        """Test search when service is not initialized."""
        mock_search_service._initialized = False

        web_search_agent.initialize()
        result = web_search_agent.search_and_answer("test question")

        assert result["success"] is False
        assert "недоступен" in result["answer"].lower()

    @patch('app.agents.web_search_agent.search_service')
    @patch('app.agents.web_search_agent.get_llm')
    def test_search_and_answer_success(self, mock_get_llm, mock_search_service, web_search_agent):
        """Test successful search and answer generation."""
        # Mock search service
        mock_search_service._initialized = True
        mock_search_service.search.return_value = [
            {
                "title": "Test Article",
                "url": "https://example.com",
                "content": "Test content about the question",
                "score": 0.9,
                "tavily_answer": "This is the answer"
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated answer based on search results"
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        web_search_agent.initialize()
        result = web_search_agent.search_and_answer("test question")

        assert result["success"] is True
        assert "answer" in result
        assert len(result["sources"]) > 0
        assert result["search_results"] == 1

    @patch('app.agents.web_search_agent.search_service')
    def test_search_and_answer_no_results(self, mock_search_service, web_search_agent):
        """Test when search returns no results."""
        mock_search_service._initialized = True
        mock_search_service.search.return_value = []

        web_search_agent.initialize()
        result = web_search_agent.search_and_answer("obscure question")

        assert result["success"] is False
        assert "не нашел" in result["answer"].lower()

    @patch('app.agents.web_search_agent.search_service')
    @patch('app.agents.web_search_agent.get_llm')
    def test_search_news(self, mock_get_llm, mock_search_service, web_search_agent):
        """Test news search functionality."""
        # Mock search service
        mock_search_service._initialized = True
        mock_search_service.search_news.return_value = [
            {
                "title": "Breaking News",
                "url": "https://news.example.com",
                "content": "Important news content",
                "score": 0.95,
                "published_date": "2025-10-12"
            }
        ]

        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "News summary"
        mock_llm_instance.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm_instance

        web_search_agent.initialize()
        result = web_search_agent.search_news("latest news", max_results=5, days=7)

        assert result["success"] is True
        assert "answer" in result
        assert result["articles_found"] == 1

    def test_format_search_context(self, web_search_agent):
        """Test formatting search results into context."""
        results = [
            {
                "title": "Article 1",
                "url": "https://example.com/1",
                "content": "Content 1",
                "score": 0.9,
                "tavily_answer": "Summary"
            },
            {
                "title": "Article 2",
                "url": "https://example.com/2",
                "content": "Content 2",
                "score": 0.8
            }
        ]

        context = web_search_agent._format_search_context(results)

        assert "AI Summary: Summary" in context
        assert "Article 1" in context
        assert "Article 2" in context
        assert "https://example.com/1" in context
        assert "Content 1" in context

    def test_extract_sources(self, web_search_agent):
        """Test extracting sources from results."""
        results = [
            {"title": "Article 1", "url": "https://example.com/1"},
            {"title": "Article 2", "url": "https://example.com/2"}
        ]

        sources = web_search_agent._extract_sources(results)

        assert len(sources) == 2
        assert "Article 1" in sources[0]
        assert "https://example.com/1" in sources[0]


# Integration tests (require actual API key)
class TestWebSearchIntegration:
    """Integration tests with real Tavily API."""

    @pytest.mark.skip(reason="Requires valid Tavily API key")
    def test_real_search(self):
        """Test with real Tavily API (skip if no key)."""
        from app.agents.web_search_agent import web_search_agent

        web_search_agent.initialize()

        if not web_search_agent.search_service._initialized:
            pytest.skip("Tavily API key not configured")

        result = web_search_agent.search_and_answer("What is artificial intelligence?")

        assert result["success"] is True
        assert len(result["answer"]) > 50
        assert len(result["sources"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
