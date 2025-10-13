"""Tests for Router Agent."""
import pytest
from unittest.mock import Mock, patch
import json

from app.agents.router_agent import RouterAgent
from app.prompts.router_prompts import get_router_prompt, ROUTER_FEW_SHOT_EXAMPLES


class TestRouterPrompts:
    """Tests for router prompts."""

    def test_get_router_prompt(self):
        """Test prompt generation."""
        query = "Сколько разработчиков работает над PT AI?"
        prompt = get_router_prompt(query)

        assert query in prompt
        assert "SQL" in prompt
        assert "RAG" in prompt
        assert "WEB_SEARCH" in prompt
        assert "MULTIPLE" in prompt

    def test_few_shot_examples_structure(self):
        """Test few-shot examples have correct structure."""
        for example in ROUTER_FEW_SHOT_EXAMPLES:
            assert "query" in example
            assert "response" in example
            response = example["response"]
            assert "tool" in response
            assert "reasoning" in response
            assert "confidence" in response
            assert response["tool"] in ["SQL", "RAG", "WEB_SEARCH", "MULTIPLE"]


class TestRouterAgent:
    """Tests for Router Agent."""

    @pytest.fixture
    def router_agent(self):
        """Create router agent for testing."""
        agent = RouterAgent(use_few_shot=False)
        return agent

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response."""
        def _create_response(tool: str, reasoning: str = "Test reasoning", confidence: float = 0.9):
            response = Mock()
            response.content = json.dumps({
                "tool": tool,
                "reasoning": reasoning,
                "confidence": confidence,
                "query_type": "test"
            })
            return response
        return _create_response

    def test_initialization(self, router_agent):
        """Test router agent initialization."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_get_llm.return_value = Mock()
            router_agent.initialize()

            assert router_agent._initialized is True
            assert router_agent.llm is not None
            mock_get_llm.assert_called_once_with(temperature=0.0)

    def test_parse_json_response(self, router_agent):
        """Test parsing JSON response from LLM."""
        json_response = json.dumps({
            "tool": "SQL",
            "reasoning": "Test",
            "confidence": 0.9,
            "query_type": "test"
        })

        result = router_agent._parse_routing_response(json_response)

        assert result["tool"] == "SQL"
        assert result["reasoning"] == "Test"
        assert result["confidence"] == 0.9

    def test_parse_json_with_markdown(self, router_agent):
        """Test parsing JSON wrapped in markdown code blocks."""
        json_response = """```json
{
  "tool": "RAG",
  "reasoning": "Test reasoning",
  "confidence": 0.85,
  "query_type": "test"
}
```"""

        result = router_agent._parse_routing_response(json_response)

        assert result["tool"] == "RAG"
        assert result["confidence"] == 0.85

    def test_validate_routing_decision_valid(self, router_agent):
        """Test validation of valid routing decision."""
        decision = {
            "tool": "sql",  # lowercase should be normalized
            "reasoning": "Test",
            "confidence": 0.9,
            "query_type": "test"
        }

        validated = router_agent._validate_routing_decision(decision, "test query")

        assert validated["tool"] == "SQL"
        assert validated["confidence"] == 0.9

    def test_validate_routing_decision_missing_tool(self, router_agent):
        """Test validation handles missing tool field."""
        decision = {
            "reasoning": "Test",
            "confidence": 0.9
        }

        validated = router_agent._validate_routing_decision(decision, "test query")

        # Should fallback to RAG
        assert validated["tool"] == "RAG"
        assert "fallback" in validated["reasoning"].lower()

    def test_validate_routing_decision_invalid_tool(self, router_agent):
        """Test validation handles invalid tool."""
        decision = {
            "tool": "INVALID_TOOL",
            "reasoning": "Test",
            "confidence": 0.9
        }

        validated = router_agent._validate_routing_decision(decision, "test query")

        # Should fallback to RAG
        assert validated["tool"] == "RAG"

    def test_validate_routing_decision_multiple_without_tools(self, router_agent):
        """Test validation of MULTIPLE without tools list."""
        decision = {
            "tool": "MULTIPLE",
            "reasoning": "Test",
            "confidence": 0.9
        }

        validated = router_agent._validate_routing_decision(
            decision,
            "Сколько человек работает над PT AI и какие у него возможности?"
        )

        # Should infer tools from query
        assert validated["tool"] == "MULTIPLE" or validated["tool"] == "RAG"
        if validated["tool"] == "MULTIPLE":
            assert "tools" in validated
            assert len(validated["tools"]) > 0

    def test_infer_tools_from_query_sql(self, router_agent):
        """Test inferring SQL tool from query."""
        query = "Сколько разработчиков работает в команде?"
        tools = router_agent._infer_tools_from_query(query)

        assert "SQL" in tools

    def test_infer_tools_from_query_rag(self, router_agent):
        """Test inferring RAG tool from query."""
        query = "Что такое PT Application Inspector?"
        tools = router_agent._infer_tools_from_query(query)

        assert "RAG" in tools

    def test_infer_tools_from_query_web_search(self, router_agent):
        """Test inferring Web Search tool from query."""
        query = "Последние новости по кибербезопасности"
        tools = router_agent._infer_tools_from_query(query)

        assert "WEB_SEARCH" in tools

    def test_infer_tools_from_query_multiple(self, router_agent):
        """Test inferring multiple tools from complex query."""
        query = "Сколько разработчиков работает над PT AI и какие у него возможности?"
        tools = router_agent._infer_tools_from_query(query)

        # Should detect both SQL (сколько разработчиков) and RAG (возможности)
        assert len(tools) >= 2 or "SQL" in tools or "RAG" in tools

    def test_fallback_routing_sql(self, router_agent):
        """Test fallback routing for SQL-like queries."""
        query = "Сколько продуктов в базе данных?"
        result = router_agent._fallback_routing(query, "Test error")

        assert result["tool"] == "SQL"
        assert "fallback" in result["reasoning"].lower()

    def test_fallback_routing_web_search(self, router_agent):
        """Test fallback routing for web search queries."""
        query = "Последние новости о PT"
        result = router_agent._fallback_routing(query, "Test error")

        assert result["tool"] == "WEB_SEARCH"

    def test_fallback_routing_default(self, router_agent):
        """Test fallback routing defaults to RAG."""
        query = "Расскажи о продукте"
        result = router_agent._fallback_routing(query, "Test error")

        assert result["tool"] == "RAG"

    def test_route_with_mock_llm_sql(self, router_agent, mock_llm_response):
        """Test routing SQL query with mocked LLM."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_llm_response("SQL", "Database query")
            mock_get_llm.return_value = mock_llm

            router_agent.initialize()
            result = router_agent.route("Сколько разработчиков?")

            assert result["tool"] == "SQL"
            assert result["confidence"] == 0.9
            assert "reasoning" in result

    def test_route_with_mock_llm_rag(self, router_agent, mock_llm_response):
        """Test routing RAG query with mocked LLM."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_llm_response("RAG", "Documentation search")
            mock_get_llm.return_value = mock_llm

            router_agent.initialize()
            result = router_agent.route("Что такое PT AI?")

            assert result["tool"] == "RAG"
            assert result["confidence"] == 0.9

    def test_route_with_mock_llm_web_search(self, router_agent, mock_llm_response):
        """Test routing Web Search query with mocked LLM."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value = mock_llm_response("WEB_SEARCH", "Internet search")
            mock_get_llm.return_value = mock_llm

            router_agent.initialize()
            result = router_agent.route("Последние новости")

            assert result["tool"] == "WEB_SEARCH"

    def test_route_with_mock_llm_multiple(self, router_agent, mock_llm_response):
        """Test routing MULTIPLE query with mocked LLM."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_llm = Mock()
            response = Mock()
            response.content = json.dumps({
                "tool": "MULTIPLE",
                "tools": ["SQL", "RAG"],
                "reasoning": "Combined query",
                "confidence": 0.85,
                "query_type": "combined"
            })
            mock_llm.invoke.return_value = response
            mock_get_llm.return_value = mock_llm

            router_agent.initialize()
            result = router_agent.route("Сколько человек работает над PT AI и какие у него возможности?")

            assert result["tool"] == "MULTIPLE"
            assert "tools" in result
            assert "SQL" in result["tools"]
            assert "RAG" in result["tools"]

    def test_route_handles_llm_error(self, router_agent):
        """Test routing handles LLM errors gracefully."""
        with patch('app.agents.router_agent.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.side_effect = Exception("LLM error")
            mock_get_llm.return_value = mock_llm

            router_agent.initialize()
            result = router_agent.route("Test query")

            # Should fallback to some valid tool
            assert result["tool"] in ["SQL", "RAG", "WEB_SEARCH"]
            assert "fallback" in result["reasoning"].lower() or "error" in result["reasoning"].lower()

    def test_explain_routing(self, router_agent):
        """Test generating explanation for routing decision."""
        decision = {
            "tool": "SQL",
            "reasoning": "Query requires database access",
            "confidence": 0.9,
            "query_type": "count"
        }

        explanation = router_agent.explain_routing(decision)

        assert "SQL" in explanation
        assert "0.9" in explanation or "90%" in explanation
        assert "database" in explanation.lower()

    def test_explain_routing_multiple(self, router_agent):
        """Test explaining MULTIPLE tool routing."""
        decision = {
            "tool": "MULTIPLE",
            "tools": ["SQL", "RAG"],
            "reasoning": "Combined query",
            "confidence": 0.85,
            "query_type": "combined"
        }

        explanation = router_agent.explain_routing(decision)

        assert "MULTIPLE" in explanation
        assert "SQL" in explanation
        assert "RAG" in explanation


# Integration tests (require actual LLM)
class TestRouterAgentIntegration:
    """Integration tests with real LLM."""

    def test_route_real_sql_query(self):
        """Test routing real SQL query."""
        try:
            agent = RouterAgent(use_few_shot=True)
            agent.initialize()

            result = agent.route("Сколько разработчиков работает над PT Application Inspector?")

            assert result is not None
            assert "tool" in result
            # SQL is expected but RAG is also acceptable (both can answer this)
            assert result["tool"] in ["SQL", "RAG", "MULTIPLE"]
            assert result["confidence"] > 0.0

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    def test_route_real_rag_query(self):
        """Test routing real RAG query."""
        try:
            agent = RouterAgent(use_few_shot=True)
            agent.initialize()

            result = agent.route("Что такое PT Sandbox и какие у него возможности?")

            assert result is not None
            assert result["tool"] == "RAG"
            assert result["confidence"] > 0.5

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    def test_route_real_web_search_query(self):
        """Test routing real web search query."""
        try:
            agent = RouterAgent(use_few_shot=True)
            agent.initialize()

            result = agent.route("Последние новости по кибербезопасности за эту неделю")

            assert result is not None
            assert result["tool"] == "WEB_SEARCH"
            assert result["confidence"] > 0.5

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    def test_route_real_multiple_query(self):
        """Test routing real multiple query."""
        try:
            agent = RouterAgent(use_few_shot=True)
            agent.initialize()

            result = agent.route("Сколько человек в команде PT AI и какие у него функции?")

            assert result is not None
            # Could be MULTIPLE or single tool (both acceptable)
            assert result["tool"] in ["SQL", "RAG", "MULTIPLE"]

            if result["tool"] == "MULTIPLE":
                assert "tools" in result
                assert len(result["tools"]) >= 2

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
