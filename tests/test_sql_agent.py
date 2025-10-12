"""Tests for SQL Agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.agents.sql_agent import SQLAgent
from app.services.database_service import DatabaseService


class TestSQLAgent:
    """Test SQL Agent functionality."""

    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        with patch('app.agents.sql_agent.db_service') as mock:
            mock._initialized = True
            mock.get_table_info_for_llm.return_value = """
            Database Schema:
            Table: team_members
            Columns:
              - id: INTEGER NOT NULL (PRIMARY KEY)
              - first_name: VARCHAR(50) NOT NULL
              - last_name: VARCHAR(50) NOT NULL
            """
            mock.validate_query.return_value = (True, "Query is valid")
            mock.execute_query.return_value = [
                {"first_name": "Ivan", "last_name": "Petrov"},
                {"first_name": "Olga", "last_name": "Sidorova"}
            ]
            yield mock

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM."""
        with patch('app.agents.sql_agent.get_llm') as mock:
            mock_response = MagicMock()
            mock_response.content = "SELECT first_name, last_name FROM team_members LIMIT 2;"
            mock.return_value.invoke.return_value = mock_response
            yield mock

    def test_sql_extraction_with_markdown(self):
        """Test SQL extraction from markdown code blocks."""
        agent = SQLAgent()

        # Test with ```sql
        response1 = "```sql\nSELECT * FROM users;\n```"
        result1 = agent._extract_sql(response1)
        assert result1 == "SELECT * FROM users"

        # Test with ```
        response2 = "```\nSELECT * FROM products;\n```"
        result2 = agent._extract_sql(response2)
        assert result2 == "SELECT * FROM products"

        # Test without markdown
        response3 = "SELECT * FROM orders;"
        result3 = agent._extract_sql(response3)
        assert result3 == "SELECT * FROM orders"

    def test_generate_sql(self, mock_db_service, mock_llm):
        """Test SQL generation from natural language."""
        agent = SQLAgent()
        agent._initialized = True
        agent.llm = mock_llm.return_value
        agent.db_info = mock_db_service.get_table_info_for_llm()

        question = "Show me all team members"
        sql = agent.generate_sql(question)

        assert "SELECT" in sql.upper()
        assert "team_members" in sql.lower()

    def test_execute_query_success(self, mock_db_service, mock_llm):
        """Test successful query execution."""
        agent = SQLAgent()
        agent._initialized = True
        agent.llm = mock_llm.return_value
        agent.db_info = mock_db_service.get_table_info_for_llm()

        result = agent.execute_query("Show me all team members")

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert result["row_count"] == 2
        assert "sql_query" in result

    def test_execute_query_invalid_sql(self, mock_db_service, mock_llm):
        """Test query execution with invalid SQL."""
        agent = SQLAgent()
        agent._initialized = True
        agent.llm = mock_llm.return_value
        agent.db_info = mock_db_service.get_table_info_for_llm()

        # Mock invalid query
        mock_db_service.validate_query.return_value = (False, "DELETE not allowed")

        mock_response = MagicMock()
        mock_response.content = "DELETE FROM team_members;"
        agent.llm.invoke.return_value = mock_response

        result = agent.execute_query("Delete all team members", validate=True)

        assert result["success"] is False
        assert "not allowed" in result["error"].lower()


class TestDatabaseService:
    """Test Database Service."""

    def test_validate_query_select_allowed(self):
        """Test that SELECT queries are allowed."""
        service = DatabaseService()

        is_valid, msg = service.validate_query("SELECT * FROM users;")
        assert is_valid is True

    def test_validate_query_delete_blocked(self):
        """Test that DELETE queries are blocked."""
        service = DatabaseService()

        is_valid, msg = service.validate_query("DELETE FROM users;")
        assert is_valid is False
        assert "DELETE" in msg

    def test_validate_query_drop_blocked(self):
        """Test that DROP queries are blocked."""
        service = DatabaseService()

        is_valid, msg = service.validate_query("DROP TABLE users;")
        assert is_valid is False
        assert "DROP" in msg

    def test_validate_query_dangerous_patterns(self):
        """Test that dangerous patterns are blocked."""
        service = DatabaseService()

        is_valid, msg = service.validate_query("SELECT * FROM users; EXEC xp_cmdshell;")
        assert is_valid is False
        assert "dangerous" in msg.lower() or "EXEC" in msg
