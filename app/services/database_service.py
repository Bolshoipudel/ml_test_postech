"""Database service for executing SQL queries."""
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from app.config import settings
from app.models.database import Base


class DatabaseService:
    """Service for database operations."""

    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self):
        """Create database engine and session."""
        try:
            logger.info(f"Connecting to database: {settings.postgres_host}:{settings.postgres_port}")

            self.engine = create_engine(
                settings.db_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10
            )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self._initialized = True
            logger.success("Database connection established")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_session(self) -> Session:
        """Get database session."""
        if not self._initialized:
            self.initialize()
        return self.SessionLocal()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dicts.

        Args:
            query: SQL query string

        Returns:
            List of dictionaries with query results
        """
        if not self._initialized:
            self.initialize()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))

                # Convert to list of dicts
                columns = result.keys()
                rows = []
                for row in result:
                    rows.append(dict(zip(columns, row)))

                logger.info(f"Query executed successfully, returned {len(rows)} rows")
                return rows

        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise Exception(f"Database error: {str(e)}")

    def get_table_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database schema information.

        Args:
            table_name: Optional specific table name

        Returns:
            Dictionary with schema information
        """
        if not self._initialized:
            self.initialize()

        inspector = inspect(self.engine)

        if table_name:
            # Get specific table schema
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)

            return {
                "table": table_name,
                "columns": columns,
                "foreign_keys": foreign_keys
            }
        else:
            # Get all tables schema
            tables = inspector.get_table_names()
            schema = {}

            for table in tables:
                columns = inspector.get_columns(table)
                foreign_keys = inspector.get_foreign_keys(table)

                schema[table] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "primary_key": col.get("primary_key", False)
                        }
                        for col in columns
                    ],
                    "foreign_keys": foreign_keys
                }

            return schema

    def get_table_info_for_llm(self) -> str:
        """
        Get formatted table information for LLM context.

        Returns:
            Formatted string with table schemas
        """
        schema = self.get_table_schema()

        info_parts = ["Database Schema:\n"]

        for table_name, table_info in schema.items():
            info_parts.append(f"\nTable: {table_name}")
            info_parts.append("Columns:")

            for col in table_info["columns"]:
                pk = " (PRIMARY KEY)" if col.get("primary_key") else ""
                nullable = " NULL" if col["nullable"] else " NOT NULL"
                info_parts.append(f"  - {col['name']}: {col['type']}{nullable}{pk}")

            if table_info["foreign_keys"]:
                info_parts.append("Foreign Keys:")
                for fk in table_info["foreign_keys"]:
                    info_parts.append(
                        f"  - {fk['constrained_columns']} -> "
                        f"{fk['referred_table']}.{fk['referred_columns']}"
                    )

        return "\n".join(info_parts)

    def validate_query(self, query: str) -> tuple[bool, str]:
        """
        Validate SQL query (Guardrails).

        Args:
            query: SQL query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        query_upper = query.strip().upper()

        # Check if query contains only allowed operations
        allowed_ops = settings.allowed_sql_ops_list

        # Check for disallowed operations
        disallowed = ["DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]

        for op in disallowed:
            if op in query_upper and op not in allowed_ops:
                return False, f"Operation '{op}' is not allowed. Only {', '.join(allowed_ops)} operations are permitted."

        # Must start with SELECT
        if not any(query_upper.startswith(op) for op in allowed_ops):
            return False, f"Query must start with one of: {', '.join(allowed_ops)}"

        # Check for dangerous patterns
        dangerous_patterns = [
            "EXECUTE",
            "EXEC",
            "xp_",
            "sp_",
            "GRANT",
            "REVOKE"
        ]

        for pattern in dangerous_patterns:
            if pattern in query_upper:
                return False, f"Dangerous pattern '{pattern}' detected in query"

        return True, "Query is valid"

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database service instance
db_service = DatabaseService()
