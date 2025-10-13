"""Сервис базы данных для выполнения SQL-запросов."""
from typing import List, Dict, Any, Optional
import re
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from app.config import settings
from app.models.database import Base


class DatabaseService:
    """Сервис для операций с базой данных."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self):
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

            # Тестирование соединения
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self._initialized = True
            logger.success("Database connection established")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_session(self) -> Session:
        if not self._initialized:
            self.initialize()
        return self.SessionLocal()

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Выполнение SQL-запроса с возвратом результатов как список словарей.

        Args:
            query: SQL-запрос

        Returns:
            Список словарей с результатами запроса
        """
        if not self._initialized:
            self.initialize()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))

                # Преобразование в список словарей
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
        Получение информации о схеме базы данных.

        Args:
            table_name: Опциональное имя конкретной таблицы

        Returns:
            Словарь с информацией о схеме
        """
        if not self._initialized:
            self.initialize()

        inspector = inspect(self.engine)

        if table_name:
            # Получение схемы конкретной таблицы
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)

            return {
                "table": table_name,
                "columns": columns,
                "foreign_keys": foreign_keys
            }
        else:
            # Получение схемы всех таблиц
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
        Получение отформатированной информации о таблицах для контекста LLM.

        Returns:
            Отформатированная строка со схемами таблиц
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
        Валидация SQL-запроса (Guardrails).

        Args:
            query: SQL-запрос

        Returns:
            Кортеж (is_valid, error_message)
        """
        query_stripped = query.strip()
        query_upper = query_stripped.upper()

        # Проверка на пустой запрос
        if not query_stripped:
            return False, "Query cannot be empty"

        # Проверка 1: Первое слово должно быть из разрешенных операций
        first_word = query_stripped.split()[0].upper()
        allowed_ops = settings.allowed_sql_ops_list

        if first_word not in allowed_ops:
            return False, f"Query must start with one of: {', '.join(allowed_ops)}. Got: {first_word}"

        # Проверка 2: Поиск опасных команд SQL (word boundaries для точности)
        dangerous_commands = [
            r'\bDELETE\b', r'\bDROP\b', r'\bTRUNCATE\b',
            r'\bALTER\b', r'\bCREATE\b', r'\bINSERT\b',
            r'\bUPDATE\b', r'\bEXECUTE\b', r'\bEXEC\b',
            r'\bGRANT\b', r'\bREVOKE\b'
        ]

        for pattern in dangerous_commands:
            if re.search(pattern, query_upper):
                cmd = pattern.strip(r'\b').strip('\\')
                # Проверка, действительно ли команда разрешена
                if cmd not in allowed_ops:
                    return False, f"Dangerous operation '{cmd}' detected in query. Only {', '.join(allowed_ops)} operations are permitted."

        # Проверка 3: Поиск опасных паттернов stored procedures
        dangerous_patterns = [
            (r'xp_', 'xp_'),
            (r'sp_', 'sp_'),
        ]

        for pattern, name in dangerous_patterns:
            if re.search(pattern, query_upper):
                return False, f"Dangerous pattern '{name}' detected in query"

        # Проверка 4: Защита от SQL-инъекций через множественные выражения
        # Допускается точка с запятой только в конце
        semicolon_count = query_stripped.count(';')
        if semicolon_count > 1:
            return False, "Multiple SQL statements are not allowed"
        if semicolon_count == 1 and not query_stripped.rstrip().endswith(';'):
            return False, "Multiple SQL statements are not allowed"

        return True, "Query is valid"

    def close(self):
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Глобальный экземпляр сервиса базы данных
db_service = DatabaseService()
