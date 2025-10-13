"""SQL агент для преобразования естественного языка в SQL."""
from typing import Dict, Any, Optional, List
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from loguru import logger

from app.services.database_service import db_service
from app.services.llm_factory import get_llm


class SQLAgent:
    """Агент для преобразования запросов на естественном языке в SQL."""

    def __init__(self):
        self.llm = None
        self.db_info = None
        self._initialized = False

    def initialize(self):
        """Инициализация SQL-агента."""
        try:
            logger.info("Initializing SQL Agent...")

            # Инициализация сервиса БД
            if not db_service._initialized:
                db_service.initialize()

            # Получение LLM с низкой температурой для SQL
            self.llm = get_llm(temperature=0.0)

            # Получение схемы БД
            self.db_info = db_service.get_table_info_for_llm()

            self._initialized = True
            logger.success("SQL Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SQL Agent: {e}")
            raise

    def generate_sql(self, question: str) -> str:
        """
        Генерация SQL запроса из вопроса на естественном языке.

        Args:
            question: Вопрос на естественном языке

        Returns:
            Сгенерированный SQL запрос
        """
        if not self._initialized:
            self.initialize()

        try:
            prompt = self._create_sql_prompt(question)
            response = self.llm.invoke(prompt)
            sql_query = self._extract_sql(response.content if hasattr(response, 'content') else str(response))

            logger.info(f"Generated SQL: {sql_query}")
            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise

    def _create_sql_prompt(self, question: str) -> str:
        prompt = f"""You are a SQL expert. Given the database schema below, write a SQL query to answer the user's question.

{self.db_info}

Rules:
1. Only use SELECT statements
2. Use proper JOINs when querying multiple tables
3. Use appropriate WHERE clauses for filtering
4. Use GROUP BY for aggregations
5. Return only the SQL query without any explanations
6. Start the query with SELECT and end with semicolon
7. Use table and column names exactly as shown in the schema

Question: {question}

SQL Query:"""
        return prompt

    def _extract_sql(self, response: str) -> str:
        # Удаление markdown блоков кода
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            sql = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            sql = response[start:end].strip()
        else:
            sql = response.strip()

        # Удаление завершающей точки с запятой (добавим позже)
        sql = sql.rstrip(";").strip()

        # Проверка что запрос начинается с SELECT
        if not sql.upper().startswith("SELECT"):
            lines = sql.split("\n")
            for line in lines:
                if line.strip().upper().startswith("SELECT"):
                    sql = line.strip()
                    break

        return sql

    def execute_query(
        self,
        question: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Выполнение запроса на естественном языке к БД.

        Args:
            question: Вопрос на естественном языке
            validate: Валидировать сгенерированный SQL

        Returns:
            Словарь с результатами и метаданными
        """
        if not self._initialized:
            self.initialize()

        try:
            sql_query = self.generate_sql(question)

            if validate:
                is_valid, error_msg = db_service.validate_query(sql_query)
                if not is_valid:
                    logger.warning(f"Generated invalid SQL: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "sql_query": sql_query,
                        "results": []
                    }

            results = db_service.execute_query(sql_query)
            logger.success(f"Query executed successfully, returned {len(results)} rows")

            return {
                "success": True,
                "sql_query": sql_query,
                "results": results,
                "row_count": len(results)
            }

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": None,
                "results": []
            }

    def format_results(
        self,
        results: List[Dict[str, Any]],
        question: str,
        max_rows: int = 10
    ) -> str:
        """
        Форматирование результатов запроса в читаемый текст.

        Args:
            results: Результаты запроса
            question: Исходный вопрос
            max_rows: Максимум строк в резюме

        Returns:
            Отформатированная строка
        """
        if not results:
            return "Запрос не вернул результатов."

        display_results = results[:max_rows]
        total_rows = len(results)

        if not self._initialized:
            self.initialize()

        try:
            results_text = "\n".join([str(row) for row in display_results])

            prompt = f"""Based on the following query results, provide a clear and concise answer to the user's question.

Question: {question}

Results ({total_rows} total rows, showing first {len(display_results)}):
{results_text}

Provide a natural language answer that directly addresses the question. Be specific with numbers and names.
Answer in Russian if the question is in Russian, otherwise in English.

Answer:"""

            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return f"Найдено {total_rows} записей. Первые результаты:\n{results_text}"


# Глобальный экземпляр SQL агента
sql_agent = SQLAgent()
