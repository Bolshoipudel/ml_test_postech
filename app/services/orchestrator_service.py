"""Сервис-оркестратор для координации множественных агентов."""
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from app.agents.router_agent import router_agent
from app.agents.sql_agent import sql_agent
from app.agents.rag_agent import rag_agent
from app.agents.web_search_agent import web_search_agent
from app.services.llm_factory import get_llm


class OrchestratorService:
    """
    Сервис-оркестратор для координации множественных агентов.

    Обязанности:
    1. Маршрутизация запросов через Router Agent
    2. Выполнение подходящих агентов
    3. Агрегация результатов от нескольких агентов
    4. Форматирование финального ответа
    """

    def __init__(self):
        self.router = router_agent
        self.llm = None
        self._initialized = False

    def initialize(self):
        try:
            logger.info("Initializing Orchestrator Service...")

            # Инициализация маршрутизатора
            if not self.router._initialized:
                self.router.initialize()

            # Получение LLM для агрегации результатов
            self.llm = get_llm(temperature=0.3)

            self._initialized = True
            logger.success("Orchestrator Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator Service: {e}")
            raise

    async def process_query(
        self,
        query: str,
        use_history: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Обработка запроса пользователя через подходящих агентов.

        Args:
            query: Запрос пользователя
            use_history: Использовать ли историю беседы
            conversation_history: Предыдущие сообщения беседы

        Returns:
            Словарь с ответом и метаданными:
            {
                "success": bool,
                "answer": str,
                "tools_used": List[Dict],
                "sources": List[str],
                "routing_decision": Dict
            }
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"Processing query: {query[:100]}...")

            # Шаг 1: Маршрутизация запроса
            routing_decision = self.router.route(query)

            logger.info(
                f"Routing: tool={routing_decision['tool']}, "
                f"confidence={routing_decision['confidence']:.2f}"
            )

            # Шаг 2: Выполнение подходящих агентов
            tool = routing_decision["tool"]

            if tool == "MULTIPLE":
                # Выполнение нескольких агентов
                result = await self._execute_multiple_agents(query, routing_decision)
            else:
                # Выполнение одного агента
                result = await self._execute_single_agent(query, tool)

            # Шаг 3: Добавление метаданных маршрутизации
            result["routing_decision"] = routing_decision

            logger.success(f"Query processed successfully using {tool}")

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "tools_used": [],
                "sources": [],
                "error": str(e)
            }

    async def _execute_single_agent(
        self,
        query: str,
        tool: str
    ) -> Dict[str, Any]:
        """
        Выполнение одного агента.

        Args:
            query: Запрос пользователя
            tool: Инструмент для использования (SQL, RAG, WEB_SEARCH)

        Returns:
            Результат агента
        """
        if tool == "SQL":
            return await self._execute_sql_agent(query)
        elif tool == "RAG":
            return await self._execute_rag_agent(query)
        elif tool == "WEB_SEARCH":
            return await self._execute_web_search_agent(query)
        else:
            # Резервный вариант: RAG
            logger.warning(f"Unknown tool: {tool}, falling back to RAG")
            return await self._execute_rag_agent(query)

    async def _execute_multiple_agents(
        self,
        query: str,
        routing_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Выполнение нескольких агентов параллельно и агрегация результатов.

        Args:
            query: Запрос пользователя
            routing_decision: Решение о маршрутизации со списком инструментов

        Returns:
            Агрегированный результат
        """
        tools = routing_decision.get("tools", [])

        if not tools:
            logger.warning("No tools specified for MULTIPLE, falling back to RAG")
            return await self._execute_rag_agent(query)

        logger.info(f"Executing multiple agents: {tools}")

        # Создание задач для каждого агента
        tasks = []
        for tool in tools:
            if tool == "SQL":
                tasks.append(self._execute_sql_agent(query))
            elif tool == "RAG":
                tasks.append(self._execute_rag_agent(query))
            elif tool == "WEB_SEARCH":
                tasks.append(self._execute_web_search_agent(query))

        # Параллельное выполнение всех агентов
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Фильтрация исключений
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {tools[i]} failed: {result}")
            elif result.get("success", False):
                valid_results.append(result)

        if not valid_results:
            return {
                "success": False,
                "answer": "Не удалось получить результаты ни от одного агента.",
                "tools_used": [],
                "sources": []
            }

        # Агрегация результатов
        aggregated = await self._aggregate_results(query, valid_results, tools)

        return aggregated

    async def _execute_sql_agent(self, query: str) -> Dict[str, Any]:
        try:
            # Инициализация при необходимости
            if not sql_agent._initialized:
                sql_agent.initialize()

            # Выполнение запроса
            result = sql_agent.execute_query(query, validate=True)

            if result["success"]:
                # Форматирование результатов
                formatted_answer = sql_agent.format_results(
                    result["results"],
                    query
                )

                return {
                    "success": True,
                    "answer": formatted_answer,
                    "tool": "SQL",
                    "tools_used": [{
                        "tool_type": "sql",
                        "query": result["sql_query"],
                        "result_summary": f"Найдено {result['row_count']} записей",
                        "metadata": {
                            "row_count": result["row_count"],
                            "sql_query": result["sql_query"]
                        }
                    }],
                    "sources": ["database: PostgreSQL"]
                }
            else:
                return {
                    "success": False,
                    "answer": f"SQL Agent: {result.get('error', 'Unknown error')}",
                    "tool": "SQL",
                    "tools_used": [],
                    "sources": []
                }

        except Exception as e:
            logger.error(f"SQL Agent error: {e}")
            return {
                "success": False,
                "answer": f"Ошибка SQL Agent: {str(e)}",
                "tool": "SQL",
                "tools_used": [],
                "sources": []
            }

    async def _execute_rag_agent(self, query: str) -> Dict[str, Any]:
        try:
            # Инициализация при необходимости
            if not rag_agent._initialized:
                rag_agent.initialize(load_docs=False)

            # Ответ на вопрос
            result = rag_agent.answer_question(query, top_k=5)

            if result["success"]:
                return {
                    "success": True,
                    "answer": result["answer"],
                    "tool": "RAG",
                    "tools_used": [{
                        "tool_type": "rag",
                        "query": query,
                        "result_summary": f"Найдено {result['relevant_chunks']} релевантных документов",
                        "metadata": {
                            "retrieved_chunks": result["retrieved_chunks"],
                            "relevant_chunks": result["relevant_chunks"],
                            "top_similarity": result.get("top_similarity", 0.0)
                        }
                    }],
                    "sources": [f"documentation: {src}" for src in result["sources"]]
                }
            else:
                return {
                    "success": False,
                    "answer": result["answer"],
                    "tool": "RAG",
                    "tools_used": [],
                    "sources": []
                }

        except Exception as e:
            logger.error(f"RAG Agent error: {e}")
            return {
                "success": False,
                "answer": f"Ошибка RAG Agent: {str(e)}",
                "tool": "RAG",
                "tools_used": [],
                "sources": []
            }

    async def _execute_web_search_agent(self, query: str) -> Dict[str, Any]:
        try:
            # Инициализация при необходимости
            if not web_search_agent._initialized:
                web_search_agent.initialize()

            # Проверка на запрос новостей
            query_lower = query.lower()
            is_news_query = any(kw in query_lower for kw in ['новости', 'последние новости'])

            # Поиск
            if is_news_query:
                result = web_search_agent.search_news(query, max_results=5, days=7)
            else:
                result = web_search_agent.search_and_answer(query, max_results=5)

            if result["success"]:
                return {
                    "success": True,
                    "answer": result["answer"],
                    "tool": "WEB_SEARCH",
                    "tools_used": [{
                        "tool_type": "web_search",
                        "query": query,
                        "result_summary": f"Найдено {result.get('search_results', result.get('articles_found', 0))} результатов",
                        "metadata": {
                            "search_results": result.get("search_results", result.get("articles_found", 0)),
                            "top_score": result.get("top_score", 0)
                        }
                    }],
                    "sources": [f"web: {src}" for src in result["sources"][:3]]
                }
            else:
                return {
                    "success": False,
                    "answer": result["answer"],
                    "tool": "WEB_SEARCH",
                    "tools_used": [],
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Web Search Agent error: {e}")
            return {
                "success": False,
                "answer": f"Ошибка Web Search Agent: {str(e)}",
                "tool": "WEB_SEARCH",
                "tools_used": [],
                "sources": []
            }

    async def _aggregate_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        tools: List[str]
    ) -> Dict[str, Any]:
        """
        Агрегация результатов от нескольких агентов.

        Args:
            query: Исходный запрос
            results: Список результатов агентов
            tools: Список использованных инструментов

        Returns:
            Агрегированный результат
        """
        logger.info(f"Aggregating results from {len(results)} agents")

        # Сбор ответов
        answers = []
        all_tools_used = []
        all_sources = []

        for result in results:
            tool = result.get("tool", "UNKNOWN")
            answer = result.get("answer", "")

            if answer:
                answers.append(f"[{tool}] {answer}")

            all_tools_used.extend(result.get("tools_used", []))
            all_sources.extend(result.get("sources", []))

        # Использование LLM для синтеза финального ответа
        try:
            synthesized_answer = await self._synthesize_answer(query, answers, tools)
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            # Резервный вариант: простая конкатенация
            synthesized_answer = "\n\n".join(answers)

        return {
            "success": True,
            "answer": synthesized_answer,
            "tools_used": all_tools_used,
            "sources": list(set(all_sources))  # Remove duplicates
        }

    async def _synthesize_answer(
        self,
        query: str,
        answers: List[str],
        tools: List[str]
    ) -> str:
        """
        Синтез финального ответа из нескольких ответов агентов.

        Args:
            query: Исходный запрос
            answers: Список ответов от агентов
            tools: Список использованных инструментов

        Returns:
            Синтезированный ответ
        """
        if len(answers) == 1:
            # Только один ответ, удаляем префикс инструмента
            return answers[0].split("] ", 1)[-1] if "] " in answers[0] else answers[0]

        # Создание промпта для синтеза
        answers_text = "\n\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

        prompt = f"""Ты - помощник, который создает единый ответ на основе информации от разных источников.

ЗАДАЧА: Объедини информацию из нескольких источников в один связный и структурированный ответ.

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

ИНФОРМАЦИЯ ИЗ РАЗНЫХ ИСТОЧНИКОВ:
{answers_text}

ПРАВИЛА:
1. Создай единый связный ответ, объединив всю информацию
2. Структурируй ответ логично (сначала количественные данные, потом описания)
3. Укажи источники информации там, где это уместно
4. Не добавляй информацию, которой нет в источниках
5. Отвечай на русском языке
6. Если информация противоречит, укажи на это

ЕДИНЫЙ ОТВЕТ:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error in LLM synthesis: {e}")
            # Резервный вариант: конкатенация с разделителями
            return "\n\n".join([ans.split("] ", 1)[-1] if "] " in ans else ans for ans in answers])


# Глобальный экземпляр оркестратора
orchestrator = OrchestratorService()
