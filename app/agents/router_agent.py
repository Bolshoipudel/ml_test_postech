"""Агент-маршрутизатор для классификации запросов и выбора инструментов."""
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from app.services.llm_factory import get_llm
from app.prompts.router_prompts import get_router_prompt, get_router_prompt_with_examples


class RouterAgent:
    """
    Агент-маршрутизатор для классификации запросов и выбора подходящих инструментов.

    Использует LLM для маршрутизации запросов к:
    - SQL Agent (запросы к базе данных)
    - RAG Agent (поиск по документации)
    - Web Search Agent (поиск в интернете)
    - Нескольким агентам (комбинированные запросы)
    """

    def __init__(self, use_few_shot: bool = True):
        """Инициализация агента-маршрутизатора.

        Args:
            use_few_shot: Использовать ли few-shot примеры в промптах
        """
        self.llm = None
        self._initialized = False
        self.use_few_shot = use_few_shot

    def initialize(self):
        """Инициализация Router Agent."""
        try:
            logger.info("Initializing Router Agent...")

            # Получение LLM с temperature=0 для стабильной классификации
            self.llm = get_llm(temperature=0.0)

            self._initialized = True
            logger.success("Router Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Router Agent: {e}")
            raise

    def route(self, query: str) -> Dict[str, Any]:
        """
        Маршрутизация запроса к подходящему инструменту.

        Args:
            query: Запрос пользователя для классификации

        Returns:
            Словарь с решением о маршрутизации:
            {
                "tool": "SQL" | "RAG" | "WEB_SEARCH" | "MULTIPLE",
                "tools": ["SQL", "RAG"],  # для MULTIPLE
                "reasoning": "explanation",
                "confidence": 0.85,
                "query_type": "description"
            }
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"Routing query: {query}")

            # Получение промпта
            if self.use_few_shot:
                prompt = get_router_prompt_with_examples(query)
            else:
                prompt = get_router_prompt(query)

            # Получение ответа от LLM
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Парсинг JSON-ответа
            routing_decision = self._parse_routing_response(response_text)

            # Валидация решения о маршрутизации
            routing_decision = self._validate_routing_decision(routing_decision, query)

            logger.info(
                f"Routing decision: tool={routing_decision['tool']}, "
                f"confidence={routing_decision['confidence']:.2f}, "
                f"reasoning={routing_decision['reasoning'][:100]}..."
            )

            return routing_decision

        except Exception as e:
            logger.error(f"Error routing query: {e}")
            # Резервный вариант: RAG (наиболее универсальный инструмент)
            return self._fallback_routing(query, str(e))

    def _parse_routing_response(self, response_text: str) -> Dict[str, Any]:
        """
        Парсинг ответа LLM для извлечения решения о маршрутизации.

        Args:
            response_text: Текст ответа LLM

        Returns:
            Распарсенное решение о маршрутизации

        Raises:
            ValueError: Если ответ не может быть распарсен
        """
        # Поиск JSON в ответе
        response_text = response_text.strip()

        # Удаление markdown блоков кода
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        # Поиск JSON-объекта
        if not response_text.startswith("{"):
            # Попытка найти первую {
            start = response_text.find("{")
            if start != -1:
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

        try:
            routing_decision = json.loads(response_text)
            return routing_decision
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _validate_routing_decision(
        self,
        decision: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """
        Валидация и нормализация решения о маршрутизации.

        Args:
            decision: Распарсенное решение о маршрутизации
            query: Исходный запрос

        Returns:
            Валидированное и нормализованное решение
        """
        # Проверка обязательных полей
        if "tool" not in decision:
            logger.warning("Missing 'tool' field in routing decision")
            return self._fallback_routing(query, "Missing 'tool' field")

        # Нормализация имени инструмента
        tool = decision["tool"].upper()
        valid_tools = ["SQL", "RAG", "WEB_SEARCH", "MULTIPLE"]

        if tool not in valid_tools:
            logger.warning(f"Invalid tool: {tool}, falling back to RAG")
            return self._fallback_routing(query, f"Invalid tool: {tool}")

        # Установка значений по умолчанию
        decision["tool"] = tool
        decision.setdefault("reasoning", "No reasoning provided")
        decision.setdefault("confidence", 0.5)
        decision.setdefault("query_type", "unknown")

        # Валидация уверенности
        try:
            confidence = float(decision["confidence"])
            decision["confidence"] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            decision["confidence"] = 0.5

        # Обработка MULTIPLE инструмента
        if tool == "MULTIPLE":
            if "tools" not in decision or not isinstance(decision["tools"], list):
                logger.warning("MULTIPLE tool requires 'tools' list")
                # Попытка вывести инструменты из запроса
                decision["tools"] = self._infer_tools_from_query(query)

            # Валидация списка инструментов
            decision["tools"] = [
                t.upper() for t in decision["tools"]
                if t.upper() in ["SQL", "RAG", "WEB_SEARCH"]
            ]

            if not decision["tools"]:
                logger.warning("No valid tools in MULTIPLE, falling back to RAG")
                return self._fallback_routing(query, "Empty tools list")

        return decision

    def _infer_tools_from_query(self, query: str) -> List[str]:
        """
        Определение необходимых инструментов на основе ключевых слов запроса.

        Args:
            query: Запрос пользователя

        Returns:
            Список выведенных инструментов
        """
        query_lower = query.lower()
        tools = []

        # Проверка SQL ключевых слов
        sql_keywords = ['сколько', 'кто работает', 'команда', 'разработчик', 'инцидент', 'статистика']
        if any(kw in query_lower for kw in sql_keywords):
            tools.append("SQL")

        # Проверка RAG ключевых слов
        rag_keywords = ['что такое', 'как работает', 'возможности', 'функции', 'описание', 'документация']
        if any(kw in query_lower for kw in rag_keywords):
            tools.append("RAG")

        # Проверка ключевых слов веб-поиска
        web_keywords = ['новости', 'актуальн', 'тренд', 'последние', 'сейчас']
        if any(kw in query_lower for kw in web_keywords):
            tools.append("WEB_SEARCH")

        # По умолчанию RAG
        if not tools:
            tools = ["RAG"]

        return tools

    def _fallback_routing(self, query: str, error_message: str) -> Dict[str, Any]:
        """
        Резервная маршрутизация при ошибке.

        Args:
            query: Запрос пользователя
            error_message: Сообщение об ошибке

        Returns:
            Резервное решение о маршрутизации (по умолчанию RAG)
        """
        logger.warning(f"Using fallback routing due to: {error_message}")

        # Простая маршрутизация на основе ключевых слов
        query_lower = query.lower()

        # Проверка явных индикаторов SQL
        if any(kw in query_lower for kw in ['сколько', 'count', 'количество', 'статистика']):
            return {
                "tool": "SQL",
                "reasoning": f"Fallback routing: query contains SQL keywords. Error: {error_message}",
                "confidence": 0.6,
                "query_type": "fallback_sql"
            }

        # Проверка явных индикаторов веб-поиска
        if any(kw in query_lower for kw in ['новости', 'news', 'тренд', 'trend']):
            return {
                "tool": "WEB_SEARCH",
                "reasoning": f"Fallback routing: query contains web search keywords. Error: {error_message}",
                "confidence": 0.6,
                "query_type": "fallback_web_search"
            }

        # По умолчанию RAG (наиболее универсальный)
        return {
            "tool": "RAG",
            "reasoning": f"Fallback routing: defaulting to RAG. Error: {error_message}",
            "confidence": 0.5,
            "query_type": "fallback_rag"
        }

    def route_with_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Маршрутизация запроса с учетом контекста беседы.

        Args:
            query: Запрос пользователя
            conversation_history: Предыдущие сообщения беседы

        Returns:
            Решение о маршрутизации с учетом контекста
        """
        # TODO: реализовать контекстно-зависимую маршрутизацию
        return self.route(query)

    def explain_routing(self, routing_decision: Dict[str, Any]) -> str:
        """
        Генерация читаемого объяснения решения о маршрутизации.

        Args:
            routing_decision: Решение о маршрутизации из route()

        Returns:
            Читаемое объяснение
        """
        tool = routing_decision.get("tool", "UNKNOWN")
        reasoning = routing_decision.get("reasoning", "No reasoning provided")
        confidence = routing_decision.get("confidence", 0.0)

        explanation = f"Выбран инструмент: {tool}\n"
        explanation += f"Уверенность: {confidence:.0%}\n"
        explanation += f"Обоснование: {reasoning}"

        if tool == "MULTIPLE":
            tools = routing_decision.get("tools", [])
            explanation += f"\nИспользуемые инструменты: {', '.join(tools)}"

        return explanation


# Глобальный экземпляр агента-маршрутизатора
router_agent = RouterAgent(use_few_shot=True)
