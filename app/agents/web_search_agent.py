"""Агент веб-поиска для поиска информации в интернете."""
from typing import Dict, Any, List
from loguru import logger

from app.services.search_service import search_service
from app.services.llm_factory import get_llm


class WebSearchAgent:
    """Агент для поиска в интернете и генерации ответов на основе результатов."""

    def __init__(self):
        self.search_service = None
        self.llm = None
        self._initialized = False

    def initialize(self):
        """Инициализация агента веб-поиска."""
        try:
            logger.info("Initializing Web Search Agent...")

            # Инициализация поискового сервиса
            if not search_service._initialized:
                search_service.initialize()
            self.search_service = search_service

            # Получение LLM с низкой температурой для фактических ответов
            self.llm = get_llm(temperature=0.3)

            self._initialized = True
            logger.success("Web Search Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Web Search Agent: {e}")
            raise

    def search_and_answer(
        self,
        question: str,
        max_results: int = 5,
        search_depth: str = "basic"
    ) -> Dict[str, Any]:
        """
        Поиск в интернете и генерация ответа.

        Args:
            question: Вопрос пользователя
            max_results: Максимальное количество результатов
            search_depth: "basic" или "advanced"

        Returns:
            Словарь с ответом и метаданными
        """
        if not self._initialized:
            self.initialize()

        if not self.search_service._initialized:
            return {
                "success": False,
                "answer": "Веб-поиск недоступен. Не настроен API ключ для Tavily. Получите ключ на https://tavily.com/",
                "sources": [],
                "search_results": []
            }

        try:
            logger.info(f"Web Search Agent processing question: {question}")

            # Выполнение поиска
            search_results = self.search_service.search(
                query=question,
                max_results=max_results,
                search_depth=search_depth
            )

            if not search_results:
                return {
                    "success": False,
                    "answer": "К сожалению, я не нашел релевантной информации в интернете по вашему запросу.",
                    "sources": [],
                    "search_results": []
                }

            # Форматирование контекста из результатов
            context = self._format_search_context(search_results)

            # Генерация ответа через LLM
            answer = self._generate_answer(question, context, search_results)

            # Извлечение источников
            sources = self._extract_sources(search_results)

            logger.success(f"Generated answer using {len(search_results)} search results")

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "search_results": len(search_results),
                "top_score": search_results[0].get("score", 0) if search_results else 0
            }

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "success": False,
                "answer": f"Произошла ошибка при поиске в интернете: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def search_news(
        self,
        query: str,
        max_results: int = 5,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Поиск последних новостей и генерация сводки.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            days: Поиск за последние N дней

        Returns:
            Словарь со сводкой новостей и источниками
        """
        if not self._initialized:
            self.initialize()

        if not self.search_service._initialized:
            return {
                "success": False,
                "answer": "Поиск новостей недоступен. Не настроен API ключ для Tavily.",
                "sources": []
            }

        try:
            logger.info(f"Searching news for: {query}")

            # Поиск новостей
            news_results = self.search_service.search_news(
                query=query,
                max_results=max_results,
                days=days
            )

            if not news_results:
                return {
                    "success": False,
                    "answer": f"Не найдено новостей по запросу '{query}' за последние {days} дней.",
                    "sources": []
                }

            # Форматирование контекста
            context = self._format_search_context(news_results)

            # Генерация сводки новостей
            answer = self._generate_news_summary(query, context, news_results)

            # Извлечение источников
            sources = self._extract_sources(news_results)

            logger.success(f"Generated news summary from {len(news_results)} articles")

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "articles_found": len(news_results)
            }

        except Exception as e:
            logger.error(f"Error in news search: {e}")
            return {
                "success": False,
                "answer": f"Произошла ошибка при поиске новостей: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def _format_search_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирование результатов поиска в контекст для LLM.

        Args:
            results: Результаты поиска

        Returns:
            Отформатированная строка контекста
        """
        context_parts = []

        # Добавление AI-сводки Tavily, если доступна
        if results and "tavily_answer" in results[0]:
            context_parts.append(f"AI Summary: {results[0]['tavily_answer']}\n")

        # Добавление отдельных результатов
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Источник {i}] {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Содержание: {result['content']}\n"
                f"Релевантность: {result['score']:.2f}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer(
        self,
        question: str,
        context: str,
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Генерация ответа с использованием LLM и результатов поиска.

        Args:
            question: Вопрос пользователя
            context: Отформатированный контекст поиска
            search_results: Сырые результаты поиска

        Returns:
            Сгенерированный ответ
        """
        # Проверка наличия AI-ответа от Tavily
        tavily_answer = search_results[0].get("tavily_answer", "") if search_results else ""

        prompt = f"""Ты - помощник, который отвечает на вопросы пользователя на основе результатов поиска в интернете.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленных результатов поиска
2. Указывай источники информации (упоминай названия сайтов)
3. Если информации недостаточно для полного ответа, скажи об этом
4. Не придумывай информацию, которой нет в результатах поиска
5. Отвечай на русском языке, четко и структурированно
6. Если есть противоречия в источниках, укажи на это
7. Укажи дату публикации, если это важно (для новостей)

РЕЗУЛЬТАТЫ ПОИСКА:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ОТВЕТ:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Резервный вариант: использовать ответ Tavily
            if tavily_answer:
                return f"На основе поиска: {tavily_answer}"
            raise

    def _generate_news_summary(
        self,
        query: str,
        context: str,
        news_results: List[Dict[str, Any]]
    ) -> str:
        """
        Генерация сводки новостей с использованием LLM.

        Args:
            query: Поисковый запрос
            context: Отформатированный контекст новостей
            news_results: Результаты поиска новостей

        Returns:
            Сводка новостей
        """
        prompt = f"""Ты - новостной аналитик. Создай краткую сводку новостей по запросу пользователя.

ПРАВИЛА:
1. Суммируй основные новости из предоставленных источников
2. Укажи даты публикации, если они есть
3. Упомяни ключевые факты и цифры
4. Структурируй ответ по пунктам
5. Отвечай на русском языке
6. Укажи источники (названия изданий)

НОВОСТИ:
{context}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {query}

СВОДКА:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating news summary: {e}")
            raise

    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Извлечение URL источников из результатов поиска.

        Args:
            results: Результаты поиска

        Returns:
            Список URL источников
        """
        sources = []
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            if url:
                sources.append(f"{title} ({url})")

        return sources


# Глобальный экземпляр агента веб-поиска
web_search_agent = WebSearchAgent()
