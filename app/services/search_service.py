"""Сервис веб-поиска через Tavily API."""
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily package not available. Install with: pip install tavily-python")

from app.config import settings


class SearchService:
    """Сервис веб-поиска через Tavily API."""

    def __init__(self):
        self.client = None
        self._initialized = False

    def initialize(self):
        try:
            if not TAVILY_AVAILABLE:
                raise ImportError("Tavily package is not installed")

            if not settings.tavily_api_key or settings.tavily_api_key == "your_tavily_api_key_here":
                logger.warning("Tavily API key not configured. Web search will not work.")
                logger.info("Get your API key at: https://tavily.com/")
                return

            logger.info("Initializing Tavily search client...")
            self.client = TavilyClient(api_key=settings.tavily_api_key)
            self._initialized = True
            logger.success("Tavily search client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize search service: {e}")
            raise

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск в интернете через Tavily API.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            search_depth: "basic" или "advanced" (advanced расходует больше кредитов)
            include_domains: Список доменов для включения
            exclude_domains: Список доменов для исключения

        Returns:
            Список результатов поиска с title, url, content, score
        """
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.client:
            logger.error("Search service not initialized. Cannot perform search.")
            return []

        try:
            logger.info(f"Searching web for: '{query}'")

            # Подготовка параметров поиска
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": True,  # Получение AI-ответа
                "include_raw_content": False,  # Без полного HTML
            }

            if include_domains:
                search_params["include_domains"] = include_domains

            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            # Выполнение поиска
            response = self.client.search(**search_params)

            # Парсинг результатов
            results = self._parse_results(response)

            logger.info(f"Found {len(results)} search results")

            return results

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []

    def _parse_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Парсинг ответа Tavily API.

        Args:
            response: Сырой ответ Tavily API

        Returns:
            Распарсенные результаты поиска
        """
        results = []

        # Извлечение AI-ответа, если доступен
        answer = response.get("answer", "")

        # Извлечение результатов поиска
        for result in response.get("results", []):
            parsed_result = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date"),
            }
            results.append(parsed_result)

        # Добавление ответа как метаданных
        if results and answer:
            results[0]["tavily_answer"] = answer

        return results

    def search_news(
        self,
        query: str,
        max_results: int = 5,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Поиск последних новостей.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов
            days: Поиск за последние N дней

        Returns:
            Список новостных результатов
        """
        if not self._initialized:
            self.initialize()

        if not self._initialized or not self.client:
            logger.error("Search service not initialized. Cannot perform news search.")
            return []

        try:
            logger.info(f"Searching news for: '{query}' (last {days} days)")

            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                topic="news",  # Фокус на новостных источниках
                days=days
            )

            results = self._parse_results(response)

            logger.info(f"Found {len(results)} news articles")

            return results

        except Exception as e:
            logger.error(f"Error performing news search: {e}")
            return []

    def get_search_context(
        self,
        query: str,
        max_results: int = 5
    ) -> str:
        """
        Получение результатов поиска в формате контекста для LLM.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Отформатированная строка контекста
        """
        results = self.search(query, max_results=max_results)

        if not results:
            return "No search results found."

        context_parts = []

        # Добавление Tavily ответа
        if "tavily_answer" in results[0]:
            context_parts.append(f"Summary: {results[0]['tavily_answer']}\n")

        # Добавление отдельных результатов
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Result {i}] {result['title']}\n"
                f"Source: {result['url']}\n"
                f"Content: {result['content']}\n"
                f"Relevance: {result['score']:.2f}\n"
            )

        return "\n---\n".join(context_parts)

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация результатов по минимальному порогу релевантности.

        Args:
            results: Результаты поиска
            min_score: Минимальный порог релевантности (0-1)

        Returns:
            Отфильтрованные результаты
        """
        filtered = [r for r in results if r.get("score", 0) >= min_score]
        logger.debug(f"Filtered {len(results)} results to {len(filtered)} with min_score={min_score}")
        return filtered


# Глобальный экземпляр поискового сервиса
search_service = SearchService()
