"""RAG-агент для ответов на вопросы по документации."""
from typing import Dict, Any, List, Optional
from loguru import logger

from app.services.rag_service import rag_service
from app.services.llm_factory import get_llm


class RAGAgent:
    """Агент для ответов на вопросы с использованием RAG (Retrieval-Augmented Generation)."""

    def __init__(self):
        self.rag_service = None
        self.llm = None
        self._initialized = False

    def initialize(self, load_docs: bool = True, docs_directory: str = "./data/docs"):
        """Инициализация RAG-агента.

        Args:
            load_docs: Загружать ли документы при инициализации
            docs_directory: Директория с документами
        """
        try:
            logger.info("Initializing RAG Agent...")

            if not rag_service._initialized:
                rag_service.initialize()
            self.rag_service = rag_service

            if load_docs:
                logger.info(f"Loading documents from {docs_directory}...")
                self.rag_service.load_documents(docs_directory)

            self.llm = get_llm(temperature=0.3)

            self._initialized = True
            logger.success("RAG Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            raise

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        min_relevance_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Ответ на вопрос на основе найденных документов.

        Args:
            question: Вопрос пользователя
            top_k: Количество документов для поиска
            min_relevance_score: Минимальный порог релевантности (0-1)

        Returns:
            Словарь с ответом и метаданными
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"RAG Agent processing question: {question}")

            # Поиск релевантных документов
            retrieved_docs = self.rag_service.search(query=question, top_k=top_k)

            if not retrieved_docs:
                return {
                    "success": False,
                    "answer": "Извините, я не нашел релевантной информации в документации для ответа на ваш вопрос.",
                    "sources": [],
                    "retrieved_chunks": 0
                }

            # Фильтрация по релевантности
            # ChromaDB возвращает distance: для косинусной метрики distance = 1 - cosine_similarity
            # Преобразуем distance в similarity [0, 1], где 1 - идеальное совпадение
            filtered_docs = []

            logger.info(f"Retrieved {len(retrieved_docs)} documents before filtering")

            for doc in retrieved_docs:
                distance = doc['distance']

                # Преобразование distance в similarity с ограничением диапазона [0, 1]
                similarity = max(0.0, min(1.0, 1.0 - distance))

                doc['similarity'] = similarity

                logger.debug(f"Document: {doc['metadata'].get('filename', 'unknown')}, "
                            f"distance={distance:.4f}, similarity={similarity:.4f}")

                if similarity >= min_relevance_score:
                    filtered_docs.append(doc)

            logger.info(f"Filtered to {len(filtered_docs)}/{len(retrieved_docs)} documents "
                       f"with min_relevance={min_relevance_score}")

            if not filtered_docs:
                # Резервный вариант: используем лучший документ даже если ниже порога
                if retrieved_docs:
                    logger.warning(f"No documents passed threshold {min_relevance_score:.2f}, "
                                  f"using top document as fallback (similarity={retrieved_docs[0].get('similarity', 0):.2f})")
                    filtered_docs = [retrieved_docs[0]]
                else:
                    return {
                        "success": False,
                        "answer": "Извините, я не нашел релевантной информации в документации для ответа на ваш вопрос.",
                        "sources": [],
                        "retrieved_chunks": 0
                    }

            # Формирование контекста из документов
            context = self._format_context(filtered_docs)

            # Генерация ответа через LLM
            answer = self._generate_answer(question, context)

            # Извлечение источников
            sources = self._extract_sources(filtered_docs)

            logger.success(f"Generated answer using {len(filtered_docs)} relevant documents")

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(retrieved_docs),
                "relevant_chunks": len(filtered_docs),
                "top_similarity": filtered_docs[0]['similarity'] if filtered_docs else 0.0
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "success": False,
                "answer": f"Произошла ошибка при обработке вопроса: {str(e)}",
                "sources": [],
                "error": str(e)
            }

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Форматирование найденных документов в контекст для LLM.

        Args:
            documents: Список найденных документов

        Returns:
            Отформатированная строка контекста
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('filename', 'Unknown')
            content = doc['content']
            similarity = doc.get('similarity', 0.0)

            context_parts.append(
                f"[Документ {i}] (Источник: {source}, Релевантность: {similarity:.2f})\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Генерация ответа с использованием LLM и контекста.

        Args:
            question: Вопрос пользователя
            context: Извлеченный контекст

        Returns:
            Сгенерированный ответ
        """
        prompt = f"""Ты - эксперт по продуктам Positive Technologies. Ответь на вопрос пользователя на основе предоставленной документации.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации недостаточно для полного ответа, скажи об этом
3. Не придумывай информацию, которой нет в контексте
4. Отвечай на русском языке, четко и структурированно
5. Если в контексте есть технические детали, включи их в ответ
6. Укажи конкретные источники, если упоминаешь продукты или функции

КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ:
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
            raise

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Извлечение уникальных источников из документов.

        Args:
            documents: Список документов

        Returns:
            Список уникальных источников
        """
        sources = set()

        for doc in documents:
            filename = doc['metadata'].get('filename', 'Unknown')
            source = doc['metadata'].get('source', filename)
            sources.add(filename)

        return sorted(list(sources))

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получение информации о коллекции документов.

        Returns:
            Статистика коллекции
        """
        if not self._initialized:
            self.initialize(load_docs=False)

        return self.rag_service.get_collection_stats()

    def reload_documents(self, directory: str = "./data/docs"):
        """
        Перезагрузка документов из директории.

        Args:
            directory: Директория с документами
        """
        if not self._initialized:
            self.initialize(load_docs=False)

        logger.info(f"Reloading documents from {directory}...")
        self.rag_service.load_documents(directory, force_reload=True)
        logger.success("Documents reloaded successfully")


# Глобальный экземпляр RAG-агента
rag_agent = RAGAgent()
