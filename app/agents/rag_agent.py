"""RAG Agent for answering questions based on documentation."""
from typing import Dict, Any, List, Optional
from loguru import logger

from app.services.rag_service import rag_service
from app.services.llm_factory import get_llm


class RAGAgent:
    """Agent for answering questions using RAG (Retrieval-Augmented Generation)."""

    def __init__(self):
        """Initialize RAG Agent."""
        self.rag_service = None
        self.llm = None
        self._initialized = False

    def initialize(self, load_docs: bool = True, docs_directory: str = "./data/docs"):
        """
        Initialize the agent.

        Args:
            load_docs: Whether to load documents on initialization
            docs_directory: Directory containing documents
        """
        try:
            logger.info("Initializing RAG Agent...")

            # Initialize RAG service
            if not rag_service._initialized:
                rag_service.initialize()
            self.rag_service = rag_service

            # Load documents if requested
            if load_docs:
                logger.info(f"Loading documents from {docs_directory}...")
                self.rag_service.load_documents(docs_directory)

            # Get LLM
            self.llm = get_llm(temperature=0.3)  # Lower temperature for more factual responses

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
        Answer a question based on retrieved documents.

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            min_relevance_score: Minimum relevance score (0-1)

        Returns:
            Dictionary with answer and metadata
        """
        if not self._initialized:
            self.initialize()

        try:
            logger.info(f"RAG Agent processing question: {question}")

            # Search for relevant documents
            retrieved_docs = self.rag_service.search(query=question, top_k=top_k)

            if not retrieved_docs:
                return {
                    "success": False,
                    "answer": "Извините, я не нашел релевантной информации в документации для ответа на ваш вопрос.",
                    "sources": [],
                    "retrieved_chunks": 0
                }

            # Filter by relevance score (distance metric from ChromaDB)
            # ChromaDB uses different distance metrics depending on configuration
            # For cosine: distance = 1 - cosine_similarity, range approximately [0, 2]
            # We convert distance to similarity score [0, 1] where 1 is perfect match
            filtered_docs = []

            logger.info(f"Retrieved {len(retrieved_docs)} documents before filtering")

            for doc in retrieved_docs:
                distance = doc['distance']

                # Convert distance to similarity score
                # For cosine distance: similarity = 1 - distance
                # Clamp to [0, 1] range to handle edge cases
                similarity = max(0.0, min(1.0, 1.0 - distance))

                doc['similarity'] = similarity

                logger.debug(f"Document: {doc['metadata'].get('filename', 'unknown')}, "
                            f"distance={distance:.4f}, similarity={similarity:.4f}")

                if similarity >= min_relevance_score:
                    filtered_docs.append(doc)

            logger.info(f"Filtered to {len(filtered_docs)}/{len(retrieved_docs)} documents "
                       f"with min_relevance={min_relevance_score}")

            if not filtered_docs:
                # Fallback: use top document even if below threshold
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

            # Format context from retrieved documents
            context = self._format_context(filtered_docs)

            # Generate answer using LLM
            answer = self._generate_answer(question, context)

            # Extract sources
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
        Format retrieved documents into context for LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
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
        Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Generated answer
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
        Extract unique sources from documents.

        Args:
            documents: List of documents

        Returns:
            List of unique source identifiers
        """
        sources = set()

        for doc in documents:
            filename = doc['metadata'].get('filename', 'Unknown')
            source = doc['metadata'].get('source', filename)
            sources.add(filename)

        return sorted(list(sources))

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the document collection.

        Returns:
            Collection statistics
        """
        if not self._initialized:
            self.initialize(load_docs=False)

        return self.rag_service.get_collection_stats()

    def reload_documents(self, directory: str = "./data/docs"):
        """
        Reload documents from directory.

        Args:
            directory: Directory containing documents
        """
        if not self._initialized:
            self.initialize(load_docs=False)

        logger.info(f"Reloading documents from {directory}...")
        self.rag_service.load_documents(directory, force_reload=True)
        logger.success("Documents reloaded successfully")


# Global RAG agent instance
rag_agent = RAGAgent()
