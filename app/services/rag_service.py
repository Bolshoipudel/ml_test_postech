"""Сервис RAG (Retrieval-Augmented Generation) для поиска по документам."""
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.config import settings
from app.utils.document_loader import Document, load_documents_from_directory


class RAGService:
    """Сервис для поиска документов через векторное сходство."""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialized = False

    def initialize(self):
        try:
            logger.info("Initializing RAG Service...")

            # Создание директории для хранения
            persist_directory = settings.chroma_persist_directory
            Path(persist_directory).mkdir(parents=True, exist_ok=True)

            # Инициализация ChromaDB клиента
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Инициализация embedding функции
            if settings.embedding_provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError("OpenAI API key is required for embeddings")

                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.openai_api_key,
                    model_name=settings.embedding_model
                )
                logger.info(f"Using OpenAI embeddings: {settings.embedding_model}")
            else:
                # По умолчанию sentence-transformers
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info("Using SentenceTransformer embeddings: all-MiniLM-L6-v2")

            # Получение или создание коллекции
            try:
                self.collection = self.client.get_collection(
                    name="documents",
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection with {self.collection.count()} documents")
            except Exception:
                self.collection = self.client.create_collection(
                    name="documents",
                    embedding_function=self.embedding_function,
                    metadata={"description": "PT Products documentation"}
                )
                logger.info("Created new collection")

            self._initialized = True
            logger.success("RAG Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise

    def load_documents(self, directory: str, force_reload: bool = False):
        """
        Загрузка документов из директории в векторное хранилище.

        Args:
            directory: Директория с документами
            force_reload: Если True, очистить существующие документы и перезагрузить
        """
        if not self._initialized:
            self.initialize()

        try:
            # Проверка уже загруженных документов
            if self.collection.count() > 0 and not force_reload:
                logger.info(f"Collection already contains {self.collection.count()} documents. Skipping load.")
                logger.info("Use force_reload=True to reload documents")
                return

            # Очистка коллекции при принудительной перезагрузке
            if force_reload and self.collection.count() > 0:
                logger.warning("Force reload: clearing existing documents")
                self.client.delete_collection("documents")
                self.collection = self.client.create_collection(
                    name="documents",
                    embedding_function=self.embedding_function,
                    metadata={"description": "PT Products documentation"}
                )

            # Загрузка и разбиение документов
            logger.info(f"Loading documents from {directory}...")
            documents = load_documents_from_directory(
                directory,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

            if not documents:
                logger.warning("No documents found to load")
                return

            # Подготовка данных для ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, doc in enumerate(documents):
                ids.append(f"doc_{i}")
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

            # Добавление документов батчами
            batch_size = 50  # Reduced from 100 to minimize API calls overhead
            total_batches = (len(documents) + batch_size - 1) // batch_size

            logger.info(f"Starting batch processing: {len(documents)} chunks in {total_batches} batches")

            import time
            start_time = time.time()

            for i in range(0, len(documents), batch_size):
                batch_start_time = time.time()
                batch_end = min(i + batch_size, len(documents))
                batch_num = i // batch_size + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} ({i}-{batch_end}/{len(documents)})...")

                self.collection.add(
                    ids=ids[i:batch_end],
                    documents=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

                batch_elapsed = time.time() - batch_start_time
                logger.info(f"✓ Batch {batch_num}/{total_batches} completed in {batch_elapsed:.2f}s")

            total_elapsed = time.time() - start_time
            logger.success(f"All batches processed in {total_elapsed:.2f}s (avg {total_elapsed/total_batches:.2f}s per batch)")

            logger.success(f"Successfully loaded {len(documents)} document chunks into vector store")

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов через векторное сходство.

        Args:
            query: Поисковый запрос
            top_k: Количество результатов для возврата
            filter_metadata: Опциональные фильтры по метаданным

        Returns:
            Список релевантных документов со scores
        """
        if not self._initialized:
            self.initialize()

        try:
            # Поиск по сходству
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata
            )

            # Форматирование результатов
            documents = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    documents.append(doc)

            logger.debug(f"Search query: '{query}' returned {len(documents)} results")

            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Добавление одного документа в векторное хранилище.

        Args:
            content: Содержимое документа
            metadata: Опциональные метаданные

        Returns:
            ID документа
        """
        if not self._initialized:
            self.initialize()

        try:
            # Генерация ID
            doc_id = f"doc_{self.collection.count()}"

            # Добавление документа
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata or {}]
            )

            logger.debug(f"Added document with ID: {doc_id}")

            return doc_id

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def delete_document(self, doc_id: str):
        """
        Удаление документа из векторного хранилища.

        Args:
            doc_id: ID документа для удаления
        """
        if not self._initialized:
            self.initialize()

        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции документов.

        Returns:
            Словарь со статистикой коллекции
        """
        if not self._initialized:
            self.initialize()

        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_function": str(type(self.embedding_function).__name__)
        }

    def clear_collection(self):
        if not self._initialized:
            self.initialize()

        try:
            self.client.delete_collection("documents")
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_function,
                metadata={"description": "PT Products documentation"}
            )
            logger.warning("Cleared all documents from collection")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise


# Глобальный экземпляр RAG-сервиса
rag_service = RAGService()
