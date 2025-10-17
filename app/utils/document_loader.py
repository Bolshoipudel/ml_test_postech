"""Утилиты загрузчика документов для RAG-системы."""
import os
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger


class Document:
    """Класс документа для хранения текста и метаданных."""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        """
        Args:
            page_content: Текстовое содержимое
            metadata: Опциональный словарь метаданных
        """
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class DocumentLoader:
    """Загрузчик для различных типов документов."""

    def __init__(self):
        self.supported_extensions = ['.md', '.txt', '.text']

    def load_file(self, file_path: str) -> Document:
        """
        Загрузка одного документа из файла.

        Args:
            file_path: Путь к файлу

        Returns:
            Объект Document
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = {
                'source': str(path),
                'filename': path.name,
                'extension': path.suffix,
                'size': path.stat().st_size
            }

            logger.debug(f"Loaded document: {path.name} ({len(content)} chars)")

            return Document(page_content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Загрузка всех документов из директории.

        Args:
            directory_path: Путь к директории
            recursive: Искать ли в поддиректориях

        Returns:
            Список объектов Document
        """
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        documents = []

        # Получение всех файлов с поддерживаемыми расширениями
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_file(str(file_path))
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents from {directory_path}")

        return documents


class TextSplitter:
    """Разбиение документов на чанки для улучшения поиска."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Количество символов перекрытия между чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Разбиение текста на чанки.

        Args:
            text: Текст для разбиения

        Returns:
            Список текстовых чанков
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        logger.info(f"Splitting text of {text_len} chars into chunks of {self.chunk_size} with overlap {self.chunk_overlap}")

        while start < text_len:
            # Определяем конец текущего chunk
            end = min(start + self.chunk_size, text_len)

            # Попытка найти хорошую точку разрыва ТОЛЬКО если мы не в конце текста
            if end < text_len:
                # Ищем оптимальное место для разрыва в последних 20% chunk
                search_start = end - int(self.chunk_size * 0.2)

                # Приоритет 1: Параграф
                paragraph_idx = text.rfind('\n\n', search_start, end)
                if paragraph_idx > start:
                    end = paragraph_idx + 2  # включаем \n\n

                #Приоритет 2: Новая строка
                elif (newline_idx := text.rfind('\n', search_start, end)) > start:
                    end = newline_idx + 1

                # Приоритет 3: Точка с пробелом
                elif (period_idx := text.rfind('. ', search_start, end)) > start:
                    end = period_idx + 2

            # Извлечение chunk и добавление в список
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Прогресс каждые 20 chunks
            if len(chunks) % 20 == 0:
                progress_pct = (start / text_len) * 100
                logger.info(f"Chunking: {len(chunks)} chunks created, {progress_pct:.1f}% complete")

            # Следующий start: двигаемся вперед на (chunk_size - overlap)
            # Это ГАРАНТИРУЕТ что мы всегда движемся вперёд
            step = max(self.chunk_size - self.chunk_overlap, 100)  # минимум 100 символов
            start += step

        logger.success(f"Split text into {len(chunks)} chunks")

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Разбиение документов на чанки.

        Args:
            documents: Список документов для разбиения

        Returns:
            Список разбитых документов
        """
        chunked_documents = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk'] = i
                metadata['total_chunks'] = len(chunks)

                chunked_documents.append(
                    Document(page_content=chunk, metadata=metadata)
                )

        logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")

        return chunked_documents


def load_documents_from_directory(directory: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Загрузка и разбиение документов из директории.

    Args:
        directory: Путь к директории
        chunk_size: Размер чанков
        chunk_overlap: Перекрытие между чанками

    Returns:
        Список разбитых документов
    """
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Загрузка документов
    documents = loader.load_directory(directory, recursive=True)

    if not documents:
        logger.warning(f"No documents found in {directory}")
        return []

    # Разбиение на чанки
    chunked_docs = splitter.split_documents(documents)

    return chunked_docs
