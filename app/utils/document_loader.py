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

        while start < len(text):
            end = start + self.chunk_size

            # Попытка разбить в удобном месте
            if end < len(text):
                # Попытка разбить на параграфе
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start:
                    end = paragraph_break
                else:
                    # Попытка разбить на новой строке
                    newline_break = text.rfind('\n', start, end)
                    if newline_break != -1 and newline_break > start:
                        end = newline_break
                    else:
                        # Попытка разбить на предложении
                        sentence_break = max(
                            text.rfind('. ', start, end),
                            text.rfind('! ', start, end),
                            text.rfind('? ', start, end)
                        )
                        if sentence_break != -1 and sentence_break > start:
                            end = sentence_break + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Смещение начала с учетом перекрытия
            start = end - self.chunk_overlap if end < len(text) else end

        logger.debug(f"Split text into {len(chunks)} chunks")

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
