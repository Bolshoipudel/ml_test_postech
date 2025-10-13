"""Фабрика LLM для создания экземпляров моделей."""
from typing import Optional, Union
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from loguru import logger

from app.config import settings


class LLMFactory:
    """Фабрика для создания экземпляров LLM."""

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Union[ChatOpenAI, Ollama]:
        """
        Создание экземпляра LLM на основе провайдера.

        Args:
            provider: Провайдер LLM ('openai' или 'ollama')
            model: Название модели
            temperature: Температура для генерации
            **kwargs: Дополнительные параметры

        Returns:
            Экземпляр LLM
        """
        provider = provider or settings.llm_provider
        temperature = temperature or settings.llm_temperature

        logger.info(f"Creating LLM with provider: {provider}")

        if provider == "openai":
            return LLMFactory._create_openai_llm(model, temperature, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama_llm(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _create_openai_llm(
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ChatOpenAI:
        model = model or settings.llm_model

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        logger.info(f"Creating OpenAI LLM with model: {model}")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens,
            **kwargs
        )

    @staticmethod
    def _create_ollama_llm(
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Ollama:
        model = model or settings.ollama_model

        logger.info(f"Creating Ollama LLM with model: {model}")

        return Ollama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            **kwargs
        )


def get_llm(**kwargs) -> Union[ChatOpenAI, Ollama]:
    """Получение экземпляра LLM по умолчанию."""
    return LLMFactory.create_llm(**kwargs)
