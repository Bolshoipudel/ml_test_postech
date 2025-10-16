from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""

    # Приложение
    app_name: str = "LLM Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # API ключи
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Эмбеддинги
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Векторное хранилище
    vector_store: str = "chroma"
    chroma_persist_directory: str = "./chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # База данных
    database_type: str = "sqlite"  # "sqlite" или "postgresql"
    sqlite_database_path: str = "./data/sql/team_mock.db"
    postgres_user: str = "llm_assistant"
    postgres_password: str = "secure_password"
    postgres_db: str = "llm_assistant_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: Optional[str] = None

    # Веб-поиск
    web_search_enabled: bool = True
    max_search_results: int = 5

    # Агенты
    agent_max_iterations: int = 5
    agent_timeout: int = 30

    # Безопасность
    enable_guardrails: bool = True
    allowed_sql_operations: str = "SELECT"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url

        if self.database_type.lower() == "sqlite":
            return f"sqlite:///{self.sqlite_database_path}"
        else:
            return (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
    
    @property
    def allowed_sql_ops_list(self) -> List[str]:
        return [op.strip().upper() for op in self.allowed_sql_operations.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Глобальный экземпляр
settings = get_settings()