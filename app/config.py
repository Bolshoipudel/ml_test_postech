from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "LLM Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # API Keys
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    # LLM Configuration
    llm_provider: str = "openai"  # openai, ollama
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # Embedding Configuration
    embedding_provider: str = "openai"  # openai, sentence-transformers
    embedding_model: str = "text-embedding-ada-002"
    
    # Vector Store Configuration
    vector_store: str = "chroma"  # chroma, faiss
    chroma_persist_directory: str = "./chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Database Configuration
    postgres_user: str = "llm_assistant"
    postgres_password: str = "secure_password"
    postgres_db: str = "llm_assistant_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: Optional[str] = None
    
    # Web Search Configuration
    web_search_enabled: bool = True
    max_search_results: int = 5
    
    # Agent Configuration
    agent_max_iterations: int = 5
    agent_timeout: int = 30
    
    # Guardrails Configuration
    enable_guardrails: bool = True
    allowed_sql_operations: str = "SELECT"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def db_url(self) -> str:
        """Construct database URL if not provided."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def allowed_sql_ops_list(self) -> List[str]:
        """Convert allowed SQL operations string to list."""
        return [op.strip().upper() for op in self.allowed_sql_operations.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()