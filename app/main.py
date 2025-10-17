"""Точка входа для приложения LLM Assistant."""
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.config import settings
from app.api.routes import router
from app.models.schemas import ErrorResponse
from app.services.database_service import db_service
from app.agents.sql_agent import sql_agent
from app.agents.rag_agent import rag_agent
from app.agents.web_search_agent import web_search_agent


logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'DEBUG' if settings.debug else 'PRODUCTION'}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Vector Store: {settings.vector_store}")

    try:
        logger.info("Initializing database connection...")
        db_service.initialize()

        logger.info("Initializing SQL Agent...")
        sql_agent.initialize()

        logger.info("Initializing RAG Agent...")
        # Временно отключаем загрузку документов при старте для отладки
        rag_agent.initialize(load_docs=False, docs_directory="./data/docs")

        logger.info("Initializing Web Search Agent...")
        web_search_agent.initialize()

        # TODO: Инициализировать Router agent

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.warning("Application started with limited functionality")

    logger.success("Application started successfully")

    yield

    logger.info("Shutting down application...")

    try:
        logger.info("Closing database connections...")
        db_service.close()

        # TODO: Сохранить состояние vector store, очистить временные файлы

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.success("Application shut down successfully")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-functional LLM Assistant with RAG, SQL, and Web Search capabilities",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Настроить для production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors())
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if settings.debug else "An error occurred"
        ).model_dump()
    )


app.include_router(router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def main():
    import uvicorn

    logger.info(f"Starting uvicorn server on {settings.api_host}:{settings.api_port}")

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
