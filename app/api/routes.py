"""API роуты для LLM Assistant."""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, List
import uuid
from datetime import datetime
from loguru import logger

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    FeedbackRequest,
    FeedbackResponse,
    ChatHistory,
    ErrorResponse,
    Message,
    MessageRole,
    ToolUsage,
    ToolType,
    EvaluateRequest,
    EvaluateResponse,
    RoutingInfo,
    MetricScore
)
from app.config import settings
from app.agents.sql_agent import sql_agent
from app.agents.rag_agent import rag_agent
from app.agents.web_search_agent import web_search_agent
from app.services.orchestrator_service import orchestrator

router = APIRouter(prefix="/api/v1", tags=["api"])

# Временное хранилище истории чата в памяти
# TODO: Заменить на постоянное хранилище (Redis или БД)
chat_history: Dict[str, List[Message]] = {}
feedback_storage: List[Dict] = []


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "api": "running",
            "llm_provider": settings.llm_provider,
            "vector_store": settings.vector_store
        }
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Основной чат-эндпоинт.

    Принимает сообщения пользователя и возвращает AI-ответы.
    Агент автоматически выбирает подходящие инструменты (RAG, SQL, Веб-поиск).
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in chat_history:
            chat_history[session_id] = []

        user_message = Message(
            role=MessageRole.USER,
            content=request.message,
            timestamp=datetime.now()
        )
        chat_history[session_id].append(user_message)

        # Обработка запроса через Orchestrator (с Router Agent)
        result = await orchestrator.process_query(
            query=request.message,
            use_history=request.use_history,
            conversation_history=[
                {"role": msg.role.value, "content": msg.content}
                for msg in chat_history[session_id]
            ] if request.use_history else None
        )

        response_text = result.get("answer", "Не удалось обработать запрос.")
        sources = result.get("sources", [])

        # Конвертация tools_used в модели ToolUsage
        tools_used = []
        for tool_usage in result.get("tools_used", []):
            # Нормализация tool_type к lowercase для соответствия ToolType enum
            raw_tool_type = tool_usage.get("tool_type", "none")
            if isinstance(raw_tool_type, str):
                raw_tool_type = raw_tool_type.lower()

            tools_used.append(ToolUsage(
                tool_type=ToolType(raw_tool_type),
                query=tool_usage.get("query"),
                result_summary=tool_usage.get("result_summary"),
                metadata=tool_usage.get("metadata")
            ))

        # Добавление решения роутера в метаданные
        routing_decision = result.get("routing_decision", {})
        if routing_decision:
            tools_used.insert(0, ToolUsage(
                tool_type=ToolType.ROUTER,
                query=request.message,
                result_summary=f"Routing: {routing_decision.get('tool', 'UNKNOWN')}",
                reasoning=routing_decision.get("reasoning"),
                confidence=routing_decision.get("confidence"),
                metadata={
                    "tool": routing_decision.get("tool"),
                    "query_type": routing_decision.get("query_type"),
                    "tools": routing_decision.get("tools", [])
                }
            ))

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            timestamp=datetime.now()
        )
        chat_history[session_id].append(assistant_message)

        logger.info(f"Chat request processed for session {session_id}")

        return ChatResponse(
            message=response_text,
            session_id=session_id,
            tools_used=tools_used,
            sources=sources if sources else None,
            metadata={
                "model": settings.llm_model,
                "provider": settings.llm_provider
            }
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    try:
        feedback_entry = {
            "session_id": request.session_id,
            "message_id": request.message_id,
            "rating": request.rating,
            "comment": request.comment,
            "timestamp": datetime.now()
        }
        feedback_storage.append(feedback_entry)

        logger.info(f"Feedback received for session {request.session_id}: rating {request.rating}")

        return FeedbackResponse(
            success=True,
            message="Feedback received successfully"
        )

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return FeedbackResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=ChatHistory)
async def get_history(session_id: str):
    """Получение истории чата для сессии."""
    if session_id not in chat_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    messages = chat_history[session_id]

    return ChatHistory(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages)
    )


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Очистка истории чата для сессии."""
    if session_id in chat_history:
        del chat_history[session_id]
        logger.info(f"History cleared for session {session_id}")
        return {"message": f"History cleared for session {session_id}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )


@router.get("/stats")
async def get_stats():
    """Статистика системы."""
    total_sessions = len(chat_history)
    total_messages = sum(len(messages) for messages in chat_history.values())
    total_feedback = len(feedback_storage)

    avg_rating = 0
    if feedback_storage:
        avg_rating = sum(f["rating"] for f in feedback_storage) / len(feedback_storage)

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_feedback": total_feedback,
        "average_rating": round(avg_rating, 2),
        "active_llm_provider": settings.llm_provider,
        "active_vector_store": settings.vector_store
    }


@router.get("/rag/stats")
async def get_rag_stats():
    """Статистика RAG коллекции и тестовый поиск."""
    try:
        if not rag_agent._initialized:
            rag_agent.initialize(load_docs=False)

        stats = rag_agent.get_collection_info()

        # Test search
        test_query = "PT Sandbox"
        test_results = rag_agent.rag_service.search(test_query, top_k=3)

        return {
            "collection_stats": stats,
            "test_search": {
                "query": test_query,
                "results_count": len(test_results),
                "top_results": [
                    {
                        "filename": r['metadata'].get('filename', 'unknown'),
                        "distance": r['distance'],
                        "similarity": max(0.0, min(1.0, 1.0 - r['distance'])),
                        "content_preview": r['content'][:100] + "..."
                    }
                    for r in test_results[:3]
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/reload")
async def reload_rag_documents():
    """Принудительная перезагрузка документов с диска."""
    try:
        logger.info("Force reloading RAG documents...")
        rag_agent.reload_documents("./data/docs")
        stats = rag_agent.get_collection_info()
        return {
            "success": True,
            "message": "Documents reloaded successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/debug/{query}")
async def debug_rag_search(query: str):
    """Отладка RAG поиска для просмотра расстояний и сходства."""
    try:
        if not rag_agent._initialized:
            rag_agent.initialize(load_docs=False)

        results = rag_agent.rag_service.search(query, top_k=5)

        debug_info = []
        for r in results:
            distance = r['distance']
            similarity = max(0.0, min(1.0, 1.0 - distance))

            debug_info.append({
                "filename": r['metadata'].get('filename', 'unknown'),
                "distance": round(distance, 4),
                "similarity": round(similarity, 4),
                "passes_threshold_0.3": similarity >= 0.3,
                "passes_threshold_0.5": similarity >= 0.5,
                "content_preview": r['content'][:150] + "..."
            })

        return {
            "query": query,
            "total_results": len(results),
            "results": debug_info
        }
    except Exception as e:
        logger.error(f"Error in debug search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EVALUATION ENDPOINT
# =============================================================================

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_single_query(request: EvaluateRequest):
    """
    Оценка качества одиночного запроса через DeepEval метрики.

    Этот endpoint позволяет оценить качество ответа системы на конкретный запрос
    с использованием метрик DeepEval:
    - Answer Relevancy: релевантность ответа вопросу
    - Faithfulness: соответствие источникам (отсутствие галлюцинаций)

    Args:
        request: EvaluateRequest с query, expected_output, retrieval_context

    Returns:
        EvaluateResponse с scores по метрикам

    Example:
        POST /api/v1/evaluate
        {
            "query": "Сколько программистов в команде?",
            "expected_output": "В команде 2 программиста",
            "retrieval_context": ["SQLite database: employees table"]
        }
    """
    try:
        # Импорт DeepEval метрик
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        # Вызов основного chat endpoint
        chat_request = ChatRequest(
            message=request.query,
            use_history=False
        )
        response = await chat(chat_request)

        # Извлечение routing decision
        tools_used = response.tools_used
        router_tool = "unknown"
        router_confidence = 0.0
        router_reasoning = ""

        if tools_used and tools_used[0].tool_type == ToolType.ROUTER:
            router_decision = tools_used[0]
            metadata = router_decision.metadata or {}
            router_tool = metadata.get("tool", "unknown").lower()
            router_confidence = router_decision.confidence or 0.0
            router_reasoning = router_decision.reasoning or ""

        # Создание routing info
        routing_info = RoutingInfo(
            tool=router_tool,
            confidence=router_confidence,
            reasoning=router_reasoning
        )

        # Создание DeepEval test case
        test_case = LLMTestCase(
            input=request.query,
            actual_output=response.message,
            expected_output=request.expected_output,
            retrieval_context=request.retrieval_context or [],
            context=response.sources or []
        )

        # Применение метрик
        metrics_to_use = [
            AnswerRelevancyMetric(threshold=0.7, model="gpt-4", include_reason=True),
            FaithfulnessMetric(threshold=0.7, model="gpt-4", include_reason=True)
        ]

        # Измерение метрик
        metric_scores = {}

        for metric in metrics_to_use:
            try:
                metric.measure(test_case)

                metric_scores[metric.__name__] = MetricScore(
                    score=metric.score,
                    threshold=metric.threshold,
                    passed=metric.is_successful(),
                    reason=getattr(metric, "reason", None)
                )

            except Exception as e:
                logger.error(f"Error measuring metric {metric.__name__}: {e}")
                metric_scores[metric.__name__] = MetricScore(
                    score=0.0,
                    threshold=metric.threshold,
                    passed=False,
                    reason=f"Error: {str(e)}"
                )

        # Формирование ответа
        return EvaluateResponse(
            query=request.query,
            response=response.message,
            routing=routing_info,
            metrics=metric_scores
        )

    except HTTPException:
        # Re-raise HTTP exceptions from chat endpoint
        raise

    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )
