"""API routes for the LLM Assistant."""
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
    ToolType
)
from app.config import settings
from app.agents.sql_agent import sql_agent
from app.agents.rag_agent import rag_agent

router = APIRouter(prefix="/api/v1", tags=["api"])

# In-memory storage for chat history (temporary solution)
# TODO: Replace with persistent storage (Redis or Database)
chat_history: Dict[str, List[Message]] = {}
feedback_storage: List[Dict] = []


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
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
    Main chat endpoint.

    This endpoint receives user messages and returns AI-generated responses.
    The agent automatically selects appropriate tools (RAG, SQL, Web Search).
    """
    try:
        # Generate or use existing session_id
        session_id = request.session_id or str(uuid.uuid4())

        # Initialize chat history for new sessions
        if session_id not in chat_history:
            chat_history[session_id] = []

        # Add user message to history
        user_message = Message(
            role=MessageRole.USER,
            content=request.message,
            timestamp=datetime.now()
        )
        chat_history[session_id].append(user_message)

        # TODO: Implement actual agent logic here
        # For now, return a placeholder response

        # Placeholder: Simple keyword-based tool selection
        tools_used = []
        response_text = ""
        sources = []

        message_lower = request.message.lower()

        # Simple routing logic (will be replaced with actual agent)
        if any(keyword in message_lower for keyword in ['сколько', 'кто работает', 'команда', 'разработчик', 'инцидент', 'баг', 'продукт', 'отдел', 'фича', 'feature']):
            # SQL-related query - use SQL Agent
            try:
                logger.info(f"Routing to SQL Agent for question: {request.message}")

                result = sql_agent.execute_query(request.message, validate=True)

                if result["success"]:
                    # Format results into natural language
                    formatted_answer = sql_agent.format_results(
                        result["results"],
                        request.message
                    )

                    tools_used.append(ToolUsage(
                        tool_type=ToolType.SQL,
                        query=result["sql_query"],
                        result_summary=f"Найдено {result['row_count']} записей",
                        metadata={
                            "row_count": result["row_count"],
                            "sql_query": result["sql_query"]
                        }
                    ))
                    response_text = formatted_answer
                    sources = ["database: PostgreSQL"]
                else:
                    # SQL generation failed
                    logger.warning(f"SQL Agent failed: {result.get('error')}")
                    response_text = f"Не удалось выполнить запрос к базе данных: {result.get('error', 'Неизвестная ошибка')}"
                    tools_used.append(ToolUsage(
                        tool_type=ToolType.SQL,
                        query=result.get("sql_query"),
                        result_summary="Ошибка выполнения запроса",
                        metadata={"error": result.get("error")}
                    ))

            except Exception as e:
                logger.error(f"SQL Agent error: {e}")
                response_text = f"Произошла ошибка при обработке запроса: {str(e)}"
                tools_used.append(ToolUsage(
                    tool_type=ToolType.SQL,
                    result_summary="Ошибка SQL Agent",
                    metadata={"error": str(e)}
                ))
            sources = ["database"]

        elif any(keyword in message_lower for keyword in ['как работает', 'что такое', 'документация', 'инструкция', 'описание', 'возможности', 'функции', 'архитектура', 'интеграция', 'поддержка', 'лицензирование', 'требования']):
            # RAG-related query - use RAG Agent
            try:
                logger.info(f"Routing to RAG Agent for question: {request.message}")

                result = rag_agent.answer_question(request.message, top_k=5)

                if result["success"]:
                    tools_used.append(ToolUsage(
                        tool_type=ToolType.RAG,
                        query=request.message,
                        result_summary=f"Найдено {result['relevant_chunks']} релевантных документов",
                        metadata={
                            "retrieved_chunks": result["retrieved_chunks"],
                            "relevant_chunks": result["relevant_chunks"],
                            "top_similarity": result.get("top_similarity", 0.0)
                        }
                    ))
                    response_text = result["answer"]
                    sources = [f"documentation: {src}" for src in result["sources"]]
                else:
                    # RAG failed
                    logger.warning(f"RAG Agent failed: {result.get('answer')}")
                    response_text = result["answer"]
                    tools_used.append(ToolUsage(
                        tool_type=ToolType.RAG,
                        query=request.message,
                        result_summary="Релевантные документы не найдены",
                        metadata={"error": result.get("error", "No relevant documents")}
                    ))
                    sources = []

            except Exception as e:
                logger.error(f"RAG Agent error: {e}")
                response_text = f"Произошла ошибка при поиске в документации: {str(e)}"
                tools_used.append(ToolUsage(
                    tool_type=ToolType.RAG,
                    result_summary="Ошибка RAG Agent",
                    metadata={"error": str(e)}
                ))
                sources = []

        elif any(keyword in message_lower for keyword in ['новости', 'тренд', 'найди информацию', 'поищи']):
            # Web search query
            tools_used.append(ToolUsage(
                tool_type=ToolType.WEB_SEARCH,
                query=request.message,
                result_summary="Web search completed"
            ))
            response_text = "Это запрос для веб-поиска. (Агент Web Search пока не реализован)"
            sources = ["web"]

        else:
            # General query
            tools_used.append(ToolUsage(
                tool_type=ToolType.NONE,
                result_summary="Direct LLM response"
            ))
            response_text = f"Вы сказали: '{request.message}'. Агенты пока не реализованы, но инфраструктура готова!"

        # Add assistant message to history
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
    """Submit feedback for a chat response."""
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
    """Get chat history for a session."""
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
    """Clear chat history for a session."""
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
    """Get system statistics."""
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
    """Get RAG collection statistics and test search."""
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
    """Force reload documents from disk."""
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
    """Debug RAG search to see actual distances and similarities."""
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
