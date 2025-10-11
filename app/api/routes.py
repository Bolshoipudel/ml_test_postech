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
        if any(keyword in message_lower for keyword in ['сколько', 'кто работает', 'команда', 'разработчик', 'инцидент', 'баг']):
            # SQL-related query
            tools_used.append(ToolUsage(
                tool_type=ToolType.SQL,
                query="Placeholder SQL query",
                result_summary="Database query executed"
            ))
            response_text = "Это запрос к базе данных. (Агент SQL пока не реализован)"
            sources = ["database"]

        elif any(keyword in message_lower for keyword in ['как работает', 'что такое', 'документация', 'инструкция']):
            # RAG-related query
            tools_used.append(ToolUsage(
                tool_type=ToolType.RAG,
                query=request.message,
                result_summary="Documentation search completed"
            ))
            response_text = "Это вопрос о документации. (Агент RAG пока не реализован)"
            sources = ["documentation"]

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
