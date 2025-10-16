"""Pydantic модели для API запросов и ответов."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ToolType(str, Enum):
    RAG = "rag"
    SQL = "sql"
    WEB_SEARCH = "web_search"
    ROUTER = "router"
    MULTIPLE = "multiple"
    NONE = "none"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ToolUsage(BaseModel):
    """Информация об использовании инструмента."""
    tool_type: ToolType
    query: Optional[str] = None
    result_summary: Optional[str] = None
    reasoning: Optional[str] = None  # Обоснование решения роутера
    confidence: Optional[float] = None  # Уверенность роутера
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Модель запроса для чат-эндпоинта."""
    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    use_history: bool = Field(True, description="Whether to use conversation history")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Сколько человек работает над PT Application Inspector?",
                "session_id": "user123",
                "use_history": True
            }
        }


class ChatResponse(BaseModel):
    """Модель ответа для чат-эндпоинта."""
    message: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID")
    tools_used: List[ToolUsage] = Field(default_factory=list, description="Tools used to generate response")
    sources: Optional[List[str]] = Field(None, description="Sources used for the answer")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Над продуктом PT Application Inspector работает 3 разработчика...",
                "session_id": "user123",
                "tools_used": [
                    {
                        "tool_type": "sql",
                        "query": "SELECT COUNT(*) FROM team_members WHERE ...",
                        "result_summary": "Found 3 team members"
                    }
                ],
                "sources": ["database: team_members table"]
            }
        }


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Optional[Dict[str, str]] = None


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class ChatHistory(BaseModel):
    session_id: str
    messages: List[Message]
    total_messages: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Evaluation Schemas
# =============================================================================

class MetricScore(BaseModel):
    """Результат одной метрики DeepEval."""
    score: float = Field(..., description="Metric score (0.0 to 1.0)")
    threshold: float = Field(..., description="Threshold for pass/fail")
    passed: bool = Field(..., description="Whether metric passed threshold")
    reason: Optional[str] = Field(None, description="Detailed reasoning from metric")


class RoutingInfo(BaseModel):
    """Информация о роутинге запроса."""
    tool: str = Field(..., description="Tool selected by router")
    confidence: float = Field(..., description="Router confidence score")
    reasoning: str = Field(..., description="Router reasoning")


class EvaluateRequest(BaseModel):
    """Модель запроса для evaluate эндпоинта."""
    query: str = Field(..., description="User query to evaluate", min_length=1)
    expected_output: str = Field(..., description="Expected ground truth answer", min_length=1)
    retrieval_context: Optional[List[str]] = Field(
        None,
        description="Context for RAG evaluation (optional)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Сколько программистов в команде?",
                "expected_output": "В команде 2 программиста",
                "retrieval_context": ["SQLite database: employees table"]
            }
        }


class EvaluateResponse(BaseModel):
    """Модель ответа для evaluate эндпоинта."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="System response")
    routing: RoutingInfo = Field(..., description="Routing decision")
    metrics: Dict[str, MetricScore] = Field(..., description="DeepEval metric scores")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Сколько программистов в команде?",
                "response": "В команде работает 2 программиста.",
                "routing": {
                    "tool": "sql",
                    "confidence": 0.95,
                    "reasoning": "Query requires database lookup"
                },
                "metrics": {
                    "Answer Relevancy": {
                        "score": 0.92,
                        "threshold": 0.7,
                        "passed": True,
                        "reason": "Response directly answers the question"
                    },
                    "Faithfulness": {
                        "score": 0.88,
                        "threshold": 0.7,
                        "passed": True,
                        "reason": "No hallucinations detected"
                    }
                }
            }
        }
