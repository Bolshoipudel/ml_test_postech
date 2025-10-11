"""Pydantic models for API requests and responses."""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ToolType(str, Enum):
    """Type of tool used by the agent."""
    RAG = "rag"
    SQL = "sql"
    WEB_SEARCH = "web_search"
    NONE = "none"


class Message(BaseModel):
    """Chat message."""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class ToolUsage(BaseModel):
    """Information about tool usage."""
    tool_type: ToolType
    query: Optional[str] = None
    result_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
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
    """Response model for chat endpoint."""
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
    """Health check response."""
    status: str = "healthy"
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Optional[Dict[str, str]] = None


class FeedbackRequest(BaseModel):
    """Feedback request model."""
    session_id: str
    message_id: Optional[str] = None
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Feedback response model."""
    success: bool
    message: str


class ChatHistory(BaseModel):
    """Chat history response."""
    session_id: str
    messages: List[Message]
    total_messages: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
