"""Basic API tests."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_chat_endpoint():
    """Test chat endpoint with basic message."""
    payload = {
        "message": "Привет!",
        "use_history": False
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "session_id" in data
    assert "tools_used" in data


def test_chat_endpoint_validation():
    """Test chat endpoint validation."""
    # Empty message should fail
    payload = {
        "message": "",
        "use_history": False
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


def test_feedback_endpoint():
    """Test feedback endpoint."""
    payload = {
        "session_id": "test123",
        "rating": 5,
        "comment": "Great response!"
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_stats_endpoint():
    """Test stats endpoint."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_sessions" in data
    assert "total_messages" in data
