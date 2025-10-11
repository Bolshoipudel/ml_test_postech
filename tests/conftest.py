"""Pytest configuration and fixtures."""
import pytest


@pytest.fixture
def sample_chat_message():
    """Sample chat message for testing."""
    return {
        "message": "Сколько человек работает над PT Application Inspector?",
        "session_id": "test_session",
        "use_history": True
    }


@pytest.fixture
def sample_feedback():
    """Sample feedback for testing."""
    return {
        "session_id": "test_session",
        "rating": 4,
        "comment": "Good response, but could be more detailed"
    }
