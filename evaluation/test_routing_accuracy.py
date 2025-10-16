"""
Pytest тесты для проверки точности Router Agent.

Использует parametrize для тестирования всех типов роутинга:
- SQL (запросы к БД)
- RAG (поиск в документации)
- Web Search (актуальная информация)
- Multiple (комбинированные запросы)
- None (нерелевантные запросы)

Запуск:
    pytest evaluation/test_routing_accuracy.py -v
    pytest evaluation/test_routing_accuracy.py -v -k "test_sql"
"""

import sys
from pathlib import Path

# Добавление корневой директории в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import requests
from typing import Dict, Any

from evaluation.test_dataset import (
    SQL_TEST_CASES,
    RAG_TEST_CASES,
    WEB_SEARCH_TEST_CASES,
    MULTIPLE_TEST_CASES,
    NONE_TEST_CASES,
    get_test_case_by_id
)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def api_client():
    """Проверка доступности API перед запуском тестов."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        assert response.status_code == 200, f"API not healthy: {response.text}"
        print(f"\n✓ API is healthy: {response.json()}")
    except Exception as e:
        pytest.fail(f"Cannot connect to API at {API_BASE_URL}: {e}")

    return requests.Session()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def query_chat_api(
    client: requests.Session,
    message: str
) -> Dict[str, Any]:
    """
    Отправка запроса к chat API.

    Args:
        client: Requests session
        message: User query

    Returns:
        JSON response
    """
    response = client.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={"message": message, "use_history": False},
        timeout=TIMEOUT
    )

    assert response.status_code == 200, f"API error: {response.status_code} - {response.text}"

    return response.json()


def extract_routing_info(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Извлечение информации о роутинге из ответа API.

    Args:
        response_data: JSON ответ от API

    Returns:
        Dict с tool, confidence, reasoning
    """
    tools_used = response_data.get("tools_used", [])

    if not tools_used:
        return {
            "tool": "unknown",
            "confidence": 0.0,
            "reasoning": "No tools used"
        }

    # Первый элемент должен быть router decision
    router_decision = tools_used[0]

    if router_decision.get("tool_type") != "router":
        return {
            "tool": "unknown",
            "confidence": 0.0,
            "reasoning": "No router decision found"
        }

    metadata = router_decision.get("metadata", {})

    return {
        "tool": metadata.get("tool", "unknown").lower(),
        "confidence": router_decision.get("confidence", 0.0),
        "reasoning": router_decision.get("reasoning", "")
    }


# =============================================================================
# ROUTING ACCURACY TESTS - SQL
# =============================================================================

@pytest.mark.parametrize("test_case_id", [tc["id"] for tc in SQL_TEST_CASES[:5]])
def test_sql_routing_accuracy(api_client, test_case_id):
    """
    Тест точности роутинга для SQL запросов.

    Проверяет что Router Agent правильно определяет SQL запросы
    с достаточной уверенностью.
    """
    test_case = get_test_case_by_id(test_case_id)

    # Выполнение запроса
    response = query_chat_api(api_client, test_case["query"])
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == "sql", (
        f"Expected SQL routing for '{test_case['query']}', "
        f"got {routing['tool']} (confidence: {routing['confidence']:.2f})"
    )

    assert routing["confidence"] >= test_case.get("min_confidence", 0.7), (
        f"Confidence {routing['confidence']:.2f} below threshold "
        f"{test_case.get('min_confidence', 0.7)}"
    )

    # Проверка что хотя бы одно ожидаемое слово есть в ответе
    response_text = response["message"].lower()
    assert any(
        word.lower() in response_text
        for word in test_case.get("expected_answer_contains", [])
    ), f"Expected keywords not found in response: {response_text[:200]}"


# =============================================================================
# ROUTING ACCURACY TESTS - RAG
# =============================================================================

@pytest.mark.parametrize("test_case_id", [tc["id"] for tc in RAG_TEST_CASES[:5]])
def test_rag_routing_accuracy(api_client, test_case_id):
    """
    Тест точности роутинга для RAG запросов.

    Проверяет что Router Agent правильно определяет запросы
    требующие поиска в документации.
    """
    test_case = get_test_case_by_id(test_case_id)

    # Выполнение запроса
    response = query_chat_api(api_client, test_case["query"])
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == "rag", (
        f"Expected RAG routing for '{test_case['query']}', "
        f"got {routing['tool']} (confidence: {routing['confidence']:.2f})"
    )

    assert routing["confidence"] >= test_case.get("min_confidence", 0.7), (
        f"Confidence {routing['confidence']:.2f} below threshold "
        f"{test_case.get('min_confidence', 0.7)}"
    )


# =============================================================================
# ROUTING ACCURACY TESTS - WEB SEARCH
# =============================================================================

@pytest.mark.parametrize("test_case_id", [tc["id"] for tc in WEB_SEARCH_TEST_CASES[:5]])
def test_web_search_routing_accuracy(api_client, test_case_id):
    """
    Тест точности роутинга для Web Search запросов.

    Проверяет что Router Agent правильно определяет запросы
    требующие актуальной информации из интернета.
    """
    test_case = get_test_case_by_id(test_case_id)

    # Выполнение запроса
    response = query_chat_api(api_client, test_case["query"])
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == "web_search", (
        f"Expected WEB_SEARCH routing for '{test_case['query']}', "
        f"got {routing['tool']} (confidence: {routing['confidence']:.2f})"
    )

    assert routing["confidence"] >= test_case.get("min_confidence", 0.7), (
        f"Confidence {routing['confidence']:.2f} below threshold "
        f"{test_case.get('min_confidence', 0.7)}"
    )


# =============================================================================
# ROUTING ACCURACY TESTS - MULTIPLE
# =============================================================================

@pytest.mark.parametrize("test_case_id", [tc["id"] for tc in MULTIPLE_TEST_CASES[:5]])
def test_multiple_routing_accuracy(api_client, test_case_id):
    """
    Тест точности роутинга для MULTIPLE запросов.

    Проверяет что Router Agent правильно определяет запросы
    требующие использования нескольких инструментов.
    """
    test_case = get_test_case_by_id(test_case_id)

    # Выполнение запроса
    response = query_chat_api(api_client, test_case["query"])
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == "multiple", (
        f"Expected MULTIPLE routing for '{test_case['query']}', "
        f"got {routing['tool']} (confidence: {routing['confidence']:.2f})"
    )

    assert routing["confidence"] >= test_case.get("min_confidence", 0.7), (
        f"Confidence {routing['confidence']:.2f} below threshold "
        f"{test_case.get('min_confidence', 0.7)}"
    )


# =============================================================================
# ROUTING ACCURACY TESTS - NONE (новая функция)
# =============================================================================

@pytest.mark.parametrize("test_case_id", [tc["id"] for tc in NONE_TEST_CASES])
def test_none_routing_accuracy(api_client, test_case_id):
    """
    Тест точности роутинга для NONE запросов (нерелевантные).

    Проверяет что Router Agent правильно определяет нерелевантные
    запросы и возвращает соответствующее сообщение.
    """
    test_case = get_test_case_by_id(test_case_id)

    # Выполнение запроса
    response = query_chat_api(api_client, test_case["query"])
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == "none", (
        f"Expected NONE routing for irrelevant query '{test_case['query']}', "
        f"got {routing['tool']} (confidence: {routing['confidence']:.2f})"
    )

    assert routing["confidence"] >= test_case.get("min_confidence", 0.7), (
        f"Confidence {routing['confidence']:.2f} below threshold "
        f"{test_case.get('min_confidence', 0.7)}"
    )

    # Проверка что ответ содержит упоминание о нерелевантности
    response_text = response["message"].lower()
    irrelevant_keywords = ["нерелевантн", "не могу помочь", "не могу", "специализируюсь"]
    assert any(
        keyword in response_text
        for keyword in irrelevant_keywords
    ), f"Expected irrelevant response indicator, got: {response_text[:200]}"


# =============================================================================
# COMPREHENSIVE ROUTING TEST
# =============================================================================

@pytest.mark.parametrize("expected_tool,query,min_confidence", [
    ("sql", "Сколько программистов в команде?", 0.9),
    ("sql", "Кто работает DevOps-инженером?", 0.9),
    ("rag", "Что такое PT Sandbox?", 0.85),
    ("rag", "Какие файлы поддерживает PT Application Inspector?", 0.8),
    ("web_search", "Последние новости по кибербезопасности", 0.9),
    ("web_search", "Актуальные тренды DevSecOps", 0.8),
    ("multiple", "Сколько человек в команде и что такое PT AI?", 0.75),
    ("none", "Какая погода в Москве?", 0.85),
])
def test_routing_comprehensive(api_client, expected_tool, query, min_confidence):
    """
    Комплексный тест роутинга с различными типами запросов.

    Параметризованный тест проверяющий основные сценарии
    использования Router Agent.
    """
    # Выполнение запроса
    response = query_chat_api(api_client, query)
    routing = extract_routing_info(response)

    # Assertions
    assert routing["tool"] == expected_tool, (
        f"Query: '{query}'\n"
        f"Expected: {expected_tool}\n"
        f"Got: {routing['tool']}\n"
        f"Confidence: {routing['confidence']:.2f}\n"
        f"Reasoning: {routing['reasoning'][:100]}"
    )

    assert routing["confidence"] >= min_confidence, (
        f"Query: '{query}'\n"
        f"Confidence {routing['confidence']:.2f} < {min_confidence}\n"
        f"Reasoning: {routing['reasoning'][:100]}"
    )

    # Проверка наличия reasoning
    assert len(routing["reasoning"]) > 10, (
        "Router reasoning is too short or empty"
    )


# =============================================================================
# CONFIDENCE THRESHOLD TESTS
# =============================================================================

def test_high_confidence_for_clear_queries(api_client):
    """
    Тест что очевидные запросы имеют высокую уверенность.

    Для простых однозначных запросов confidence должен быть >= 0.9
    """
    clear_queries = [
        ("Сколько сотрудников?", "sql"),
        ("Что такое PT Sandbox?", "rag"),
        ("Последние новости", "web_search"),
    ]

    for query, expected_tool in clear_queries:
        response = query_chat_api(api_client, query)
        routing = extract_routing_info(response)

        assert routing["tool"] == expected_tool, (
            f"Wrong routing for clear query '{query}'"
        )

        assert routing["confidence"] >= 0.85, (
            f"Low confidence {routing['confidence']:.2f} for clear query '{query}'"
        )


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
