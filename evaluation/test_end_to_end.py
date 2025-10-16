"""
E2E тесты с интеграцией DeepEval метрик.

Полноценные end-to-end тесты системы с использованием:
- AnswerRelevancyMetric
- FaithfulnessMetric
- ContextualRelevancyMetric
- RouterAccuracyMetric (custom)

Запуск:
    pytest evaluation/test_end_to_end.py -v
    pytest evaluation/test_end_to_end.py -v -k "test_sql"
"""

import sys
from pathlib import Path

# Добавление корневой директории в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import requests
from typing import Dict, Any, List

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

from evaluation.test_dataset import (
    get_test_case_by_id,
    SQL_TEST_CASES,
    RAG_TEST_CASES,
    WEB_SEARCH_TEST_CASES,
    MULTIPLE_TEST_CASES
)
from evaluation.metrics_config import (
    RouterAccuracyMetric,
    create_test_case_with_routing,
    MetricsConfig
)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# Metrics configuration
metrics_config = MetricsConfig(
    model="gpt-4",
    threshold=0.7,
    include_reason=True
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def api_client():
    """Проверка доступности API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        assert response.status_code == 200
        print(f"\n✓ API is healthy")
    except Exception as e:
        pytest.fail(f"Cannot connect to API: {e}")

    return requests.Session()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def query_and_create_test_case(
    client: requests.Session,
    test_case_data: Dict[str, Any]
) -> LLMTestCase:
    """
    Выполнение API запроса и создание DeepEval test case.

    Args:
        client: Requests session
        test_case_data: Тестовый кейс из датасета

    Returns:
        LLMTestCase для evaluation
    """
    # Выполнение запроса
    response = client.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={"message": test_case_data["query"], "use_history": False},
        timeout=TIMEOUT
    )

    assert response.status_code == 200, f"API error: {response.status_code}"

    data = response.json()

    # Извлечение routing decision
    tools_used = data.get("tools_used", [])
    router_tool = "unknown"
    router_confidence = 0.0
    router_reasoning = ""

    if tools_used and tools_used[0].get("tool_type") == "router":
        router_decision = tools_used[0]
        metadata = router_decision.get("metadata", {})
        router_tool = metadata.get("tool", "unknown").lower()
        router_confidence = router_decision.get("confidence", 0.0)
        router_reasoning = router_decision.get("reasoning", "")

    # Извлечение контекста
    retrieval_context = test_case_data.get("context", [])
    sources = data.get("sources", [])

    # Создание test case
    test_case = create_test_case_with_routing(
        query=test_case_data["query"],
        actual_output=data["message"],
        expected_output=test_case_data["ground_truth"],
        expected_tool=test_case_data["expected_tool"],
        actual_tool=router_tool,
        confidence=router_confidence,
        reasoning=router_reasoning,
        retrieval_context=retrieval_context,
        context=sources or retrieval_context
    )

    return test_case


# =============================================================================
# SQL AGENT E2E TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case_id", ["sql_001", "sql_002", "sql_003"])
async def test_sql_agent_end_to_end(api_client, test_case_id):
    """
    E2E тест SQL Agent с DeepEval метриками.

    Проверяет:
    - Правильность роутинга (RouterAccuracyMetric)
    - Релевантность ответа (AnswerRelevancyMetric)
    - Отсутствие галлюцинаций (FaithfulnessMetric)
    """
    test_case_data = get_test_case_by_id(test_case_id)

    # Создание test case из API response
    test_case = query_and_create_test_case(api_client, test_case_data)

    # Метрики для SQL
    metrics = [
        RouterAccuracyMetric(threshold=0.7, confidence_threshold=0.8),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4"),
        FaithfulnessMetric(threshold=0.7, model="gpt-4")
    ]

    # DeepEval assertion
    assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# RAG AGENT E2E TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case_id", ["rag_001", "rag_002", "rag_003"])
async def test_rag_agent_end_to_end(api_client, test_case_id):
    """
    E2E тест RAG Agent с DeepEval метриками.

    Проверяет:
    - Правильность роутинга
    - Релевантность ответа
    - Релевантность контекста (ContextualRelevancyMetric)
    - Соответствие источникам (FaithfulnessMetric)
    """
    test_case_data = get_test_case_by_id(test_case_id)

    # Создание test case
    test_case = query_and_create_test_case(api_client, test_case_data)

    # Метрики для RAG (включая Contextual Relevancy)
    metrics = [
        RouterAccuracyMetric(threshold=0.7, confidence_threshold=0.75),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4"),
        FaithfulnessMetric(threshold=0.7, model="gpt-4"),
        ContextualRelevancyMetric(threshold=0.7, model="gpt-4")
    ]

    # DeepEval assertion
    assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# WEB SEARCH AGENT E2E TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case_id", ["web_001", "web_002", "web_003"])
async def test_web_search_agent_end_to_end(api_client, test_case_id):
    """
    E2E тест Web Search Agent с DeepEval метриками.

    Проверяет:
    - Правильность роутинга
    - Релевантность ответа
    """
    test_case_data = get_test_case_by_id(test_case_id)

    # Создание test case
    test_case = query_and_create_test_case(api_client, test_case_data)

    # Метрики для Web Search (без Faithfulness т.к. нет фиксированного контекста)
    metrics = [
        RouterAccuracyMetric(threshold=0.7, confidence_threshold=0.8),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4")
    ]

    # DeepEval assertion
    assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# MULTIPLE AGENTS E2E TESTS (ORCHESTRATOR)
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case_id", ["multi_001", "multi_002"])
async def test_multiple_agents_end_to_end(api_client, test_case_id):
    """
    E2E тест Orchestrator (MULTIPLE agents) с DeepEval метриками.

    Проверяет:
    - Правильность роутинга MULTIPLE
    - Релевантность комбинированного ответа
    - Контекстуальная релевантность
    """
    test_case_data = get_test_case_by_id(test_case_id)

    # Создание test case
    test_case = query_and_create_test_case(api_client, test_case_data)

    # Метрики для MULTIPLE (полный набор)
    metrics = [
        RouterAccuracyMetric(threshold=0.7, confidence_threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4"),
        FaithfulnessMetric(threshold=0.7, model="gpt-4"),
        ContextualRelevancyMetric(threshold=0.7, model="gpt-4")
    ]

    # DeepEval assertion
    assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# COMPREHENSIVE E2E TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_comprehensive_sql_workflow(api_client):
    """
    Комплексный тест SQL workflow.

    Тестирует несколько SQL запросов подряд для проверки
    консистентности работы системы.
    """
    sql_queries = [
        "sql_001",  # Количество QA-инженеров
        "sql_004",  # Информация о Bob Johnson
        "sql_007"   # Специализация на безопасности API
    ]

    for test_id in sql_queries:
        test_case_data = get_test_case_by_id(test_id)
        test_case = query_and_create_test_case(api_client, test_case_data)

        # Базовые метрики
        metrics = [
            RouterAccuracyMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7, model="gpt-4")
        ]

        assert_test(test_case=test_case, metrics=metrics)


@pytest.mark.asyncio
async def test_comprehensive_rag_workflow(api_client):
    """
    Комплексный тест RAG workflow.

    Тестирует несколько RAG запросов для проверки
    качества поиска в документации.
    """
    rag_queries = [
        "rag_001",  # PT Sandbox overview
        "rag_004",  # PDF support
        "rag_008"   # Link extraction
    ]

    for test_id in rag_queries:
        test_case_data = get_test_case_by_id(test_id)
        test_case = query_and_create_test_case(api_client, test_case_data)

        # Полные RAG метрики
        metrics = [
            RouterAccuracyMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7, model="gpt-4"),
            FaithfulnessMetric(threshold=0.7, model="gpt-4"),
            ContextualRelevancyMetric(threshold=0.7, model="gpt-4")
        ]

        assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# QUALITY THRESHOLD TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_high_quality_responses(api_client):
    """
    Тест что высококачественные запросы дают высокие scores.

    Проверяет что для простых однозначных запросов
    все метрики должны быть выше 0.8.
    """
    high_quality_cases = [
        "sql_001",  # Простой count
        "rag_001",  # Описание продукта
    ]

    for test_id in high_quality_cases:
        test_case_data = get_test_case_by_id(test_id)
        test_case = query_and_create_test_case(api_client, test_case_data)

        # Метрики с повышенным порогом
        metrics = [
            RouterAccuracyMetric(threshold=0.8, confidence_threshold=0.85),
            AnswerRelevancyMetric(threshold=0.8, model="gpt-4")
        ]

        assert_test(test_case=test_case, metrics=metrics)


# =============================================================================
# ROUTER DECISION QUALITY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_router_reasoning_quality(api_client):
    """
    Тест качества reasoning от Router Agent.

    Проверяет что Router дает подробное обоснование
    своего решения.
    """
    test_cases = ["sql_001", "rag_001", "web_001"]

    for test_id in test_cases:
        test_case_data = get_test_case_by_id(test_id)
        test_case = query_and_create_test_case(api_client, test_case_data)

        # Проверка наличия reasoning
        reasoning = test_case.additional_metadata.get("reasoning", "")
        assert len(reasoning) > 20, (
            f"Router reasoning too short for {test_id}: {reasoning}"
        )

        # Проверка confidence
        confidence = test_case.additional_metadata.get("confidence", 0)
        assert confidence > 0.7, (
            f"Low confidence {confidence} for {test_id}"
        )


# =============================================================================
# MIXED QUERY TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("query,expected_keywords", [
    ("Сколько программистов работает в команде?", ["программист", "2", "3"]),
    ("Что такое PT Sandbox?", ["PT Sandbox", "анализ", "файл"]),
    ("Последние новости по кибербезопасности", ["новост", "безопасность"]),
])
async def test_answer_contains_expected_keywords(api_client, query, expected_keywords):
    """
    Тест что ответы содержат ожидаемые ключевые слова.

    Проверяет релевантность ответов на семантическом уровне.
    """
    # Выполнение запроса
    response = api_client.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={"message": query, "use_history": False},
        timeout=TIMEOUT
    )

    assert response.status_code == 200
    data = response.json()

    response_text = data["message"].lower()

    # Проверка что хотя бы одно ключевое слово присутствует
    found_keywords = [
        kw for kw in expected_keywords
        if kw.lower() in response_text
    ]

    assert len(found_keywords) > 0, (
        f"Query: {query}\n"
        f"Expected keywords: {expected_keywords}\n"
        f"Response: {response_text[:200]}\n"
        f"Found keywords: {found_keywords}"
    )


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
