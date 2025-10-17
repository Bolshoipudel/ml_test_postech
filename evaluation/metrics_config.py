"""
Конфигурация метрик DeepEval для оценки качества LLM Assistant.

Содержит:
- 3 стандартные метрики DeepEval (Answer Relevancy, Faithfulness, Contextual Relevancy)
- 1 custom метрику (Router Accuracy)
- Утилиты для настройки и использования метрик
"""

from typing import List, Optional, Dict, Any
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    BaseMetric
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from loguru import logger


# =============================================================================
# CUSTOM METRIC: Router Accuracy
# =============================================================================

class RouterAccuracyMetric(BaseMetric):
    """
    Кастомная метрика для оценки точности Router Agent.

    Проверяет:
    1. Правильность выбора инструмента (expected vs actual)
    2. Уровень уверенности (confidence >= threshold)
    3. Качество reasoning
    """

    def __init__(
        self,
        threshold: float = 0.7,
        confidence_threshold: float = 0.7,
        include_reason: bool = True
    ):
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.include_reason = include_reason
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Оценка точности роутинга.

        Args:
            test_case: Тест-кейс с дополнительными параметрами в metadata

        Returns:
            Score: 1.0 если роутинг правильный, 0.0 если нет
        """
        try:
            # Извлечение данных из metadata (безопасный доступ)
            metadata = getattr(test_case, 'additional_metadata', {})
            expected_tool = metadata.get("expected_tool", "").lower()
            actual_tool = metadata.get("actual_tool", "").lower()
            confidence = metadata.get("confidence", 0.0)
            reasoning = metadata.get("reasoning", "")

            # Проверка наличия необходимых данных
            if not expected_tool or not actual_tool:
                self.score = 0.0
                self.reason = "Missing routing data in metadata"
                self.success = False
                return self.score

            # Проверка правильности выбора инструмента
            tool_match = (expected_tool == actual_tool)

            # Проверка уверенности
            confidence_pass = (confidence >= self.confidence_threshold)

            # Расчет финального score
            if tool_match and confidence_pass:
                self.score = 1.0
                self.reason = f"✅ Correct routing: {actual_tool} (confidence: {confidence:.2f})"
                self.success = True
            elif tool_match and not confidence_pass:
                self.score = 0.5
                self.reason = f"⚠️ Correct tool but low confidence: {actual_tool} (confidence: {confidence:.2f} < {self.confidence_threshold})"
                self.success = False
            else:
                self.score = 0.0
                self.reason = f"❌ Incorrect routing: expected {expected_tool}, got {actual_tool} (confidence: {confidence:.2f})"
                self.success = False

            # Добавление reasoning в отчет
            if self.include_reason and reasoning:
                self.reason += f"\nReasoning: {reasoning[:200]}"

            return self.score

        except Exception as e:
            logger.error(f"Error in RouterAccuracyMetric: {e}")
            self.score = 0.0
            self.reason = f"Error: {str(e)}"
            self.success = False
            return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async версия measure."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Проверка успешности метрики."""
        return self.success

    @property
    def __name__(self):
        return "Router Accuracy"


# =============================================================================
# METRIC CONFIGURATIONS
# =============================================================================

class MetricsConfig:
    """Конфигурация метрик с настраиваемыми параметрами."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        threshold: float = 0.7,
        include_reason: bool = True
    ):
        """
        Args:
            model: Модель для LLM-as-a-judge метрик
            threshold: Минимальный порог для успешной метрики
            include_reason: Включать ли reasoning в результаты
        """
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason

    def get_answer_relevancy_metric(self) -> AnswerRelevancyMetric:
        """
        Метрика релевантности ответа.

        Оценивает насколько ответ релевантен исходному вопросу.
        """
        return AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason
        )

    def get_faithfulness_metric(self) -> FaithfulnessMetric:
        """
        Метрика соответствия источникам (faithfulness).

        Проверяет отсутствие галлюцинаций - все утверждения
        в ответе должны подтверждаться контекстом.
        """
        return FaithfulnessMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason
        )

    def get_contextual_relevancy_metric(self) -> ContextualRelevancyMetric:
        """
        Метрика релевантности контекста.

        Оценивает качество retrieval - насколько извлеченный
        контекст релевантен для ответа на вопрос.
        """
        return ContextualRelevancyMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=self.include_reason
        )

    def get_router_accuracy_metric(
        self,
        confidence_threshold: float = 0.7
    ) -> RouterAccuracyMetric:
        """
        Кастомная метрика точности роутинга.

        Args:
            confidence_threshold: Минимальная уверенность роутера
        """
        return RouterAccuracyMetric(
            threshold=self.threshold,
            confidence_threshold=confidence_threshold,
            include_reason=self.include_reason
        )

    def get_all_metrics(
        self,
        include_router: bool = True,
        include_contextual: bool = True
    ) -> List[BaseMetric]:
        """
        Получить все метрики для evaluation.

        Args:
            include_router: Включать ли Router Accuracy
            include_contextual: Включать ли Contextual Relevancy
                               (только для RAG запросов)

        Returns:
            Список метрик
        """
        metrics = [
            self.get_answer_relevancy_metric(),
            self.get_faithfulness_metric()
        ]

        if include_contextual:
            metrics.append(self.get_contextual_relevancy_metric())

        if include_router:
            metrics.append(self.get_router_accuracy_metric())

        return metrics

    def get_metrics_for_tool_type(self, tool_type: str) -> List[BaseMetric]:
        """
        Получить метрики в зависимости от типа инструмента.

        Args:
            tool_type: Тип инструмента (sql, rag, web_search, multiple)

        Returns:
            Список подходящих метрик
        """
        tool_type = tool_type.lower()

        # Базовые метрики для всех типов
        metrics = [
            self.get_answer_relevancy_metric(),
            self.get_router_accuracy_metric()
        ]

        # Для RAG и Multiple добавляем Contextual Relevancy
        if tool_type in ["rag", "multiple"]:
            metrics.append(self.get_contextual_relevancy_metric())

        # Faithfulness для всех, кроме Web Search (там нет контекста)
        if tool_type != "web_search":
            metrics.append(self.get_faithfulness_metric())

        return metrics


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_test_case_with_routing(
    query: str,
    actual_output: str,
    expected_output: str,
    expected_tool: str,
    actual_tool: str,
    confidence: float,
    reasoning: str,
    retrieval_context: Optional[List[str]] = None,
    context: Optional[List[str]] = None
) -> LLMTestCase:
    """
    Создание тест-кейса с данными о роутинге для метрик.

    Args:
        query: Входной запрос
        actual_output: Фактический ответ системы
        expected_output: Ожидаемый ответ (ground truth)
        expected_tool: Ожидаемый инструмент
        actual_tool: Фактически выбранный инструмент
        confidence: Уверенность роутера
        reasoning: Reasoning роутера
        retrieval_context: Контекст для RAG
        context: Источники информации

    Returns:
        LLMTestCase с метаданными роутинга
    """
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context or [],
        context=context or []
    )

    # Добавляем metadata как атрибут ПОСЛЕ создания
    test_case.additional_metadata = {
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "confidence": confidence,
        "reasoning": reasoning
    }

    return test_case


def print_metric_results(
    metric: BaseMetric,
    test_case_id: str
) -> None:
    """
    Вывод результатов метрики в читаемом виде.

    Args:
        metric: Метрика после вызова measure()
        test_case_id: ID тест-кейса
    """
    status = "✅ PASS" if metric.is_successful() else "❌ FAIL"

    print(f"\n{'='*80}")
    print(f"Test Case: {test_case_id}")
    print(f"Metric: {metric.__name__}")
    print(f"Score: {metric.score:.3f}")
    print(f"Threshold: {metric.threshold}")
    print(f"Status: {status}")

    if hasattr(metric, 'reason') and metric.reason:
        print(f"\nReason:")
        print(f"{metric.reason}")

    print(f"{'='*80}")


def calculate_aggregate_scores(
    metric_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Расчет агрегированных scores по всем метрикам.

    Args:
        metric_results: Список результатов метрик

    Returns:
        Словарь с агрегированными scores
    """
    if not metric_results:
        return {}

    # Группировка по метрикам
    metrics_data = {}
    for result in metric_results:
        metric_name = result.get("metric_name", "unknown")
        score = result.get("score", 0.0)

        if metric_name not in metrics_data:
            metrics_data[metric_name] = []
        metrics_data[metric_name].append(score)

    # Расчет средних значений
    aggregate = {}
    for metric_name, scores in metrics_data.items():
        aggregate[metric_name] = {
            "average": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
            "pass_rate": sum(1 for s in scores if s >= 0.7) / len(scores)
        }

    return aggregate


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Глобальная конфигурация по умолчанию
default_config = MetricsConfig(
    model="gpt-4.1",
    threshold=0.7,
    include_reason=True
)


def get_default_metrics(
    include_router: bool = True,
    include_contextual: bool = True
) -> List[BaseMetric]:
    """
    Получить метрики с конфигурацией по умолчанию.

    Args:
        include_router: Включать ли Router Accuracy
        include_contextual: Включать ли Contextual Relevancy

    Returns:
        Список метрик
    """
    return default_config.get_all_metrics(
        include_router=include_router,
        include_contextual=include_contextual
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Пример использования метрик
    print("DeepEval Metrics Configuration")
    print("=" * 80)

    # Создание конфигурации
    config = MetricsConfig(
        model="gpt-4.1",
        threshold=0.7,
        include_reason=True
    )

    # Получение всех метрик
    all_metrics = config.get_all_metrics()

    print(f"\nConfigured {len(all_metrics)} metrics:")
    for metric in all_metrics:
        print(f"  - {metric.__name__}")

    print("\n" + "=" * 80)

    # Пример создания тест-кейса
    test_case = create_test_case_with_routing(
        query="Сколько программистов в команде?",
        actual_output="В команде работает 2 программиста.",
        expected_output="В команде 2 программиста",
        expected_tool="sql",
        actual_tool="sql",
        confidence=0.95,
        reasoning="Вопрос о количестве требует SQL запроса"
    )

    print("\nExample test case created successfully!")
    print(f"Input: {test_case.input}")
    print(f"Metadata: {test_case.additional_metadata}")
