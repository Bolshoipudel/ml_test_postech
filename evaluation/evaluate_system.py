"""
Основной скрипт для evaluation LLM Assistant с использованием DeepEval.

Запуск:
    python evaluation/evaluate_system.py

Аргументы:
    --api-url: URL API (по умолчанию http://localhost:8000)
    --limit: Ограничение количества тестов (по умолчанию все)
    --category: Фильтр по категории (sql, rag, web_search, multiple, none)
    --output: Путь для сохранения результатов (по умолчанию evaluation/results/)
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from loguru import logger
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Добавление корневой директории в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.test_dataset import (
    ALL_TEST_CASES,
    SQL_TEST_CASES,
    RAG_TEST_CASES,
    WEB_SEARCH_TEST_CASES,
    MULTIPLE_TEST_CASES,
    NONE_TEST_CASES
)
from evaluation.metrics_config import (
    MetricsConfig,
    create_test_case_with_routing,
    calculate_aggregate_scores
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class EvaluationConfig:
    """Конфигурация evaluation запуска."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        output_dir: str = "evaluation/results",
        limit: Optional[int] = None,
        category: Optional[str] = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: int = 5
    ):
        self.api_url = api_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.category = category
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Создание директории для результатов
        self.output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API CLIENT
# =============================================================================

class LLMAssistantClient:
    """Клиент для взаимодействия с LLM Assistant API."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.base_url = config.api_url
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Проверка работоспособности API."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API Health: {data.get('status', 'unknown')}")
                return True
            else:
                logger.error(f"API Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Cannot connect to API: {e}")
            return False

    def query_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_history: bool = False,
        attempt: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Отправка запроса в chat endpoint.

        Args:
            message: Текст запроса
            session_id: ID сессии
            use_history: Использовать ли историю
            attempt: Номер попытки (для retry)

        Returns:
            JSON ответ или None при ошибке
        """
        try:
            payload = {
                "message": message,
                "use_history": use_history
            }
            if session_id:
                payload["session_id"] = session_id

            response = self.session.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"API returned status {response.status_code}: {response.text[:200]}"
                )

                # Retry logic
                if attempt < self.config.retry_attempts:
                    logger.info(f"Retrying... (attempt {attempt + 1}/{self.config.retry_attempts})")
                    time.sleep(self.config.retry_delay)
                    return self.query_chat(message, session_id, use_history, attempt + 1)

                return None

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for query: {message[:50]}...")

            # Retry on timeout
            if attempt < self.config.retry_attempts:
                logger.info(f"Retrying... (attempt {attempt + 1}/{self.config.retry_attempts})")
                time.sleep(self.config.retry_delay)
                return self.query_chat(message, session_id, use_history, attempt + 1)

            return None

        except Exception as e:
            logger.error(f"Error querying API: {e}")
            return None


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

class EvaluationRunner:
    """Класс для запуска evaluation процесса."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = LLMAssistantClient(config)
        self.metrics_config = MetricsConfig(
            model="gpt-4",
            threshold=0.7,
            include_reason=True
        )

        # Статистика
        self.stats = {
            "total_tests": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0
        }

        # Результаты
        self.results = []
        self.test_cases = []

    def select_test_cases(self) -> List[Dict[str, Any]]:
        """Выбор тестовых кейсов на основе фильтров."""
        # Фильтрация по категории
        if self.config.category:
            category = self.config.category.lower()
            if category == "sql":
                cases = SQL_TEST_CASES
            elif category == "rag":
                cases = RAG_TEST_CASES
            elif category == "web_search":
                cases = WEB_SEARCH_TEST_CASES
            elif category == "multiple":
                cases = MULTIPLE_TEST_CASES
            elif category == "none":
                cases = NONE_TEST_CASES
            else:
                logger.warning(f"Unknown category: {category}, using all tests")
                cases = ALL_TEST_CASES
        else:
            cases = ALL_TEST_CASES

        # Применение лимита
        if self.config.limit:
            cases = cases[:self.config.limit]

        logger.info(f"Selected {len(cases)} test cases for evaluation")
        return cases

    def run_query_and_create_test_case(
        self,
        test_case_data: Dict[str, Any]
    ) -> Optional[LLMTestCase]:
        """
        Выполнение запроса к API и создание DeepEval test case.

        Args:
            test_case_data: Данные тест-кейса из датасета

        Returns:
            LLMTestCase или None при ошибке
        """
        test_id = test_case_data["id"]
        query = test_case_data["query"]

        logger.info(f"Processing test case: {test_id}")
        logger.debug(f"Query: {query}")

        # Вызов API
        response = self.client.query_chat(message=query)

        if not response:
            logger.error(f"Failed to get response for test case: {test_id}")
            self.stats["failed_queries"] += 1
            return None

        self.stats["successful_queries"] += 1

        # Извлечение данных ответа
        actual_output = response.get("message", "")
        tools_used = response.get("tools_used", [])
        sources = response.get("sources", [])

        # Извлечение routing decision
        router_tool = "unknown"
        router_confidence = 0.0
        router_reasoning = ""

        if tools_used:
            # Первый элемент должен быть router decision
            router_decision = tools_used[0]
            if router_decision.get("tool_type") == "router":
                metadata = router_decision.get("metadata", {})
                router_tool = metadata.get("tool", "unknown").lower()
                router_confidence = router_decision.get("confidence", 0.0)
                router_reasoning = router_decision.get("reasoning", "")

        # Извлечение контекста для RAG
        retrieval_context = []
        if test_case_data["expected_tool"] in ["rag", "multiple"]:
            retrieval_context = test_case_data.get("context", [])

        # Создание DeepEval test case
        test_case = create_test_case_with_routing(
            query=query,
            actual_output=actual_output,
            expected_output=test_case_data["ground_truth"],
            expected_tool=test_case_data["expected_tool"],
            actual_tool=router_tool,
            confidence=router_confidence,
            reasoning=router_reasoning,
            retrieval_context=retrieval_context,
            context=sources
        )

        # Сохранение дополнительной информации
        test_case.additional_metadata.update({
            "test_id": test_id,
            "category": test_case_data["category"],
            "min_confidence": test_case_data.get("min_confidence", 0.7),
            "response_data": response
        })

        return test_case

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Запуск полного evaluation процесса.

        Returns:
            Словарь с результатами evaluation
        """
        logger.info("="*80)
        logger.info("Starting LLM Assistant Evaluation")
        logger.info("="*80)

        # Проверка доступности API
        if not self.client.check_health():
            logger.error("API is not available. Aborting evaluation.")
            return {"error": "API not available"}

        # Выбор тестовых кейсов
        test_cases_data = self.select_test_cases()
        self.stats["total_tests"] = len(test_cases_data)

        logger.info(f"\nConfiguration:")
        logger.info(f"  API URL: {self.config.api_url}")
        logger.info(f"  Total tests: {self.stats['total_tests']}")
        logger.info(f"  Category filter: {self.config.category or 'all'}")
        logger.info(f"  Limit: {self.config.limit or 'none'}")

        # Начало evaluation
        self.stats["start_time"] = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info("Running test cases...")
        logger.info(f"{'='*80}\n")

        # Обработка каждого тест-кейса
        for i, tc_data in enumerate(test_cases_data, 1):
            logger.info(f"\n[{i}/{len(test_cases_data)}] {tc_data['id']}: {tc_data['query'][:60]}...")

            test_case = self.run_query_and_create_test_case(tc_data)

            if test_case:
                self.test_cases.append(test_case)
                logger.info(f"  ✓ Test case created successfully")
            else:
                logger.warning(f"  ✗ Failed to create test case")

            # Небольшая пауза между запросами
            time.sleep(0.5)

        # Конец сбора данных
        logger.info(f"\n{'='*80}")
        logger.info(f"Query phase completed: {len(self.test_cases)}/{self.stats['total_tests']} successful")
        logger.info(f"{'='*80}\n")

        # Запуск DeepEval evaluation
        if not self.test_cases:
            logger.error("No test cases to evaluate!")
            return {"error": "No test cases"}

        logger.info("Running DeepEval metrics...")
        metrics = self.metrics_config.get_all_metrics(
            include_router=True,
            include_contextual=True
        )

        try:
            # DeepEval evaluate
            eval_results = evaluate(
                test_cases=self.test_cases,
                metrics=metrics,
                print_results=True
            )

            logger.success("DeepEval evaluation completed!")

        except Exception as e:
            logger.error(f"Error during DeepEval evaluation: {e}")
            eval_results = None

        # Завершение
        self.stats["end_time"] = datetime.now()
        self.stats["duration_seconds"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()

        # Формирование результатов
        results = self.compile_results(eval_results)

        # Сохранение результатов
        self.save_results(results)

        # Вывод итоговой статистики
        self.print_final_statistics(results)

        return results

    def compile_results(self, eval_results: Any) -> Dict[str, Any]:
        """
        Компиляция результатов evaluation.

        Args:
            eval_results: Результаты от DeepEval

        Returns:
            Структурированные результаты
        """
        # Извлечение метрик из тест-кейсов
        all_metric_results = []

        for test_case in self.test_cases:
            test_id = test_case.additional_metadata.get("test_id", "unknown")
            category = test_case.additional_metadata.get("category", "unknown")

            # Placeholder для результатов метрик
            # В реальности DeepEval их добавляет в test_case
            # Здесь упрощенная версия

            test_result = {
                "test_id": test_id,
                "query": test_case.input,
                "category": category,
                "expected_tool": test_case.additional_metadata.get("expected_tool"),
                "actual_tool": test_case.additional_metadata.get("actual_tool"),
                "confidence": test_case.additional_metadata.get("confidence"),
                "reasoning": test_case.additional_metadata.get("reasoning"),
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output
            }

            all_metric_results.append(test_result)

        return {
            "metadata": {
                "evaluation_date": self.stats["start_time"].isoformat(),
                "duration_seconds": self.stats["duration_seconds"],
                "api_url": self.config.api_url,
                "total_tests": self.stats["total_tests"],
                "successful_queries": self.stats["successful_queries"],
                "failed_queries": self.stats["failed_queries"]
            },
            "test_results": all_metric_results,
            "aggregate_stats": self.calculate_statistics(all_metric_results)
        }

    def calculate_statistics(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Расчет агрегированной статистики."""
        if not test_results:
            return {}

        # Группировка по expected_tool
        by_tool = {}
        for result in test_results:
            tool = result["expected_tool"]
            if tool not in by_tool:
                by_tool[tool] = []
            by_tool[tool].append(result)

        # Статистика по каждому tool
        tool_stats = {}
        for tool, results in by_tool.items():
            correct = sum(
                1 for r in results
                if r["expected_tool"] == r["actual_tool"]
            )
            total = len(results)
            avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

            tool_stats[tool] = {
                "total": total,
                "correct_routing": correct,
                "routing_accuracy": correct / total if total > 0 else 0,
                "average_confidence": avg_confidence
            }

        # Общая статистика
        total_correct = sum(stats["correct_routing"] for stats in tool_stats.values())
        total_tests = sum(stats["total"] for stats in tool_stats.values())

        return {
            "by_tool": tool_stats,
            "overall": {
                "total_tests": total_tests,
                "correct_routing": total_correct,
                "routing_accuracy": total_correct / total_tests if total_tests > 0 else 0
            }
        }

    def save_results(self, results: Dict[str, Any]) -> None:
        """Сохранение результатов в JSON файл."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_dir / f"evaluation_results_{timestamp}.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.success(f"Results saved to: {output_file}")

            # Также сохранить как latest
            latest_file = self.config.output_dir / "evaluation_results_latest.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Latest results saved to: {latest_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_final_statistics(self, results: Dict[str, Any]) -> None:
        """Вывод итоговой статистики."""
        metadata = results.get("metadata", {})
        stats = results.get("aggregate_stats", {}).get("overall", {})
        by_tool = results.get("aggregate_stats", {}).get("by_tool", {})

        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)

        logger.info(f"\nMetadata:")
        logger.info(f"  Duration: {metadata.get('duration_seconds', 0):.1f} seconds")
        logger.info(f"  Total tests: {metadata.get('total_tests', 0)}")
        logger.info(f"  Successful: {metadata.get('successful_queries', 0)}")
        logger.info(f"  Failed: {metadata.get('failed_queries', 0)}")

        logger.info(f"\nOverall Performance:")
        logger.info(f"  Routing Accuracy: {stats.get('routing_accuracy', 0):.1%}")
        logger.info(f"  Correct: {stats.get('correct_routing', 0)}/{stats.get('total_tests', 0)}")

        logger.info(f"\nPerformance by Tool:")
        for tool, tool_stats in by_tool.items():
            accuracy = tool_stats.get('routing_accuracy', 0)
            confidence = tool_stats.get('average_confidence', 0)
            logger.info(
                f"  {tool.upper():12s}: "
                f"Accuracy {accuracy:.1%} | "
                f"Confidence {confidence:.2f} | "
                f"Tests {tool_stats.get('total', 0)}"
            )

        logger.info("\n" + "="*80)


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM Assistant with DeepEval"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the LLM Assistant API"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases"
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=["sql", "rag", "web_search", "multiple", "none", "all"],
        default=None,
        help="Filter test cases by category"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Парсинг аргументов
    args = parse_arguments()

    # Конфигурация
    config = EvaluationConfig(
        api_url=args.api_url,
        output_dir=args.output,
        limit=args.limit,
        category=args.category if args.category != "all" else None,
        timeout=args.timeout
    )

    # Запуск evaluation
    runner = EvaluationRunner(config)
    results = runner.run_evaluation()

    # Выход
    if "error" in results:
        logger.error(f"Evaluation failed: {results['error']}")
        sys.exit(1)
    else:
        logger.success("Evaluation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
