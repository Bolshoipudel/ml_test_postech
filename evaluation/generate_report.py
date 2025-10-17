"""
Генератор markdown отчета из результатов evaluation.

Читает JSON результаты из evaluation/results/evaluation_results_latest.json
и генерирует человекочитаемый markdown отчет с:
- Executive Summary
- Router Performance по каждому tool
- DeepEval metrics scores
- Failed test cases с reasoning
- Рекомендации по улучшению

Запуск:
    python evaluation/generate_report.py
    python evaluation/generate_report.py --input evaluation/results/custom_results.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT = "evaluation/results/evaluation_results_latest.json"
DEFAULT_OUTPUT = "evaluation/results/evaluation_report.md"


# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_results(results_file: str) -> Dict[str, Any]:
    """
    Загрузка результатов из JSON файла.

    Args:
        results_file: Путь к JSON файлу с результатами

    Returns:
        Dict с результатами evaluation
    """
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Results file not found: {results_file}")
        print(f"   Please run evaluation first: python evaluation/evaluate_system.py")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in results file: {e}")
        sys.exit(1)


# =============================================================================
# REPORT SECTIONS
# =============================================================================

def generate_header() -> str:
    """Генерация заголовка отчета."""
    return f"""# LLM Assistant Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""


def generate_executive_summary(results: Dict[str, Any]) -> str:
    """
    Генерация Executive Summary секции.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown секция
    """
    metadata = results.get("metadata", {})
    overall = results.get("aggregate_stats", {}).get("overall", {})

    total_tests = metadata.get("total_tests", 0)
    successful = metadata.get("successful_queries", 0)
    failed = metadata.get("failed_queries", 0)
    duration = metadata.get("duration_seconds", 0)
    routing_accuracy = overall.get("routing_accuracy", 0)

    # Определение статуса
    if routing_accuracy >= 0.9:
        status = "🟢 EXCELLENT"
    elif routing_accuracy >= 0.85:
        status = "🟡 GOOD"
    elif routing_accuracy >= 0.7:
        status = "🟠 ACCEPTABLE"
    else:
        status = "🔴 NEEDS IMPROVEMENT"

    return f"""## Executive Summary

**Overall Status:** {status}

| Metric | Value |
|--------|-------|
| **Total Tests** | {total_tests} |
| **Successful Queries** | {successful} ({successful/total_tests*100:.1f}%) |
| **Failed Queries** | {failed} ({failed/total_tests*100:.1f}%) |
| **Overall Routing Accuracy** | {routing_accuracy:.1%} |
| **Evaluation Duration** | {duration:.1f} seconds |
| **API URL** | {metadata.get('api_url', 'N/A')} |
| **Evaluation Date** | {metadata.get('evaluation_date', 'N/A')} |

---

"""


def generate_routing_performance_table(results: Dict[str, Any]) -> str:
    """
    Генерация таблицы Router Performance.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown таблица
    """
    by_tool = results.get("aggregate_stats", {}).get("by_tool", {})

    if not by_tool:
        return "## Router Agent Performance\n\nNo routing data available.\n\n---\n\n"

    table = """## Router Agent Performance

| Tool Type | Accuracy | Avg Confidence | Total Tests | Status |
|-----------|----------|----------------|-------------|--------|
"""

    # Сортировка по accuracy (descending)
    sorted_tools = sorted(
        by_tool.items(),
        key=lambda x: x[1].get("routing_accuracy", 0),
        reverse=True
    )

    for tool, stats in sorted_tools:
        accuracy = stats.get("routing_accuracy", 0)
        confidence = stats.get("average_confidence", 0)
        total = stats.get("total", 0)
        correct = stats.get("correct_routing", 0)

        # Определение статуса
        if accuracy >= 0.9:
            status = "✅ PASS"
        elif accuracy >= 0.8:
            status = "⚠️ WARN"
        else:
            status = "❌ FAIL"

        table += f"| {tool.upper():9s} | {accuracy:7.1%} | {confidence:14.2f} | {total:11d} | {status:6s} |\n"

    table += "\n**Legend:**\n"
    table += "- ✅ PASS: Accuracy >= 90%\n"
    table += "- ⚠️ WARN: Accuracy >= 80%\n"
    table += "- ❌ FAIL: Accuracy < 80%\n\n"
    table += "---\n\n"

    return table


def generate_test_results_summary(results: Dict[str, Any]) -> str:
    """
    Генерация сводки по результатам тестов.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown секция
    """
    test_results = results.get("test_results", [])

    if not test_results:
        return "## Test Results Summary\n\nNo test results available.\n\n---\n\n"

    # Подсчет статистики
    by_category = {}
    failed_tests = []

    for result in test_results:
        category = result.get("category", "unknown")
        expected = result.get("expected_tool", "")
        actual = result.get("actual_tool", "")
        test_id = result.get("test_id", "unknown")

        if category not in by_category:
            by_category[category] = {"total": 0, "correct": 0}

        by_category[category]["total"] += 1

        if expected == actual:
            by_category[category]["correct"] += 1
        else:
            failed_tests.append(result)

    # Генерация таблицы по категориям
    section = """## Test Results by Category

| Category | Total Tests | Correct | Accuracy |
|----------|-------------|---------|----------|
"""

    for category, stats in sorted(by_category.items()):
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0

        section += f"| {category:20s} | {total:11d} | {correct:7d} | {accuracy:8.1%} |\n"

    section += "\n---\n\n"

    return section


def generate_failed_tests_section(results: Dict[str, Any]) -> str:
    """
    Генерация секции с неуспешными тестами.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown секция
    """
    test_results = results.get("test_results", [])

    # Фильтрация failed тестов
    failed_tests = [
        r for r in test_results
        if r.get("expected_tool") != r.get("actual_tool")
    ]

    if not failed_tests:
        return """## Failed Tests

✅ **All tests passed routing correctly!**

---

"""

    section = f"""## Failed Tests

Found **{len(failed_tests)}** tests with incorrect routing:

"""

    for i, result in enumerate(failed_tests, 1):
        test_id = result.get("test_id", "unknown")
        query = result.get("query", "")
        expected = result.get("expected_tool", "")
        actual = result.get("actual_tool", "")
        confidence = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")

        section += f"""### {i}. Test ID: `{test_id}`

**Query:** {query}

**Expected Tool:** `{expected.upper()}`
**Actual Tool:** `{actual.upper()}`
**Confidence:** {confidence:.2f}

**Router Reasoning:**
```
{reasoning[:300]}{"..." if len(reasoning) > 300 else ""}
```

---

"""

    return section


def generate_recommendations(results: Dict[str, Any]) -> str:
    """
    Генерация рекомендаций по улучшению.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown секция
    """
    by_tool = results.get("aggregate_stats", {}).get("by_tool", {})
    overall = results.get("aggregate_stats", {}).get("overall", {})

    routing_accuracy = overall.get("routing_accuracy", 0)

    recommendations = []

    # Рекомендации по общей accuracy
    if routing_accuracy < 0.85:
        recommendations.append(
            "**Improve Overall Routing Accuracy**: Current accuracy is below target (85%). "
            "Consider reviewing and improving Router Agent prompts with more few-shot examples."
        )

    # Рекомендации по отдельным tools
    for tool, stats in by_tool.items():
        accuracy = stats.get("routing_accuracy", 0)
        confidence = stats.get("average_confidence", 0)

        if accuracy < 0.8:
            recommendations.append(
                f"**Improve {tool.upper()} Routing**: Accuracy {accuracy:.1%} is below 80%. "
                f"Add more training examples for {tool} queries in Router prompts."
            )

        if confidence < 0.75:
            recommendations.append(
                f"**Increase {tool.upper()} Confidence**: Average confidence {confidence:.2f} is low. "
                f"Review ambiguous queries and clarify routing logic."
            )

    # Рекомендации по failed queries
    metadata = results.get("metadata", {})
    failed_queries = metadata.get("failed_queries", 0)

    if failed_queries > 0:
        recommendations.append(
            f"**Fix Failed Queries**: {failed_queries} queries failed to execute. "
            "Check API stability, timeouts, and error handling."
        )

    # Финальная секция
    if not recommendations:
        return """## Recommendations

✅ **System is performing well!**

All metrics are within acceptable ranges. Continue monitoring performance.

---

"""

    section = "## Recommendations\n\n"

    for i, rec in enumerate(recommendations, 1):
        section += f"{i}. {rec}\n\n"

    section += "---\n\n"

    return section


def generate_metrics_summary(results: Dict[str, Any]) -> str:
    """
    Генерация сводки по DeepEval метрикам.

    Args:
        results: Результаты evaluation

    Returns:
        Markdown секция
    """
    deepeval_metrics = results.get("aggregate_stats", {}).get("deepeval_metrics", {})

    if not deepeval_metrics:
        return """## DeepEval Metrics Summary

_No DeepEval metrics data available. Metrics may not have been run during evaluation._

**Metrics Configuration:**
- Answer Relevancy Metric (threshold: 0.7)
- Faithfulness Metric (threshold: 0.7)
- Contextual Relevancy Metric (threshold: 0.7)
- Router Accuracy Metric (threshold: 0.7)

---

"""

    # Генерация таблицы с метриками
    section = """## DeepEval Metrics Summary

| Metric Name | Avg Score | Min | Max | Tests | Pass Rate | Status |
|-------------|-----------|-----|-----|-------|-----------|--------|
"""

    # Сортировка метрик по average_score
    sorted_metrics = sorted(
        deepeval_metrics.items(),
        key=lambda x: x[1].get("average_score", 0),
        reverse=True
    )

    for metric_name, metric_stats in sorted_metrics:
        avg_score = metric_stats.get("average_score", 0)
        min_score = metric_stats.get("min_score", 0)
        max_score = metric_stats.get("max_score", 0)
        count = metric_stats.get("count", 0)
        pass_rate = metric_stats.get("pass_rate", 0)

        # Определение статуса
        if avg_score >= 0.85:
            status = "✅ EXCELLENT"
        elif avg_score >= 0.7:
            status = "🟢 GOOD"
        elif avg_score >= 0.5:
            status = "🟡 ACCEPTABLE"
        else:
            status = "🔴 POOR"

        section += f"| {metric_name:20s} | {avg_score:9.3f} | {min_score:3.2f} | {max_score:3.2f} | {count:5d} | {pass_rate:9.1%} | {status:13s} |\n"

    section += "\n**Legend:**\n"
    section += "- ✅ EXCELLENT: Average score >= 0.85\n"
    section += "- 🟢 GOOD: Average score >= 0.70\n"
    section += "- 🟡 ACCEPTABLE: Average score >= 0.50\n"
    section += "- 🔴 POOR: Average score < 0.50\n\n"

    section += "**Metrics Description:**\n"
    section += "- **Answer Relevancy**: Measures how relevant the LLM's answer is to the question\n"
    section += "- **Faithfulness**: Checks if the answer is faithful to the context (no hallucinations)\n"
    section += "- **Contextual Relevancy**: Evaluates quality of retrieved context (for RAG)\n"
    section += "- **Router Accuracy**: Custom metric for routing decision correctness\n\n"

    section += "---\n\n"

    return section


def generate_footer() -> str:
    """Генерация footer отчета."""
    return """## Conclusion

This report provides a comprehensive overview of the LLM Assistant evaluation results.
Review the recommendations and failed tests to identify areas for improvement.

For detailed test results, see `evaluation_results_latest.json`.

---

**Generated by:** LLM Assistant Evaluation System
**Framework:** DeepEval + Pytest
**Repository:** https://github.com/PositiveTechnologies/ml_test_postech
"""


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_full_report(results: Dict[str, Any]) -> str:
    """
    Генерация полного markdown отчета.

    Args:
        results: Результаты evaluation

    Returns:
        Полный markdown отчет
    """
    report = ""
    report += generate_header()
    report += generate_executive_summary(results)
    report += generate_routing_performance_table(results)
    report += generate_test_results_summary(results)
    report += generate_metrics_summary(results)
    report += generate_failed_tests_section(results)
    report += generate_recommendations(results)
    report += generate_footer()

    return report


def save_report(report: str, output_file: str) -> None:
    """
    Сохранение отчета в файл.

    Args:
        report: Markdown отчет
        output_file: Путь для сохранения
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✅ Report saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error saving report: {e}")
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Generate markdown evaluation report from JSON results"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input JSON file with evaluation results (default: {DEFAULT_INPUT})"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output markdown file (default: {DEFAULT_OUTPUT})"
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Print report to stdout in addition to saving"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    print("="*80)
    print("LLM Assistant - Evaluation Report Generator")
    print("="*80)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}\n")

    # Загрузка результатов
    print("Loading results...")
    results = load_results(args.input)

    # Генерация отчета
    print("Generating report...")
    report = generate_full_report(results)

    # Сохранение отчета
    save_report(report, args.output)

    # Вывод в stdout (если запрошено)
    if args.print:
        print("\n" + "="*80)
        print("REPORT PREVIEW")
        print("="*80 + "\n")
        print(report)

    print("\n✅ Report generation completed successfully!")
    print(f"\nView report: cat {args.output}")


if __name__ == "__main__":
    main()
