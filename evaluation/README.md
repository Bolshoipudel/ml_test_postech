# Evaluation System - Phase 7

Система evaluation для LLM Assistant с использованием DeepEval метрик.

## 📁 Структура файлов

```
evaluation/
├── __init__.py                      # Package init
├── README.md                        # Эта инструкция
├── test_dataset.py                  # 50 тестовых кейсов (SQL, RAG, Web, Multiple, None)
├── metrics_config.py                # 4 DeepEval метрики (3 стандартные + 1 custom)
├── evaluate_system.py               # Основной evaluation runner
├── test_routing_accuracy.py         # Pytest тесты для Router Agent
├── test_end_to_end.py               # E2E тесты с DeepEval
├── generate_report.py               # Генератор markdown отчетов
└── results/                         # Директория для результатов
    ├── evaluation_results_latest.json
    └── evaluation_report.md
```

## 🚀 Быстрый старт

### 1. Запуск Docker контейнера

```bash
# Из корневой директории проекта
docker-compose up -d

# Проверка что API работает
curl http://localhost:8000/api/v1/health
```

### 2. Тестовый evaluation run (5-10 тестов)

```bash
# Запуск с лимитом
python evaluation/evaluate_system.py --limit 10

# Или по категориям
python evaluation/evaluate_system.py --category sql --limit 5
```

### 3. Генерация markdown отчета

```bash
python evaluation/generate_report.py

# Просмотр отчета
cat evaluation/results/evaluation_report.md
```

### 4. Запуск pytest тестов

```bash
# Тесты точности роутинга
pytest evaluation/test_routing_accuracy.py -v

# E2E тесты (только первые 3 из каждой категории)
pytest evaluation/test_end_to_end.py -v

# Запуск конкретного теста
pytest evaluation/test_routing_accuracy.py -v -k "test_sql"
```

## 📊 Полный evaluation run (все 50 тестов)

```bash
# Займет 20-30 минут из-за API calls
python evaluation/evaluate_system.py

# После завершения - генерация отчета
python evaluation/generate_report.py

# Просмотр результатов
cat evaluation/results/evaluation_report.md
```

## 🎯 API Endpoint /evaluate

Новый endpoint для оценки одиночных запросов:

```bash
# Пример запроса
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Сколько программистов в команде?",
    "expected_output": "В команде 2 программиста",
    "retrieval_context": ["SQLite database: employees table"]
  }'
```

**Ответ:**
```json
{
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
      "passed": true,
      "reason": "Response directly answers the question"
    },
    "Faithfulness": {
      "score": 0.88,
      "threshold": 0.7,
      "passed": true,
      "reason": "No hallucinations detected"
    }
  }
}
```

## 📈 Метрики DeepEval

### 1. **Answer Relevancy Metric** (threshold: 0.7)
Оценивает насколько ответ релевантен исходному вопросу.

### 2. **Faithfulness Metric** (threshold: 0.7)
Проверяет отсутствие галлюцинаций - все утверждения в ответе должны подтверждаться контекстом.

### 3. **Contextual Relevancy Metric** (threshold: 0.7)
Оценивает качество retrieval - насколько извлеченный контекст релевантен для ответа.

### 4. **Router Accuracy Metric** (custom, threshold: 0.7)
Проверяет правильность выбора инструмента Router Agent.

## 📝 Тестовый датасет

**Всего: 50 тестовых кейсов**

- **SQL** (12): Запросы к БД команды (team_mock.db)
- **RAG** (12): Поиск в документации PT продуктов
- **Web Search** (12): Актуальные новости и тренды
- **Multiple** (10): Комбинированные запросы (SQL+RAG, SQL+Web и т.д.)
- **None** (4): Нерелевантные запросы (новая функция Router Agent)

## 🎯 Целевые результаты

- **Router Accuracy**: ≥85% (цель: 90%)
- **Answer Relevancy**: ≥0.75
- **Faithfulness**: ≥0.70
- **Contextual Relevancy**: ≥0.70

## 🛠️ Troubleshooting

### API не отвечает
```bash
# Проверка Docker
docker-compose ps

# Перезапуск
docker-compose restart

# Логи
docker-compose logs -f app
```

### Timeout ошибки
```bash
# Увеличить timeout в evaluate_system.py
python evaluation/evaluate_system.py --timeout 60
```

### Ошибки DeepEval
Убедитесь что установлены зависимости:
```bash
pip install -r requirements.txt
```

## 📦 Зависимости

- `deepeval==0.20.99` - Framework для evaluation
- `pytest==7.4.4` - Testing framework
- `pytest-asyncio==0.23.3` - Async support для pytest
- `requests` - HTTP клиент

## 🎓 Примеры использования

### Запуск только SQL тестов
```bash
pytest evaluation/test_routing_accuracy.py -v -k "test_sql"
```

### Запуск E2E тестов для RAG
```bash
pytest evaluation/test_end_to_end.py -v -k "test_rag"
```

### Просмотр статистики датасета
```bash
python evaluation/test_dataset.py
```

### Проверка метрик конфигурации
```bash
python evaluation/metrics_config.py
```

## 📊 Формат результатов

### evaluation_results_latest.json
```json
{
  "metadata": {
    "evaluation_date": "2025-10-15T01:30:00",
    "duration_seconds": 1250.5,
    "total_tests": 50,
    "successful_queries": 48,
    "failed_queries": 2
  },
  "aggregate_stats": {
    "overall": {
      "routing_accuracy": 0.88
    },
    "by_tool": {
      "sql": {"routing_accuracy": 0.92, "average_confidence": 0.91},
      "rag": {"routing_accuracy": 0.87, "average_confidence": 0.85}
    }
  },
  "test_results": [...]
}
```

### evaluation_report.md
Человекочитаемый markdown отчет с:
- Executive Summary
- Router Performance таблица
- Test Results by Category
- Failed Tests с reasoning
- Рекомендации по улучшению

---

**Автор:** ML Test Task - Positive Technologies
**Дата:** October 2025
**Фаза:** 7/7 (Evaluation System)
