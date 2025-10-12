# 🚀 Quick Start Guide - SQL Agent Ready!

## ✅ Что уже реализовано

**Фаза 1: Инфраструктура** ✅
- FastAPI приложение
- API endpoints
- База данных PostgreSQL
- Docker конфигурация

**Фаза 3: SQL Agent** ✅ (ГОТОВ К ИСПОЛЬЗОВАНИЮ!)
- Natural Language to SQL преобразование
- Подключение к PostgreSQL
- Guardrails (только SELECT запросы)
- Форматирование результатов на естественном языке
- Интеграция с chat endpoint

## 📦 Быстрый старт (Docker)

### 1. Настройте .env файл

```bash
# Создайте .env из примера
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

Откройте `.env` и добавьте ваш OpenAI API ключ:
```env
OPENAI_API_KEY=sk-your-key-here
```

**Важно:** Для работы SQL Agent нужен только `OPENAI_API_KEY`. Остальное опционально.

### 2. Запустите с Docker

```bash
# Windows
docker-run.bat

# Linux/Mac
docker-compose up --build -d
```

Это запустит:
- FastAPI приложение на http://localhost:8000
- PostgreSQL с тестовыми данными

### 3. Проверьте, что всё работает

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Тестовый запрос к SQL Agent
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Сколько разработчиков в команде?\"}"
```

### 4. Откройте Swagger UI

Перейдите в браузере: http://localhost:8000/docs

Здесь вы можете:
- Посмотреть все доступные endpoints
- Попробовать API прямо в браузере
- Увидеть примеры запросов/ответов

## 🧪 Примеры запросов к SQL Agent

### Через Swagger UI (самый простой способ)

1. Откройте http://localhost:8000/docs
2. Найдите `POST /api/v1/chat`
3. Нажмите "Try it out"
4. Вставьте JSON:
```json
{
  "message": "Сколько инцидентов открыто для PT Application Inspector?",
  "use_history": false
}
```
5. Нажмите "Execute"

### Через curl

**Вопрос 1: Статистика по команде**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Сколько человек работает в отделе разработки?",
    "use_history": false
  }'
```

**Вопрос 2: Информация о продуктах**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Какие продукты есть в компании?",
    "use_history": false
  }'
```

**Вопрос 3: Анализ инцидентов**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Покажи все критические баги которые сейчас открыты",
    "use_history": false
  }'
```

**Вопрос 4: Статус фич**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Кто работает над RAG-based Documentation Search?",
    "use_history": false
  }'
```

## 📊 Что есть в базе данных

База содержит тестовые данные команды Positive Technologies:

- **6 отделов**: Development, Research, QA, Product Management, DevOps, Documentation
- **5 продуктов**: PT AI, PT NAD, PT Sandbox, PT MultiScanner, PT ISIM
- **12 сотрудников** с разными навыками и позициями
- **8 фич в разработке**: RAG search, LLM integration, Threat Intelligence, и др.
- **8 инцидентов**: от критических до low priority

## 🔍 Ключевые слова для SQL Agent

SQL Agent активируется при наличии ключевых слов:
- **сколько**, **кто работает**, **команда**, **разработчик**
- **инцидент**, **баг**, **продукт**, **отдел**
- **фича**, **feature**

Примеры:
- ✅ "Сколько багов открыто?" → SQL Agent
- ✅ "Кто работает над PT AI?" → SQL Agent
- ❌ "Как работает LangChain?" → будет RAG (когда реализуем)

## 🛠️ Локальный запуск (без Docker)

### Требования
- Python 3.11+
- PostgreSQL 15+

### Шаги

1. **Установите PostgreSQL** и создайте БД:
```sql
CREATE DATABASE llm_assistant_db;
CREATE USER llm_assistant WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE llm_assistant_db TO llm_assistant;
```

2. **Примените схему**:
```bash
psql -U llm_assistant -d llm_assistant_db -f data/sql/01_schema.sql
psql -U llm_assistant -d llm_assistant_db -f data/sql/02_sample_data.sql
```

3. **Установите зависимости**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

4. **Настройте .env**:
```env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=llm_assistant
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=llm_assistant_db
```

5. **Запустите приложение**:
```bash
python -m app.main
```

## 🐛 Troubleshooting

### Проблема: "Failed to connect to database"

**Решение:**
1. Убедитесь, что PostgreSQL запущен
2. Проверьте credentials в `.env`
3. Проверьте, что БД создана и схема применена

### Проблема: "OPENAI_API_KEY is not set"

**Решение:**
1. Создайте `.env` файл из `.env.example`
2. Добавьте ваш OpenAI API ключ
3. Перезапустите приложение

### Проблема: SQL Agent возвращает ошибку "Operation DELETE is not allowed"

**Решение:**
Это нормально! Guardrails блокируют опасные операции. SQL Agent принимает только SELECT запросы для безопасности.

### Проблема: Низкое качество SQL запросов

**Решение:**
1. Используйте более конкретные вопросы
2. Укажите имена таблиц/полей из схемы
3. Попробуйте переформулировать вопрос

## 📝 Логи и отладка

### Просмотр логов (Docker)
```bash
docker-compose logs -f app
```

### Просмотр логов БД
```bash
docker-compose logs -f postgres
```

### Подключение к БД для проверки
```bash
docker exec -it llm-assistant-postgres psql -U llm_assistant -d llm_assistant_db
```

Полезные SQL команды:
```sql
-- Список таблиц
\dt

-- Структура таблицы
\d team_members

-- Количество записей
SELECT COUNT(*) FROM team_members;

-- Все сотрудники
SELECT first_name, last_name, position FROM team_members;
```

## ⏭️ Что дальше?

**Фаза 2: RAG система** (следующая)
- Загрузка документации
- ChromaDB для векторного поиска
- RAG Agent для вопросов о документации

**Фаза 4: Web Search**
- Интеграция Tavily
- Поиск актуальной информации

**Фаза 5: Router Agent**
- Умный выбор между RAG, SQL, Web Search
- Обработка комбинированных запросов

## 📚 Дополнительные материалы

- [SQL_AGENT_EXAMPLES.md](SQL_AGENT_EXAMPLES.md) - Больше примеров SQL запросов
- [README.md](README.md) - Полная документация проекта
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc документация

## 💡 Полезные советы

1. **Начните с простых вопросов** и постепенно усложняйте
2. **Изучите схему БД** в файле `data/sql/01_schema.sql`
3. **Смотрите на generated SQL** в ответах, чтобы понять как работает агент
4. **Используйте Swagger UI** для интерактивного тестирования
5. **Проверяйте логи** при возникновении проблем

---

**Готово к тестированию!** 🎉

Запустите Docker и попробуйте задать вопрос к базе данных!
