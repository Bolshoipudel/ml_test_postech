# LLM Assistant - Многофункциональный AI Ассистент

> Тестовое задание для стажировки в ML команду Positive Technologies

Многофункциональный LLM-ассистент с возможностями RAG, SQL-запросов и веб-поиска для продуктов Positive Technologies.

## 🎯 Описание проекта

Бэкенд-сервис интеллектуального ассистента с автоматическим выбором инструментов для ответа на вопросы:
- 🤖 **Router Agent** - автоматически классифицирует запросы и выбирает подходящий инструмент
- 📚 **RAG (Retrieval-Augmented Generation)** - отвечает на вопросы на основе документации продуктов PT
- 🗄️ **SQL Agent** - извлекает информацию из базы данных командных проектов
- 🌐 **Web Search Agent** - ищет актуальную информацию в интернете
- 🎼 **Orchestrator** - координирует работу нескольких агентов одновременно

## 📊 Текущий статус: **ВСЕ ФАЗЫ РЕАЛИЗОВАНЫ** ✅

### ✅ Реализованные компоненты

#### 🏗️ Базовая инфраструктура
- ✅ FastAPI приложение с полной настройкой
- ✅ Pydantic Settings для конфигурации через .env
- ✅ Структурированное логирование через loguru
- ✅ Обработка ошибок и CORS middleware
- ✅ Health check и мониторинг
- ✅ Асинхронная обработка запросов

#### 🔌 API Endpoints
```
POST   /api/v1/chat              - Основной endpoint для общения с ассистентом
GET    /api/v1/health            - Проверка здоровья сервиса
POST   /api/v1/feedback          - Отправка обратной связи
GET    /api/v1/history/{id}      - Получение истории чата
DELETE /api/v1/history/{id}      - Очистка истории сессии
GET    /api/v1/stats             - Статистика использования
GET    /api/v1/rag/stats         - Статистика RAG коллекции
POST   /api/v1/rag/reload        - Перезагрузка документов
GET    /api/v1/rag/debug/{query} - Отладка RAG поиска
```

#### 🤖 Router Agent (Интеллектуальная маршрутизация)
- ✅ Классификация запросов с использованием LLM
- ✅ Определение подходящего инструмента (SQL/RAG/WEB_SEARCH/MULTIPLE)
- ✅ Few-shot примеры для улучшения точности
- ✅ Оценка уверенности (confidence score)
- ✅ Fallback логика при ошибках
- ✅ Поддержка комбинированных запросов (MULTIPLE)

**Примеры маршрутизации:**
```python
"Сколько разработчиков работает над PT AI?"        → SQL
"Что такое PT Sandbox?"                            → RAG
"Последние новости по кибербезопасности"           → WEB_SEARCH
"Сколько человек в команде PT AI и какие функции?" → SQL + RAG
```

#### 📚 RAG Agent (Документация)
- ✅ ChromaDB для векторного хранилища
- ✅ OpenAI Embeddings / Sentence Transformers
- ✅ Загрузка и индексация документов (Markdown, TXT)
- ✅ Chunking с умным разбиением по параграфам
- ✅ Semantic search с оценкой релевантности
- ✅ Генерация ответов на основе контекста
- ✅ Фильтрация по минимальному порогу релевантности
- ✅ Перезагрузка документов без перезапуска

#### 🗄️ SQL Agent (База данных)
- ✅ Natural Language to SQL преобразование
- ✅ Автоматическая генерация SQL из вопросов
- ✅ Guardrails для безопасности (только SELECT)
- ✅ Валидация SQL запросов
- ✅ Форматирование результатов в естественный язык
- ✅ Защита от SQL injection
- ✅ Схема БД с продуктами PT

**База данных содержит:**
- `departments` - 6 отделов
- `products` - 5 продуктов (PT AI, PT NAD, PT Sandbox, PT MS, PT ISIM)
- `team_members` - 12 разработчиков
- `features` - 8 фич в разработке
- `incidents` - 8 инцидентов/багов

#### 🌐 Web Search Agent
- ✅ Интеграция с Tavily API
- ✅ Semantic search в интернете
- ✅ Поиск новостей за N дней
- ✅ Генерация ответов на основе результатов поиска
- ✅ Извлечение источников и дат публикации
- ✅ Фильтрация по релевантности

#### 🎼 Orchestrator Service
- ✅ Координация работы всех агентов
- ✅ Параллельное выполнение нескольких агентов
- ✅ Агрегация результатов от разных источников
- ✅ Синтез финального ответа с использованием LLM
- ✅ Обработка ошибок с fallback стратегиями
- ✅ История разговоров

#### 🔧 LLM Factory (Поддержка провайдеров)
- ✅ OpenAI API (GPT-4, GPT-3.5)
- ✅ Ollama для локальных моделей
- ✅ Гибкое переключение провайдеров
- ✅ Настройка temperature под задачу
- ✅ Управление через переменные окружения

#### 🧪 Тестирование
- ✅ Unit тесты для API
- ✅ Тесты агентов (SQL, RAG, Web Search, Router)
- ✅ Pytest fixtures и конфигурация
- ✅ Интеграционные тесты
- ✅ Ручное тестирование (test_router_manual.py)

#### 🐳 Docker Infrastructure
- ✅ Multi-stage Dockerfile для оптимизации
- ✅ docker-compose.yml с PostgreSQL
- ✅ Автоматическая инициализация БД
- ✅ Health checks для всех сервисов
- ✅ Volume для персистентности данных

#### 🌍 Локализация
- ✅ **Все комментарии и docstrings переведены на русский язык**
- ✅ **Удалены избыточные и очевидные документ-строки**
- ✅ **Оставлена только информативная документация**
- ✅ Логи оставлены на английском (стандартная практика)
- ✅ Промпты для LLM на языках оригинала

## 🚀 Быстрый старт

### Требования
- Python 3.11+
- Docker & Docker Compose (рекомендуется)
- PostgreSQL (если без Docker)
- OpenAI API Key
- Tavily API Key (опционально, для веб-поиска)

### Установка (Docker) - Рекомендуется

1. **Клонируйте репозиторий**
```bash
git clone <your-repo-url>
cd ml_test_postech
```

2. **Создайте .env файл**
```bash
# Windows
copy .env.example .env
# Linux/Mac
cp .env.example .env
```

3. **Настройте переменные окружения**
Отредактируйте `.env` и добавьте ваши API ключи:
```env
# LLM Provider
OPENAI_API_KEY=sk-your_openai_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1

# Web Search (опционально)
TAVILY_API_KEY=tvly-your_tavily_key_here

# Database
POSTGRES_USER=llm_assistant
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=llm_assistant_db
```

4. **Запустите с Docker Compose**
```bash
docker-compose up --build -d
```

5. **Проверьте статус**
```bash
docker-compose ps
docker-compose logs -f app
```

6. **Доступ к API**
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

7. **Остановка**
```bash
# Windows
stop.bat

# Linux/Mac
docker-compose down
```

### Установка (локально)

1. **Создайте виртуальное окружение**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Установите зависимости**
```bash
pip install -r requirements.txt
```

3. **Настройте PostgreSQL**
```bash
# Создайте БД
createdb llm_assistant_db

# Инициализируйте схему
psql -d llm_assistant_db -f data/sql/init.sql
```

4. **Создайте .env и настройте переменные**
```bash
copy .env.example .env
```

5. **Запустите приложение**
```bash
# Windows
run.bat

# Linux/Mac
python -m uvicorn app.main:app --reload
```

## 📖 Примеры использования

### 1. Вопрос о документации (RAG)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Что такое PT Application Inspector и какие у него возможности?"
  }'
```

**Ответ:**
```json
{
  "message": "PT Application Inspector (PT AI) - это статический анализатор кода...",
  "tools_used": [
    {
      "tool_type": "router",
      "reasoning": "Вопрос о возможностях продукта требует поиска в документации",
      "confidence": 0.9
    },
    {
      "tool_type": "rag",
      "result_summary": "Найдено 5 релевантных документов"
    }
  ],
  "sources": ["documentation: PT_AI.md", "documentation: PT_Products.md"]
}
```

### 2. SQL запрос к базе данных
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Сколько разработчиков работает над PT Sandbox?"
  }'
```

**Ответ:**
```json
{
  "message": "Над продуктом PT Sandbox работает 3 разработчика...",
  "tools_used": [
    {
      "tool_type": "router",
      "reasoning": "Вопрос о количестве разработчиков требует запроса к БД",
      "confidence": 0.95
    },
    {
      "tool_type": "sql",
      "query": "SELECT COUNT(*) FROM team_members...",
      "result_summary": "Найдено 3 записи"
    }
  ],
  "sources": ["database: PostgreSQL"]
}
```

### 3. Веб-поиск
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Какие последние новости по кибербезопасности?"
  }'
```

### 4. Комбинированный запрос (несколько агентов)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Сколько человек в команде PT AI и какие у него функции?"
  }'
```

**Router выберет:** SQL + RAG → Orchestrator объединит результаты

## 🏗️ Архитектура системы

```
                                 ┌─────────────┐
                                 │   Client    │
                                 └──────┬──────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │       FastAPI Application      │
                        │      (app/main.py)             │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │    Orchestrator Service        │
                        │  (Координация агентов)         │
                        └───────────────┬────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │      Router Agent              │
                        │  (Классификация запросов)      │
                        │  • Few-shot примеры            │
                        │  • Confidence scoring          │
                        │  • Fallback логика             │
                        └───────────────┬────────────────┘
                                        │
                    ┌───────────────────┼─────────────────┐
                    │                   │                 │
                    ▼                   ▼                 ▼
        ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
        │   RAG Agent      │  │  SQL Agent   │  │ Web Search   │
        │                  │  │              │  │   Agent      │
        │ • ChromaDB       │  │ • NL→SQL     │  │ • Tavily API │
        │ • Embeddings     │  │ • Guardrails │  │ • News       │
        │ • Similarity     │  │ • Validation │  │ • Filtering  │
        └────────┬─────────┘  └──────┬───────┘  └──────┬───────┘
                 │                   │                  │
                 ▼                   ▼                  ▼
        ┌──────────────┐    ┌──────────────┐  ┌──────────────┐
        │   ChromaDB   │    │  PostgreSQL  │  │    Tavily    │
        │ (Документы)  │    │  (Команда)   │  │  (Интернет)  │
        └──────────────┘    └──────────────┘  └──────────────┘
```

### Поток обработки запроса

1. **Client** → отправляет вопрос
2. **API Layer** → валидация и сохранение в истории
3. **Orchestrator** → координация процесса
4. **Router Agent** → классификация запроса и выбор инструмента(ов)
5. **Agent(s)** → выполнение задачи (RAG/SQL/Web Search или их комбинация)
6. **Orchestrator** → агрегация результатов от нескольких агентов (если нужно)
7. **API Layer** → формирование ответа с метаданными
8. **Client** ← получает ответ с источниками

## 🗂️ Структура проекта

```
ml_test_postech/
├── app/                           # Основное приложение
│   ├── agents/                    # Агенты для разных задач
│   │   ├── router_agent.py       # ✅ Маршрутизатор запросов
│   │   ├── rag_agent.py          # ✅ RAG для документации
│   │   ├── sql_agent.py          # ✅ NL→SQL агент
│   │   └── web_search_agent.py   # ✅ Веб-поиск
│   ├── api/                       # API endpoints
│   │   └── routes.py             # ✅ REST API
│   ├── models/                    # Модели данных
│   │   ├── database.py           # ✅ SQLAlchemy модели
│   │   └── schemas.py            # ✅ Pydantic схемы
│   ├── services/                  # Бизнес-логика
│   │   ├── orchestrator_service.py  # ✅ Координация агентов
│   │   ├── rag_service.py          # ✅ Векторное хранилище
│   │   ├── database_service.py     # ✅ Работа с БД
│   │   ├── search_service.py       # ✅ Веб-поиск
│   │   └── llm_factory.py          # ✅ Управление LLM
│   ├── prompts/                   # Промпты для LLM
│   │   └── router_prompts.py     # ✅ Промпты роутера
│   ├── utils/                     # Утилиты
│   │   └── document_loader.py    # ✅ Загрузка документов
│   ├── config.py                  # ✅ Конфигурация
│   └── main.py                    # ✅ Точка входа
├── data/                          # Данные
│   ├── docs/                      # Документация для RAG
│   │   ├── PT_AI.md              # PT Application Inspector
│   │   ├── PT_Sandbox.md         # PT Sandbox
│   │   └── ...                   # Другие продукты
│   └── sql/                       # SQL скрипты
│       └── init.sql              # ✅ Инициализация БД
├── tests/                         # Тесты
│   ├── test_api.py               # ✅ Тесты API
│   ├── test_sql_agent.py         # ✅ Тесты SQL агента
│   ├── test_rag_agent.py         # ✅ Тесты RAG агента
│   ├── test_web_search_agent.py  # ✅ Тесты веб-поиска
│   ├── test_router_agent.py      # ✅ Тесты роутера
│   └── conftest.py               # Fixtures
├── chroma_db/                     # ChromaDB персистентность
├── docker-compose.yml             # ✅ Docker оркестрация
├── Dockerfile                     # ✅ Docker образ
├── requirements.txt               # ✅ Зависимости
├── .env.example                   # Пример конфигурации
├── pytest.ini                     # Настройки pytest
└── README.md                      # Этот файл
```

## 🔧 Технологический стек

### Backend & API
- **FastAPI** 0.104+ - Современный async веб-фреймворк
- **Pydantic** 2.4+ - Валидация данных и настройки
- **Uvicorn** - ASGI сервер

### LLM & AI
- **LangChain** 0.1+ - Фреймворк для LLM приложений
- **OpenAI API** - GPT-4 / GPT-3.5-turbo
- **Ollama** - Локальные LLM модели (опционально)

### Vector Database & Embeddings
- **ChromaDB** 0.4+ - Векторное хранилище
- **OpenAI Embeddings** - text-embedding-3-small
- **Sentence Transformers** - Локальные эмбеддинги (опционально)

### Relational Database
- **PostgreSQL** 15+ - Основная БД
- **SQLAlchemy** 2.0+ - ORM
- **Psycopg2** - PostgreSQL адаптер

### Web Search
- **Tavily** - Semantic search API
- **httpx** - HTTP клиент

### Testing
- **Pytest** - Фреймворк тестирования
- **pytest-asyncio** - Асинхронные тесты
- **pytest-cov** - Coverage отчеты

### DevOps
- **Docker** - Контейнеризация
- **Docker Compose** - Оркестрация сервисов

### Utilities
- **Loguru** - Структурированное логирование
- **Python-dotenv** - Управление переменными окружения

## 🧪 Тестирование

### Запуск всех тестов
```bash
pytest tests/ -v
```

### Запуск конкретного теста
```bash
pytest tests/test_sql_agent.py -v
pytest tests/test_router_agent.py -v
pytest tests/test_rag_agent.py -v
```

### С coverage отчётом
```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Ручное тестирование Router Agent
```bash
python test_router_manual.py
```

## 🎛️ Конфигурация

Все настройки управляются через переменные окружения в `.env`:

```env
# Приложение
APP_NAME=LLM Assistant
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000

# LLM Provider (openai или ollama)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
OPENAI_API_KEY=sk-your_key_here

# Ollama (если используете)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Embeddings (openai или sentence-transformers)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Vector Store
VECTOR_STORE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Database
POSTGRES_USER=llm_assistant
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=llm_assistant_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Web Search
TAVILY_API_KEY=tvly-your_key_here
WEB_SEARCH_ENABLED=true
MAX_SEARCH_RESULTS=5

# Безопасность
ENABLE_GUARDRAILS=true
ALLOWED_SQL_OPERATIONS=SELECT
```

## 📊 Мониторинг и отладка

### Проверка здоровья сервиса
```bash
curl http://localhost:8000/api/v1/health
```

### Статистика использования
```bash
curl http://localhost:8000/api/v1/stats
```

### Статистика RAG коллекции
```bash
curl http://localhost:8000/api/v1/rag/stats
```

### Отладка RAG поиска
```bash
curl http://localhost:8000/api/v1/rag/debug/PT%20Sandbox
```

### Логи Docker
```bash
docker-compose logs -f app
docker-compose logs -f db
```

## 🎯 Примеры сценариев использования

### Сценарий 1: Информация о продукте
**Вопрос:** "Расскажи о PT Network Attack Discovery"
- **Router:** RAG Agent
- **Источник:** Документация
- **Результат:** Описание продукта, возможности, требования

### Сценарий 2: Статистика команды
**Вопрос:** "Сколько открытых инцидентов по PT Sandbox?"
- **Router:** SQL Agent
- **Источник:** PostgreSQL
- **Результат:** Количество с детализацией

### Сценарий 3: Актуальная информация
**Вопрос:** "Последние новости по Application Security"
- **Router:** Web Search Agent
- **Источник:** Tavily (интернет)
- **Результат:** Сводка новостей за 7 дней

### Сценарий 4: Комплексный запрос
**Вопрос:** "Сколько человек работает над PT AI и какие у него основные возможности?"
- **Router:** MULTIPLE (SQL + RAG)
- **Источники:** PostgreSQL + Документация
- **Orchestrator:** Синтезирует единый ответ
- **Результат:** "Над PT AI работает 3 разработчика. Основные возможности: SAST анализ, поддержка 30+ языков..."

## 🔐 Безопасность

### Реализованные меры:
- ✅ **SQL Guardrails** - только SELECT запросы разрешены
- ✅ **SQL Injection защита** - валидация и проверка запросов
- ✅ **Запрет опасных команд** - DROP, DELETE, UPDATE заблокированы
- ✅ **Проверка паттернов** - блокировка xp_, sp_ и других опасных паттернов
- ✅ **CORS настройка** - настраиваемые политики
- ✅ **Переменные окружения** - чувствительные данные не в коде

## 🚀 Производительность

### Оптимизации:
- Асинхронная обработка запросов (FastAPI + asyncio)
- Параллельное выполнение нескольких агентов
- Кеширование настроек (lru_cache)
- Connection pooling для БД
- Batch обработка в ChromaDB
- Персистентность векторного хранилища

## 📝 Roadmap

### ✅ Реализовано
- [x] Базовая инфраструктура
- [x] API endpoints
- [x] RAG система с ChromaDB
- [x] SQL Agent с guardrails
- [x] Web Search Agent
- [x] Router Agent
- [x] Orchestrator для координации
- [x] LLM Factory (OpenAI + Ollama)
- [x] Docker инфраструктура
- [x] Тесты всех компонентов
- [x] Локализация документации на русский

### 🔮 Возможные улучшения
- [ ] DeepEval для автоматической оценки качества
- [ ] Streamlit UI для демонстрации
- [ ] Redis для кеширования и истории
- [ ] Мониторинг с Prometheus/Grafana
- [ ] CI/CD pipeline
- [ ] Speech-to-Text интеграция
- [ ] Multi-turn диалоги с контекстом
- [ ] Fine-tuned Router модель

## 🤝 Разработка

### Стиль кода
- Следуем PEP 8
- Docstrings на русском языке
- Логи на английском
- Type hints везде где возможно

### Добавление нового агента
1. Создайте файл в `app/agents/your_agent.py`
2. Реализуйте класс с методами `initialize()` и основным методом
3. Добавьте в Router промпты и логику классификации
4. Добавьте в Orchestrator обработку нового типа
5. Напишите тесты в `tests/test_your_agent.py`

## 📚 Полезные ссылки

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Tavily API](https://tavily.com/)

## 📄 Лицензия

MIT License

## 👤 Автор

Орлов Григорий

---

**Статус проекта:** Production Ready ✅

Все основные компоненты реализованы, протестированы и готовы к использованию.
