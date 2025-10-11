# LLM Assistant - Multi-functional AI Assistant

> Тестовое задание для стажировки в ML команду Positive Technologies

Многофункциональный LLM-ассистент с возможностями RAG, SQL-запросов и веб-поиска для продуктов Positive Technologies.

## 🎯 Описание проекта

Бэкенд-сервис интеллектуального ассистента, способного:
- 📚 **RAG (Retrieval-Augmented Generation)** - отвечать на вопросы на основе документации
- 🗄️ **SQL Agent** - извлекать информацию из базы данных командных проектов
- 🌐 **Web Search** - искать актуальную информацию в интернете
- 🤖 **Router Agent** - автоматически выбирать подходящий инструмент для каждого запроса

## 📊 Текущий статус: **Фаза 1 завершена (Инфраструктура)** ✅

### Реализовано

#### ✅ Базовая инфраструктура
- FastAPI приложение с полной настройкой
- Pydantic Settings для конфигурации
- Логирование через loguru
- Обработка ошибок и middleware
- Health check endpoint

#### ✅ API Endpoints
- `POST /api/v1/chat` - основной endpoint для чата
- `GET /api/v1/health` - проверка здоровья сервиса
- `POST /api/v1/feedback` - отправка обратной связи
- `GET /api/v1/history/{session_id}` - получение истории чата
- `DELETE /api/v1/history/{session_id}` - очистка истории
- `GET /api/v1/stats` - статистика использования

#### ✅ База данных PostgreSQL
Создана схема для команды продуктов PT:
- `departments` - отделы (6 записей)
- `products` - продукты безопасности (5 продуктов: PT AI, PT NAD, PT Sandbox, PT MS, PT ISIM)
- `team_members` - члены команды (12 человек)
- `features` - функции в разработке (8 фич)
- `incidents` - инциденты и баги (8 инцидентов)

#### ✅ Docker инфраструктура
- Dockerfile для FastAPI приложения
- docker-compose.yml с PostgreSQL
- Автоматическая инициализация БД
- Health checks для всех сервисов

#### ✅ Тесты
- Базовые тесты API endpoints
- Pytest конфигурация
- Test fixtures

### 🚧 В разработке (Следующие шаги)

#### Фаза 2: RAG система
- [ ] Загрузка и индексация документации
- [ ] Vector store (ChromaDB)
- [ ] RAG сервис и агент
- [ ] Интеграция с chat endpoint

#### Фаза 3: SQL Agent
- [ ] Подключение к PostgreSQL
- [ ] NL-to-SQL преобразование
- [ ] Guardrails (только SELECT)
- [ ] SQL агент

#### Фаза 4: Web Search Agent
- [ ] Интеграция Tavily/DuckDuckGo
- [ ] Web search сервис
- [ ] Web search агент

#### Фаза 5: Router Agent (ЯДРО)
- [ ] Классификация запросов
- [ ] Выбор инструментов
- [ ] Оркестрация агентов
- [ ] Агрегация результатов

#### Фаза 6: Локальные LLM
- [ ] LLM Factory
- [ ] Поддержка OpenAI
- [ ] Поддержка Ollama
- [ ] Переключение провайдеров

#### Фаза 7: Оценка качества
- [ ] Датасет для тестирования
- [ ] Интеграция DeepEval
- [ ] Метрики качества
- [ ] Отчет с результатами

## 🚀 Быстрый старт

### Требования
- Python 3.11+
- Docker & Docker Compose (опционально)
- PostgreSQL (если без Docker)

### Установка (локально)

1. **Клонируйте репозиторий**
```bash
git clone <your-repo-url>
cd ml_test_postech
```

2. **Создайте виртуальное окружение**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Установите зависимости**
```bash
pip install -r requirements.txt
```

4. **Настройте переменные окружения**
```bash
# Windows
copy .env.example .env
# Linux/Mac
cp .env.example .env
```

Отредактируйте `.env` и добавьте ваши API ключи:
```env
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here  # Опционально для web search
```

5. **Запустите приложение**
```bash
# Windows
run.bat
# Linux/Mac
./run.sh
```

API будет доступен по адресу: http://localhost:8000

### Установка (Docker)

1. **Создайте .env файл**
```bash
copy .env.example .env  # Windows
```

2. **Запустите с Docker Compose**
```bash
# Windows
docker-run.bat
# Linux/Mac
docker-compose up --build -d
```

3. **Проверьте статус**
```bash
docker-compose ps
docker-compose logs -f app
```

4. **Остановка**
```bash
# Windows
stop.bat
# Linux/Mac
docker-compose down
```

## 📖 Документация API

После запуска приложения, документация доступна по адресам:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Примеры запросов

#### 1. Health Check
```bash
curl http://localhost:8000/api/v1/health
```

#### 2. Отправить сообщение в чат
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Сколько человек работает над PT Application Inspector?",
    "use_history": true
  }'
```

#### 3. Получить историю
```bash
curl http://localhost:8000/api/v1/history/{session_id}
```

#### 4. Отправить feedback
```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test123",
    "rating": 5,
    "comment": "Отличный ответ!"
  }'
```

## 🏗️ Архитектура

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI App                 │
│  ┌─────────────────────────────┐   │
│  │     Router Agent            │   │  ◄── Выбирает нужный инструмент
│  └────────┬────────────────────┘   │
│           │                         │
│     ┌─────┴─────┬─────────────┐   │
│     ▼           ▼             ▼   │
│  ┌────┐     ┌─────┐      ┌──────┐ │
│  │RAG │     │ SQL │      │ Web  │ │
│  │Agent│    │Agent│      │Search│ │
│  └──┬─┘    └──┬──┘      └──┬───┘ │
└─────┼────────┼─────────────┼─────┘
      │        │             │
      ▼        ▼             ▼
┌──────────┐ ┌──────────┐ ┌─────────┐
│ChromaDB/ │ │PostgreSQL│ │ Tavily/ │
│  FAISS   │ │          │ │DuckDuck │
└──────────┘ └──────────┘ └─────────┘
```

### Компоненты

1. **API Layer** (`app/api/routes.py`)
   - REST API endpoints
   - Валидация запросов
   - История чатов

2. **Agents** (`app/agents/`)
   - Router Agent - выбор инструмента
   - RAG Agent - работа с документацией
   - SQL Agent - работа с БД
   - Web Search Agent - поиск в интернете

3. **Services** (`app/services/`)
   - RAG Service - векторное хранилище
   - Database Service - SQL запросы
   - Search Service - веб-поиск
   - LLM Factory - управление LLM провайдерами

4. **Tools** (`app/tools/`)
   - Вспомогательные утилиты для агентов

## 🧪 Тестирование

### Запуск тестов
```bash
pytest tests/ -v
```

### Запуск с coverage
```bash
pytest tests/ -v --cov=app --cov-report=html
```

## 📊 База данных

### Схема
База данных содержит информацию о команде продуктов Positive Technologies:

- **departments** - отделы компании
- **products** - продукты безопасности
- **team_members** - сотрудники с навыками
- **features** - функции в разработке
- **incidents** - баги и инциденты

### Примеры SQL запросов
```sql
-- Сколько человек работает над каждым продуктом
SELECT p.name, COUNT(DISTINCT f.assigned_to) as developers
FROM products p
JOIN features f ON p.id = f.product_id
GROUP BY p.name;

-- Активные инциденты по продуктам
SELECT p.name, COUNT(*) as open_incidents
FROM products p
JOIN incidents i ON p.id = i.product_id
WHERE i.status IN ('open', 'in_progress')
GROUP BY p.name;
```

## 🔧 Технологический стек

### Backend
- **FastAPI** - современный веб-фреймворк
- **Pydantic** - валидация данных
- **SQLAlchemy** - ORM для работы с БД
- **Uvicorn** - ASGI сервер

### LLM & AI
- **LangChain** - фреймворк для LLM приложений
- **OpenAI API** - GPT модели
- **ChromaDB** - векторное хранилище
- **Sentence Transformers** - embeddings

### Database
- **PostgreSQL** - основная БД
- **Alembic** - миграции

### Search
- **Tavily** - веб-поиск API
- **DuckDuckGo** - альтернативный поиск

### Testing & Quality
- **Pytest** - тестирование
- **DeepEval** - оценка качества LLM
- **Black** - форматирование кода
- **Flake8** - линтинг

### DevOps
- **Docker** - контейнеризация
- **Docker Compose** - оркестрация

## 📝 Roadmap

- [x] **Фаза 1**: Базовая инфраструктура ✅
- [ ] **Фаза 2**: RAG система
- [ ] **Фаза 3**: SQL Agent
- [ ] **Фаза 4**: Web Search Agent
- [ ] **Фаза 5**: Router Agent (ядро)
- [ ] **Фаза 6**: Поддержка локальных LLM
- [ ] **Фаза 7**: Оценка качества

### Опциональные фичи
- [ ] Deep Search для аналитики
- [ ] Guardrails для безопасности
- [ ] Streamlit UI
- [ ] Speech-to-Text интеграция

## 🤝 Разработка

### Структура проекта
```
ml_test_postech/
├── app/                    # Основное приложение
│   ├── agents/            # LLM агенты
│   ├── api/               # API endpoints
│   ├── models/            # Pydantic модели
│   ├── services/          # Бизнес-логика
│   ├── tools/             # Вспомогательные инструменты
│   ├── utils/             # Утилиты
│   ├── config.py          # Конфигурация
│   └── main.py            # Точка входа
├── data/                   # Данные
│   ├── docs/              # Документация для RAG
│   └── sql/               # SQL скрипты
├── tests/                  # Тесты
├── evaluation/             # Оценка качества
├── docker-compose.yml      # Docker конфигурация
├── requirements.txt        # Python зависимости
└── README.md              # Этот файл
```

### Команды разработки
```bash
# Установка зависимостей
make install

# Запуск приложения
make run

# Запуск тестов
make test

# Форматирование кода
make format

# Линтинг
make lint

# Docker команды
make docker-build
make docker-up
make docker-down
make logs
```

## 📄 Лицензия

MIT License

## 👤 Автор

Тестовое задание для стажировки в ML команду Positive Technologies

---

**Статус**: 🚧 В разработке | **Фаза**: 1/7 завершена | **Дата**: Октябрь 2024
