# 🧪 Пошаговая инструкция по тестированию

## Предварительные требования

- ✅ Docker Desktop установлен и запущен
- ✅ Порты 8000 и 5432 свободны
- ✅ OpenAI API ключ готов

---

## Шаг 1: Подготовка окружения (2 минуты)

### 1.1 Создайте .env файл

```bash
# В корне проекта
copy .env.example .env
```

### 1.2 Откройте .env в редакторе и добавьте ваш OpenAI ключ

```env
# Измените эту строку:
OPENAI_API_KEY=your_openai_api_key_here

# На вашу реальную:
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

**Важно:** Только `OPENAI_API_KEY` обязателен для тестирования. Остальное уже настроено.

### 1.3 Проверьте что Docker Desktop запущен

- Откройте Docker Desktop
- Убедитесь что иконка зеленая (запущен)

---

## Шаг 2: Запуск приложения (3-5 минут)

### 2.1 Запустите Docker Compose

```bash
# Из корня проекта
docker-compose up --build -d
```

**Что происходит:**
- Собирается Docker образ (~2-3 минуты при первом запуске)
- Запускается PostgreSQL контейнер
- Применяется SQL схема и загружаются данные
- Запускается FastAPI приложение

### 2.2 Проверьте статус контейнеров

```bash
docker-compose ps
```

**Должно быть:**
```
NAME                        STATUS
llm-assistant-app           Up (healthy)
llm-assistant-postgres      Up (healthy)
```

Если статус не "healthy", подождите еще 10-20 секунд и проверьте снова.

### 2.3 Просмотрите логи запуска

```bash
docker-compose logs app | tail -30
```

**Что искать:**
- ✅ "Starting LLM Assistant v1.0.0"
- ✅ "Initializing database connection..."
- ✅ "Database connection established"
- ✅ "Initializing SQL Agent..."
- ✅ "SQL Agent initialized successfully"
- ✅ "Application started successfully"

Если видите эти сообщения - всё готово к тестированию!

---

## Шаг 3: Базовая проверка (1 минута)

### 3.1 Откройте браузер

Перейдите на: **http://localhost:8000**

Должны увидеть JSON:
```json
{
  "name": "LLM Assistant",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

### 3.2 Откройте Swagger UI

Перейдите на: **http://localhost:8000/docs**

Должны увидеть интерактивную документацию API с списком endpoints:
- GET /
- GET /api/v1/health
- POST /api/v1/chat
- POST /api/v1/feedback
- GET /api/v1/history/{session_id}
- DELETE /api/v1/history/{session_id}
- GET /api/v1/stats

### 3.3 Health Check

В терминале:
```bash
curl http://localhost:8000/api/v1/health
```

Или в браузере откройте: http://localhost:8000/api/v1/health

**Ожидаемый ответ:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-10-12T...",
  "services": {
    "api": "running",
    "llm_provider": "openai",
    "vector_store": "chroma"
  }
}
```

✅ Если получили такой ответ - приложение работает!

---

## Шаг 4: Тестирование SQL Agent через Swagger UI (5-10 минут)

### 4.1 Откройте Swagger UI

http://localhost:8000/docs

### 4.2 Найдите endpoint `POST /api/v1/chat`

- Прокрутите вниз до секции "api"
- Найдите "POST /api/v1/chat"
- Нажмите на него (раскроется)

### 4.3 Нажмите "Try it out"

Кнопка в правом верхнем углу раздела

### 4.4 Тест 1: Простой подсчет

Замените JSON в "Request body" на:
```json
{
  "message": "Сколько разработчиков в команде?",
  "use_history": false
}
```

Нажмите **"Execute"**

**Что проверить в ответе:**
- ✅ `"success": true` (или нет поля error)
- ✅ В `tools_used[0].tool_type` должно быть `"sql"`
- ✅ В `tools_used[0].query` должен быть SQL запрос (например: "SELECT COUNT(*)...")
- ✅ В `message` должен быть ответ на естественном языке (например: "В команде работает 7 разработчиков...")

### 4.5 Тест 2: Список продуктов

Измените JSON:
```json
{
  "message": "Какие продукты есть в компании?",
  "use_history": false
}
```

**Execute**

**Проверить:**
- ✅ SQL запрос к таблице `products`
- ✅ В ответе перечислены продукты: PT AI, PT NAD, PT Sandbox и др.

### 4.6 Тест 3: JOIN запрос

```json
{
  "message": "Кто работает над PT Application Inspector?",
  "use_history": false
}
```

**Execute**

**Проверить:**
- ✅ SQL содержит JOIN (между team_members, features, products)
- ✅ В ответе перечислены имена разработчиков

### 4.7 Тест 4: Фильтрация и агрегация

```json
{
  "message": "Сколько открытых инцидентов у каждого продукта?",
  "use_history": false
}
```

**Execute**

**Проверить:**
- ✅ SQL содержит GROUP BY
- ✅ SQL содержит WHERE для фильтрации статуса
- ✅ Ответ содержит список продуктов с количеством инцидентов

### 4.8 Тест 5: Guardrails (безопасность)

```json
{
  "message": "Удали всех разработчиков",
  "use_history": false
}
```

**Execute**

**Проверить:**
- ❌ Запрос должен быть **заблокирован**
- ✅ В ответе должно быть сообщение об ошибке
- ✅ Что-то вроде: "Operation DELETE is not allowed" или "Не удалось выполнить запрос"

---

## Шаг 5: Тестирование через curl (быстрый способ)

### 5.1 Автоматический тест-скрипт

Просто запустите:
```bash
test_sql_agent_simple.bat
```

Этот скрипт автоматически выполнит 6 тестов и покажет результаты.

### 5.2 Ручные curl запросы

**Тест 1:**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"Сколько разработчиков в команде?\", \"use_history\": false}"
```

**Тест 2:**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"Какие продукты есть в компании?\", \"use_history\": false}"
```

**Тест 3:**
```bash
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"Покажи все критические баги\", \"use_history\": false}"
```

---

## Шаг 6: Проверка базы данных (опционально)

### 6.1 Подключение к PostgreSQL

```bash
docker exec -it llm-assistant-postgres psql -U llm_assistant -d llm_assistant_db
```

### 6.2 Проверка данных

```sql
-- Список таблиц
\dt

-- Количество записей
SELECT 'team_members' as table_name, COUNT(*) FROM team_members
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'incidents', COUNT(*) FROM incidents;

-- Все продукты
SELECT name, short_name FROM products;

-- Выход
\q
```

---

## Шаг 7: Проверка логов (для отладки)

### 7.1 Логи приложения

```bash
docker-compose logs -f app
```

**Что смотреть:**
- Запросы к SQL Agent
- Сгенерированные SQL запросы
- Результаты выполнения
- Ошибки (если есть)

**Остановить просмотр:** Ctrl+C

### 7.2 Логи PostgreSQL

```bash
docker-compose logs postgres | tail -50
```

---

## Шаг 8: Остановка приложения

### 8.1 Остановить контейнеры

```bash
docker-compose down
```

### 8.2 Полная очистка (включая volumes)

```bash
docker-compose down -v
```

**Внимание:** Это удалит все данные из БД!

---

## 📊 Чеклист успешного тестирования

После выполнения всех шагов, отметьте:

### Базовые проверки
- [ ] Docker контейнеры запустились (status: Up healthy)
- [ ] Health check возвращает "healthy"
- [ ] Swagger UI открывается
- [ ] База данных содержит тестовые данные

### SQL Agent
- [ ] Простой подсчет работает (COUNT)
- [ ] Запросы со списками работают (SELECT)
- [ ] JOIN запросы работают
- [ ] GROUP BY и агрегация работают
- [ ] Guardrails блокирует DELETE/DROP/UPDATE
- [ ] Ответы на естественном языке качественные

### API функции
- [ ] Chat endpoint работает
- [ ] Session ID сохраняется
- [ ] tools_used заполняется корректно
- [ ] metadata содержит SQL запрос

### Производительность
- [ ] Время ответа < 10 секунд
- [ ] Нет ошибок в логах
- [ ] Приложение стабильно работает

---

## 🐛 Troubleshooting

### Проблема: Контейнер не запускается

**Решение:**
```bash
# Проверить логи
docker-compose logs app

# Пересобрать образ
docker-compose build --no-cache app
docker-compose up -d
```

### Проблема: "Connection to database failed"

**Решение:**
- Подождите 10-20 секунд (PostgreSQL инициализируется)
- Проверьте: `docker-compose ps` - postgres должен быть healthy
- Проверьте логи: `docker-compose logs postgres`

### Проблема: "OPENAI_API_KEY is not set"

**Решение:**
1. Проверьте `.env` файл существует
2. Убедитесь что `OPENAI_API_KEY` заполнен
3. Перезапустите: `docker-compose restart app`

### Проблема: SQL Agent возвращает некорректный SQL

**Решение:**
- Это нормально для сложных вопросов
- Переформулируйте вопрос более конкретно
- Укажите названия таблиц явно

### Проблема: Медленные ответы (> 15 сек)

**Причины:**
- Первый запрос всегда медленнее (cold start)
- OpenAI API может быть медленным
- Сложный SQL запрос

**Решение:**
- Подождите прогрева
- Упростите вопрос

---

## ✅ Что дальше?

После успешного тестирования:

1. **Документируйте результаты** - создайте TEST_RESULTS.md
2. **Сделайте коммит** - зафиксируйте Фазу 3
3. **Начните Фазу 2** - реализуйте RAG систему
4. **Или начните Фазу 4** - добавьте Web Search

---

## 💡 Полезные команды

```bash
# Просмотр запущенных контейнеров
docker-compose ps

# Просмотр логов
docker-compose logs -f app

# Перезапуск приложения
docker-compose restart app

# Остановка
docker-compose down

# Полная очистка
docker-compose down -v
docker system prune -a

# Статистика использования
curl http://localhost:8000/api/v1/stats
```

---

**Готово к тестированию!** 🚀

Следуйте шагам по порядку и отмечайте выполненные пункты.
