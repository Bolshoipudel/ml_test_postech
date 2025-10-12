# SQL Agent - Примеры использования

## Описание
SQL Agent автоматически преобразует вопросы на естественном языке в SQL запросы и выполняет их к базе данных PostgreSQL.

## Безопасность (Guardrails)
- ✅ Разрешены только SELECT запросы
- ❌ Блокируются DELETE, DROP, UPDATE, INSERT
- ❌ Блокируются опасные команды (EXEC, xp_, sp_, GRANT, REVOKE)

## Примеры вопросов

### 1. Вопросы о команде

**Вопрос:** "Сколько разработчиков работает в команде?"
```sql
-- Ожидаемый SQL:
SELECT COUNT(*) as total_developers
FROM team_members
WHERE position LIKE '%Developer%' AND is_active = true;
```

**Вопрос:** "Кто работает над PT Application Inspector?"
```sql
-- Ожидаемый SQL:
SELECT DISTINCT tm.first_name, tm.last_name, tm.position
FROM team_members tm
JOIN features f ON tm.id = f.assigned_to
JOIN products p ON f.product_id = p.id
WHERE p.name = 'PT Application Inspector';
```

**Вопрос:** "Покажи всех сотрудников из отдела разработки с их навыками"
```sql
-- Ожидаемый SQL:
SELECT tm.first_name, tm.last_name, tm.position, tm.skills, tm.experience_years
FROM team_members tm
JOIN departments d ON tm.department_id = d.id
WHERE d.name = 'Development' AND tm.is_active = true;
```

### 2. Вопросы о продуктах

**Вопрос:** "Какие продукты есть в компании?"
```sql
-- Ожидаемый SQL:
SELECT name, short_name, description, status, version
FROM products
ORDER BY name;
```

**Вопрос:** "Сколько инцидентов открыто для каждого продукта?"
```sql
-- Ожидаемый SQL:
SELECT p.name, p.short_name, COUNT(i.id) as open_incidents
FROM products p
LEFT JOIN incidents i ON p.id = i.product_id AND i.status IN ('open', 'in_progress')
GROUP BY p.id, p.name, p.short_name
ORDER BY open_incidents DESC;
```

### 3. Вопросы об инцидентах

**Вопрос:** "Какие критические баги сейчас открыты?"
```sql
-- Ожидаемый SQL:
SELECT i.title, i.description, p.name as product,
       tm.first_name || ' ' || tm.last_name as assigned_to,
       i.reported_date
FROM incidents i
JOIN products p ON i.product_id = p.id
LEFT JOIN team_members tm ON i.assigned_to = tm.id
WHERE i.severity = 'critical' AND i.status != 'resolved'
ORDER BY i.reported_date DESC;
```

**Вопрос:** "Сколько багов было закрыто в феврале 2024?"
```sql
-- Ожидаемый SQL:
SELECT COUNT(*) as resolved_bugs
FROM incidents
WHERE status = 'resolved'
  AND resolved_date >= '2024-02-01'
  AND resolved_date < '2024-03-01';
```

### 4. Вопросы о фичах

**Вопрос:** "Какие функции сейчас в разработке?"
```sql
-- Ожидаемый SQL:
SELECT f.title, f.description, p.name as product, f.priority,
       tm.first_name || ' ' || tm.last_name as assigned_to,
       f.target_date
FROM features f
JOIN products p ON f.product_id = p.id
JOIN team_members tm ON f.assigned_to = tm.id
WHERE f.status = 'in_development'
ORDER BY f.priority, f.target_date;
```

**Вопрос:** "Кто работает над фичами с высоким приоритетом?"
```sql
-- Ожидаемый SQL:
SELECT DISTINCT tm.first_name, tm.last_name, tm.position,
       COUNT(f.id) as high_priority_features
FROM team_members tm
JOIN features f ON tm.id = f.assigned_to
WHERE f.priority = 'high' AND f.status != 'completed'
GROUP BY tm.id, tm.first_name, tm.last_name, tm.position
ORDER BY high_priority_features DESC;
```

### 5. Сложные аналитические вопросы

**Вопрос:** "Какая средняя загрузка разработчиков по часам?"
```sql
-- Ожидаемый SQL:
SELECT tm.first_name, tm.last_name,
       SUM(f.estimated_hours) as total_estimated,
       SUM(f.completed_hours) as total_completed,
       ROUND(AVG(f.completed_hours::float / NULLIF(f.estimated_hours, 0) * 100), 2) as avg_progress_pct
FROM team_members tm
JOIN features f ON tm.id = f.assigned_to
WHERE f.status IN ('in_development', 'planning')
GROUP BY tm.id, tm.first_name, tm.last_name
ORDER BY total_estimated DESC;
```

**Вопрос:** "Какой отдел решает больше всего багов?"
```sql
-- Ожидаемый SQL:
SELECT d.name as department,
       COUNT(DISTINCT tm.id) as team_size,
       COUNT(i.id) as resolved_incidents,
       ROUND(COUNT(i.id)::float / COUNT(DISTINCT tm.id), 2) as bugs_per_person
FROM departments d
JOIN team_members tm ON d.id = tm.department_id
LEFT JOIN incidents i ON tm.id = i.assigned_to AND i.status = 'resolved'
GROUP BY d.id, d.name
ORDER BY resolved_incidents DESC;
```

## Тестирование через API

### Пример 1: Простой запрос
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Сколько разработчиков в команде?",
    "use_history": false
  }'
```

**Ожидаемый ответ:**
```json
{
  "message": "В команде работает 7 разработчиков...",
  "session_id": "...",
  "tools_used": [
    {
      "tool_type": "sql",
      "query": "SELECT COUNT(*) FROM team_members WHERE position LIKE '%Developer%'",
      "result_summary": "Найдено 1 записей",
      "metadata": {
        "row_count": 1,
        "sql_query": "..."
      }
    }
  ],
  "sources": ["database: PostgreSQL"]
}
```

### Пример 2: Сложный запрос
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Кто работает над RAG-based Documentation Search и сколько часов уже потрачено?",
    "use_history": false
  }'
```

## Ограничения и известные проблемы

1. **Качество SQL зависит от LLM**: Для сложных запросов может потребоваться несколько попыток
2. **Только SELECT**: Невозможно изменять данные (это сделано намеренно для безопасности)
3. **Ограничение контекста**: LLM видит только схему БД, не сами данные
4. **Производительность**: Генерация SQL через LLM занимает 1-3 секунды

## Улучшения для будущих версий

- [ ] Кэширование частых запросов
- [ ] Few-shot examples для улучшения качества SQL
- [ ] Поддержка follow-up вопросов с контекстом
- [ ] Explain plan для оптимизации запросов
- [ ] Поддержка графиков и визуализации результатов
