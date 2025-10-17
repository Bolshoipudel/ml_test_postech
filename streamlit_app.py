
import streamlit as st
import httpx
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

# API URL можно переопределить через переменную окружения (для Docker)
API_URL = os.getenv("API_URL", "http://localhost:8000")
PAGE_TITLE = "LLM Assistant"
PAGE_ICON = "🤖"

# Иконки для разных типов агентов
TOOL_ICONS = {
    "sql": "🗄️",
    "rag": "📚",
    "web_search": "🌐",
    "router": "🧭",
    "none": "❓"
}

# Цвета для разных типов агентов
TOOL_COLORS = {
    "sql": "#3498db",      # Синий
    "rag": "#2ecc71",      # Зеленый
    "web_search": "#e67e22",  # Оранжевый
    "router": "#9b59b6",   # Фиолетовый
    "none": "#95a5a6"      # Серый
}

# Примеры запросов для быстрого старта
EXAMPLE_QUERIES = {
    "📊 SQL запросы": [
        "Сколько разработчиков работает над PT Sandbox?",
        "Какие открытые инциденты есть по продуктам PT?",
        "Покажи статистику по командным проектам"
    ],
    "📚 Вопросы о продуктах": [
        "Что такое PT Application Inspector?",
        "Какие возможности у PT Sandbox?",
        "Расскажи о PT Network Attack Discovery"
    ],
    "🌐 Веб-поиск": [
        "Последние новости по кибербезопасности",
        "Актуальные тренды в Application Security",
        "Новые уязвимости в веб-приложениях"
    ],
    "🔀 Комбинированные": [
        "Сколько человек работает над PT AI и какие у него функции?",
        "Покажи команду PT Sandbox и опиши его возможности",
        "Какие фичи в разработке для PT NAD?"
    ]
}


# =============================================================================
# НАСТРОЙКА СТРАНИЦЫ
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    /* Основные стили */
    .main {
        background-color: #f5f7fa;
    }

    /* Стили для сообщений */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Стили для метаданных */
    .metadata-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9em;
    }

    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85em;
        font-weight: 500;
        margin: 0.25rem;
    }

    .source-item {
        padding: 0.25rem 0.5rem;
        margin: 0.25rem 0;
        background-color: #e9ecef;
        border-radius: 0.25rem;
        font-size: 0.85em;
    }

    /* Стили для боковой панели */
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "use_history" not in st.session_state:
    st.session_state.use_history = True

if "stats" not in st.session_state:
    st.session_state.stats = None


# =============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С API
# =============================================================================

def check_api_health() -> bool:
    """Проверка доступности API."""
    try:
        response = httpx.get(f"{API_URL}/api/v1/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def send_message(message: str, use_history: bool = True) -> Optional[Dict[str, Any]]:
    """Отправка сообщения в API."""
    try:
        payload = {
            "message": message,
            "use_history": use_history,
            "session_id": st.session_state.session_id
        }

        with st.spinner("Обрабатываю запрос..."):
            response = httpx.post(
                f"{API_URL}/api/v1/chat",
                json=payload,
                timeout=60.0
            )

        if response.status_code == 200:
            data = response.json()
            # Сохраняем session_id
            if "session_id" in data:
                st.session_state.session_id = data["session_id"]
            return data
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None


def get_stats() -> Optional[Dict[str, Any]]:
    """Получение статистики системы."""
    try:
        response = httpx.get(f"{API_URL}/api/v1/stats", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def clear_history() -> bool:
    """Очистка истории чата."""
    try:
        if st.session_state.session_id:
            response = httpx.delete(
                f"{API_URL}/api/v1/history/{st.session_state.session_id}",
                timeout=5.0
            )
            return response.status_code == 200
        return True
    except Exception:
        return False


# =============================================================================
# ФУНКЦИИ ДЛЯ ОТОБРАЖЕНИЯ
# =============================================================================

def render_tool_badge(tool_type: str, confidence: Optional[float] = None) -> str:
    """Создание HTML badge для типа инструмента."""
    icon = TOOL_ICONS.get(tool_type, "❓")
    color = TOOL_COLORS.get(tool_type, "#95a5a6")
    label = tool_type.upper()

    confidence_text = ""
    if confidence is not None:
        confidence_text = f" ({confidence:.0%})"

    return f"""
    <span class="tool-badge" style="background-color: {color}; color: white;">
        {icon} {label}{confidence_text}
    </span>
    """


def render_metadata(response_data: Dict[str, Any]):
    """Отображение метаданных ответа."""
    tools_used = response_data.get("tools_used", [])
    sources = response_data.get("sources", [])

    if not tools_used and not sources:
        return

    with st.expander("🔍 Детали обработки", expanded=False):
        # Отображение использованных инструментов
        if tools_used:
            st.markdown("**Использованные агенты:**")

            for tool in tools_used:
                tool_type = tool.get("tool_type", "none")
                confidence = tool.get("confidence")
                reasoning = tool.get("reasoning")
                result_summary = tool.get("result_summary")
                metadata = tool.get("metadata", {})

                # Badge инструмента
                st.markdown(
                    render_tool_badge(tool_type, confidence),
                    unsafe_allow_html=True
                )

                # Reasoning (только для router)
                if tool_type == "router" and reasoning:
                    st.caption(f"💭 **Логика:** {reasoning}")

                # Результат
                if result_summary:
                    st.caption(f"📊 **Результат:** {result_summary}")

                # SQL запрос (если есть)
                if tool_type == "sql" and metadata.get("sql_query"):
                    with st.expander("SQL запрос"):
                        st.code(metadata["sql_query"], language="sql")

                st.markdown("---")

        # Отображение источников
        if sources:
            st.markdown("**Источники информации:**")
            for source in sources:
                st.markdown(f"- 🔗 {source}")


def render_example_queries():
    """Отображение примеров запросов."""
    st.sidebar.markdown("### 💡 Примеры запросов")

    for category, queries in EXAMPLE_QUERIES.items():
        with st.sidebar.expander(category):
            for query in queries:
                if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                    # Добавляем запрос в чат
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Отправляем запрос
                    response = send_message(query, st.session_state.use_history)

                    if response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.get("message", ""),
                            "metadata": response,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.rerun()


def render_sidebar():
    """Отображение боковой панели."""
    with st.sidebar:
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.markdown("---")

        # Статус API
        st.markdown("### 🔌 Статус системы")
        api_healthy = check_api_health()

        if api_healthy:
            st.success("✅ API доступен")
        else:
            st.error("❌ API недоступен")
            st.caption(f"Проверьте, что сервер запущен на {API_URL}")

        st.markdown("---")

        # Настройки
        st.markdown("### ⚙️ Настройки")

        st.session_state.use_history = st.toggle(
            "Использовать историю",
            value=st.session_state.use_history,
            help="Включить контекст предыдущих сообщений"
        )

        # Кнопка очистки истории
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Очистить чат", use_container_width=True):
                clear_history()
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()

        with col2:
            if st.button("🔄 Обновить", use_container_width=True):
                st.session_state.stats = get_stats()
                st.rerun()

        st.markdown("---")

        # Статистика системы
        if api_healthy:
            st.markdown("### 📊 Статистика")

            if st.session_state.stats is None:
                st.session_state.stats = get_stats()

            if st.session_state.stats:
                stats = st.session_state.stats

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Сессий", stats.get("total_sessions", 0))
                    st.metric("Сообщений", stats.get("total_messages", 0))

                with col2:
                    st.metric("Отзывов", stats.get("total_feedback", 0))
                    avg_rating = stats.get("average_rating", 0)
                    st.metric("Средняя оценка", f"{avg_rating:.1f}/5")

                st.caption(f"🤖 LLM: {stats.get('active_llm_provider', 'N/A')}")
                st.caption(f"🗄️ Vector Store: {stats.get('active_vector_store', 'N/A')}")

        st.markdown("---")

        # Примеры запросов
        render_example_queries()

        st.markdown("---")

        # Информация о проекте
        st.markdown("### ℹ️ О проекте")
        st.caption("**LLM Assistant** - многофункциональный AI ассистент")
        st.caption("🔹 RAG для документации")
        st.caption("🔹 SQL запросы к БД")
        st.caption("🔹 Веб-поиск")
        st.caption("🔹 Умная маршрутизация")

        st.markdown("---")
        st.caption("Made with ❤️")

def main():
    render_sidebar()

    st.title("💬 Чат с LLM Assistant")
    st.markdown("Задайте вопрос о продуктах PT, команде или последних новостях кибербезопасности")

    if not check_api_health():
        st.error("""
        ⚠️ **API недоступен!**

        Убедитесь, что FastAPI сервер запущен:
        ```bash
        python -m uvicorn app.main:app --reload
        ```
        """)
        st.stop()

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant" and "metadata" in message:
                render_metadata(message["metadata"])

    if prompt := st.chat_input("Введите ваш вопрос..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        response = send_message(prompt, st.session_state.use_history)

        if response:
            assistant_message = {
                "role": "assistant",
                "content": response.get("message", ""),
                "metadata": response,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                st.markdown(response.get("message", ""))
                render_metadata(response)

            st.session_state.stats = get_stats()
        else:
            st.error("Не удалось получить ответ от API")


if __name__ == "__main__":
    main()
