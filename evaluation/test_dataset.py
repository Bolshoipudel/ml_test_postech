"""
Тестовый датасет для evaluation LLM Assistant.

Содержит 50 тестовых кейсов для проверки:
- Router Agent (правильность выбора инструмента)
- SQL Agent (запросы к БД)
- RAG Agent (поиск в документации)
- Web Search Agent (актуальная информация)
- Orchestrator (комбинированные запросы)
"""

from typing import List, Dict, Any


# =============================================================================
# SQL QUERIES (12 кейсов) - Запросы к базе данных team_mock.db
# =============================================================================

SQL_TEST_CASES = [
    {
        "id": "sql_001",
        "query": "Сколько QA-инженеров работает в команде?",
        "expected_tool": "sql",
        "expected_answer_contains": ["QA", "инженер", "1"],
        "ground_truth": "В команде работает 1 QA-инженер.",
        "context": ["SQLite database: employees table"],
        "category": "team_count",
        "min_confidence": 0.9
    },
    {
        "id": "sql_002",
        "query": "Кто работает программистом?",
        "expected_tool": "sql",
        "expected_answer_contains": ["программист", "Jane", "John"],
        "ground_truth": "Программистами работают Jane Doe и John Brown.",
        "context": ["SQLite database: employees table"],
        "category": "team_members",
        "min_confidence": 0.9
    },
    {
        "id": "sql_003",
        "query": "Сколько всего сотрудников в компании?",
        "expected_tool": "sql",
        "expected_answer_contains": ["10", "сотрудник"],
        "ground_truth": "В компании работает 10 сотрудников.",
        "context": ["SQLite database: employees table"],
        "category": "team_count",
        "min_confidence": 0.95
    },
    {
        "id": "sql_004",
        "query": "Кто такой Bob Johnson и чем он занимается?",
        "expected_tool": "sql",
        "expected_answer_contains": ["Bob", "Johnson", "программист", "ведущий"],
        "ground_truth": "Bob Johnson — Ведущий программист с экспертизой в производительности бэкенда, безопасности API и аутентификации пользователей.",
        "context": ["SQLite database: employees, employee_expertise tables"],
        "category": "employee_info",
        "min_confidence": 0.85
    },
    {
        "id": "sql_005",
        "query": "Какие роли представлены в команде?",
        "expected_tool": "sql",
        "expected_answer_contains": ["менеджер", "программист", "аналитик", "инженер"],
        "ground_truth": "В команде есть следующие роли: Менеджер продукта, Ведущий программист, Программист, Аналитик по кибербезопасности, Аналитик по киберугрозам, QA-инженер, Главный инженер, DevOps-инженер.",
        "context": ["SQLite database: employees table, DISTINCT roles"],
        "category": "team_structure",
        "min_confidence": 0.9
    },
    {
        "id": "sql_006",
        "query": "Сколько менеджеров продукта в команде?",
        "expected_tool": "sql",
        "expected_answer_contains": ["менеджер", "продукт", "2"],
        "ground_truth": "В команде работает 2 менеджера продукта.",
        "context": ["SQLite database: employees table"],
        "category": "team_count",
        "min_confidence": 0.95
    },
    {
        "id": "sql_007",
        "query": "Кто специализируется на безопасности API?",
        "expected_tool": "sql",
        "expected_answer_contains": ["Bob", "безопасность", "API"],
        "ground_truth": "Безопасностью API занимается Bob Johnson (Ведущий программист).",
        "context": ["SQLite database: employee_expertise, expertise_areas tables"],
        "category": "expertise",
        "min_confidence": 0.85
    },
    {
        "id": "sql_008",
        "query": "У кого больше всего лет опыта в команде?",
        "expected_tool": "sql",
        "expected_answer_contains": ["опыт", "лет", "Bob", "12"],
        "ground_truth": "Больше всего опыта у Bob Johnson — 12 лет.",
        "context": ["SQLite database: employees table, MAX(years_of_experience)"],
        "category": "employee_info",
        "min_confidence": 0.9
    },
    {
        "id": "sql_009",
        "query": "Какие области экспертизы есть в команде?",
        "expected_tool": "sql",
        "expected_answer_contains": ["экспертиза", "безопасность", "киберугроз"],
        "ground_truth": "В команде представлены следующие области экспертизы: Аналитика киберугроз, Аутентификация пользователей, Безопасность API, Безопасность CI/CD, Дорожная карта продукта, Интеграция с SIEM, Настройка файрволов, Облачная безопасность, Предотвращение утечек данных (DLP), Производительность бэкенда, Системы обнаружения вторжений (IDS), Сканирование уязвимостей.",
        "context": ["SQLite database: expertise_areas table"],
        "category": "expertise",
        "min_confidence": 0.85
    },
    {
        "id": "sql_010",
        "query": "Есть ли в команде DevOps-инженер?",
        "expected_tool": "sql",
        "expected_answer_contains": ["DevOps", "да", "есть"],
        "ground_truth": "Да, в команде есть DevOps-инженер — George Miller.",
        "context": ["SQLite database: employees table"],
        "category": "team_members",
        "min_confidence": 0.9
    },
    {
        "id": "sql_011",
        "query": "Кто занимается интеграцией с SIEM?",
        "expected_tool": "sql",
        "expected_answer_contains": ["SIEM", "интеграция", "Charlie"],
        "ground_truth": "Интеграцией с SIEM занимается Charlie Smith (Аналитик по кибербезопасности).",
        "context": ["SQLite database: employee_expertise, expertise_areas tables"],
        "category": "expertise",
        "min_confidence": 0.85
    },
    {
        "id": "sql_012",
        "query": "Покажи email всех менеджеров продукта",
        "expected_tool": "sql",
        "expected_answer_contains": ["email", "@pt.corp", "менеджер"],
        "ground_truth": "Email менеджеров продукта: a.williams@pt.corp (Alice Williams), h.wilson@pt.corp (Hannah Wilson).",
        "context": ["SQLite database: employees table"],
        "category": "contact_info",
        "min_confidence": 0.9
    }
]


# =============================================================================
# RAG QUERIES (12 кейсов) - Поиск в документации
# =============================================================================

RAG_TEST_CASES = [
    {
        "id": "rag_001",
        "query": "Что такое PT Sandbox и для чего он используется?",
        "expected_tool": "rag",
        "expected_answer_contains": ["PT Sandbox", "анализ", "вредоносн", "файл"],
        "ground_truth": "PT Sandbox — система динамического анализа файлов для обнаружения вредоносного ПО и угроз безопасности.",
        "context": ["PT Sandbox documentation"],
        "category": "product_overview",
        "min_confidence": 0.85
    },
    {
        "id": "rag_002",
        "query": "Какие типы файлов поддерживает PT Sandbox для анализа в Windows?",
        "expected_tool": "rag",
        "expected_answer_contains": ["файл", "Windows", "EXE", "DOC", "PDF"],
        "ground_truth": "PT Sandbox для Windows поддерживает исполняемые файлы (EXE, BAT, CMD, PS1), офисные документы (DOC, DOCX, XLS, XLSX, PPT, PPTX), PDF, архивы и другие форматы.",
        "context": ["PT Sandbox file types documentation"],
        "category": "features",
        "min_confidence": 0.8
    },
    {
        "id": "rag_003",
        "query": "Может ли PT Sandbox извлекать файлы из архивов?",
        "expected_tool": "rag",
        "expected_answer_contains": ["архив", "извлечение", "ZIP", "RAR", "вложенн"],
        "ground_truth": "Да, PT Sandbox извлекает и проверяет файлы из архивов (7Z, ZIP, RAR, TAR и др.), включая вложенные архивы.",
        "context": ["PT Sandbox file extraction documentation"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_004",
        "query": "Поддерживает ли PT Sandbox анализ PDF файлов?",
        "expected_tool": "rag",
        "expected_answer_contains": ["PDF", "поддержива", "да", "анализ"],
        "ground_truth": "Да, PT Sandbox поддерживает анализ PDF файлов, включая извлечение ссылок и проверку QR-кодов на изображениях.",
        "context": ["PT Sandbox supported file types"],
        "category": "features",
        "min_confidence": 0.9
    },
    {
        "id": "rag_005",
        "query": "Какие форматы архивов поддерживаются для извлечения?",
        "expected_tool": "rag",
        "expected_answer_contains": ["архив", "7Z", "ZIP", "RAR", "TAR"],
        "ground_truth": "PT Sandbox поддерживает извлечение из следующих архивов: 7Z, ACE, ARJ, CPIO, LZH, RAR, TAR, ZIP.",
        "context": ["PT Sandbox file extraction documentation"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_006",
        "query": "Может ли PT Sandbox анализировать исполняемые файлы Linux?",
        "expected_tool": "rag",
        "expected_answer_contains": ["Linux", "ELF", "SH", "исполняем"],
        "ground_truth": "Да, PT Sandbox поддерживает анализ исполняемых файлов Linux: ELF и SH скрипты.",
        "context": ["PT Sandbox supported file types for Linux"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_007",
        "query": "Какие офисные документы можно проверить в PT Sandbox?",
        "expected_tool": "rag",
        "expected_answer_contains": ["офисн", "документ", "Word", "Excel", "PowerPoint"],
        "ground_truth": "PT Sandbox поддерживает офисные документы: Word (DOC, DOCX), Excel (XLS, XLSX), PowerPoint (PPT, PPTX), а также ODT, ODS, ODP и RTF форматы.",
        "context": ["PT Sandbox office file formats"],
        "category": "features",
        "min_confidence": 0.8
    },
    {
        "id": "rag_008",
        "query": "Извлекает ли PT Sandbox ссылки из документов?",
        "expected_tool": "rag",
        "expected_answer_contains": ["ссылк", "извлечение", "URL", "документ"],
        "ground_truth": "Да, PT Sandbox извлекает и проверяет ссылки из HTML, RTF, офисных документов (MSOOXML, ODF), PDF и email файлов.",
        "context": ["PT Sandbox link extraction features"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_009",
        "query": "Какие установочные пакеты поддерживаются в PT Sandbox?",
        "expected_tool": "rag",
        "expected_answer_contains": ["установочн", "пакет", "MSI", "DEB", "RPM"],
        "ground_truth": "PT Sandbox поддерживает следующие установочные пакеты: для Windows — MSI, APPX, MSIX; для Linux — DEB, RPM.",
        "context": ["PT Sandbox installer file types"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_010",
        "query": "Работает ли PT Sandbox с email файлами?",
        "expected_tool": "rag",
        "expected_answer_contains": ["email", "письм", "EML", "MSG"],
        "ground_truth": "Да, PT Sandbox работает с email файлами (EML, MSG, TNEF), извлекая заголовки, тело письма, вложения и ссылки.",
        "context": ["PT Sandbox email processing"],
        "category": "features",
        "min_confidence": 0.85
    },
    {
        "id": "rag_011",
        "query": "Может ли PT Sandbox декодировать QR-коды в PDF?",
        "expected_tool": "rag",
        "expected_answer_contains": ["QR", "код", "PDF", "декодирова"],
        "ground_truth": "Да, PT Sandbox извлекает и проверяет ссылки, закодированные в QR-кодах на изображениях внутри PDF файлов.",
        "context": ["PT Sandbox PDF analysis features"],
        "category": "features",
        "min_confidence": 0.8
    },
    {
        "id": "rag_012",
        "query": "Поддерживает ли PT Sandbox анализ Java приложений?",
        "expected_tool": "rag",
        "expected_answer_contains": ["Java", "JAR", "JNLP"],
        "ground_truth": "Да, PT Sandbox поддерживает анализ Java приложений в форматах JAR и JNLP на Windows и Linux.",
        "context": ["PT Sandbox Java file support"],
        "category": "features",
        "min_confidence": 0.85
    }
]


# =============================================================================
# WEB SEARCH QUERIES (12 кейсов) - Актуальная информация из интернета
# =============================================================================

WEB_SEARCH_TEST_CASES = [
    {
        "id": "web_001",
        "query": "Последние новости по кибербезопасности за эту неделю",
        "expected_tool": "web_search",
        "expected_answer_contains": ["новости", "кибербезопасность", "2025"],
        "ground_truth": "Актуальные новости по кибербезопасности включают последние инциденты, уязвимости и тренды в области защиты информации.",
        "context": ["Web search: Tavily API, recent news"],
        "category": "news",
        "min_confidence": 0.9
    },
    {
        "id": "web_002",
        "query": "Какие новые уязвимости были обнаружены недавно?",
        "expected_tool": "web_search",
        "expected_answer_contains": ["уязвимост", "CVE", "обнаружен"],
        "ground_truth": "Недавно обнаруженные уязвимости включают критические баги в популярном ПО и системах.",
        "context": ["Web search: recent vulnerabilities"],
        "category": "vulnerabilities",
        "min_confidence": 0.85
    },
    {
        "id": "web_003",
        "query": "Актуальные тренды в области Application Security",
        "expected_tool": "web_search",
        "expected_answer_contains": ["тренд", "Application Security", "SAST", "DAST"],
        "ground_truth": "Актуальные тренды включают DevSecOps, AI-powered security testing и shift-left подходы.",
        "context": ["Web search: AppSec trends"],
        "category": "trends",
        "min_confidence": 0.8
    },
    {
        "id": "web_004",
        "query": "Что нового в области защиты от вредоносного ПО?",
        "expected_tool": "web_search",
        "expected_answer_contains": ["вредоносн", "malware", "защита"],
        "ground_truth": "Новые методы защиты включают AI-based detection, behavioral analysis и sandboxing технологии.",
        "context": ["Web search: malware protection innovations"],
        "category": "trends",
        "min_confidence": 0.8
    },
    {
        "id": "web_005",
        "query": "Последние кибератаки на крупные компании",
        "expected_tool": "web_search",
        "expected_answer_contains": ["кибератак", "компани", "инцидент"],
        "ground_truth": "Последние инциденты включают атаки на инфраструктуру, ransomware и data breaches.",
        "context": ["Web search: recent cyber attacks"],
        "category": "news",
        "min_confidence": 0.85
    },
    {
        "id": "web_006",
        "query": "Новости о Positive Technologies",
        "expected_tool": "web_search",
        "expected_answer_contains": ["Positive Technologies", "PT", "новост"],
        "ground_truth": "Последние новости о компании Positive Technologies, её продуктах и достижениях.",
        "context": ["Web search: PT news"],
        "category": "company_news",
        "min_confidence": 0.85
    },
    {
        "id": "web_007",
        "query": "Сколько стоит PT Sandbox?",
        "expected_tool": "web_search",
        "expected_answer_contains": ["PT Sandbox", "стоимость", "цена"],
        "ground_truth": "Информация о ценах на PT Sandbox доступна на сайте Positive Technologies или у партнёров.",
        "context": ["Web search: PT Sandbox pricing"],
        "category": "product_info",
        "min_confidence": 0.7
    },
    {
        "id": "web_008",
        "query": "Актуальные конференции по кибербезопасности в 2025",
        "expected_tool": "web_search",
        "expected_answer_contains": ["конференци", "2025", "кибербезопасность"],
        "ground_truth": "В 2025 году запланированы крупные конференции по кибербезопасности: Black Hat, RSA Conference и др.",
        "context": ["Web search: cybersecurity conferences 2025"],
        "category": "events",
        "min_confidence": 0.8
    },
    {
        "id": "web_009",
        "query": "Что такое DevSecOps и почему это важно сейчас?",
        "expected_tool": "web_search",
        "expected_answer_contains": ["DevSecOps", "security", "DevOps"],
        "ground_truth": "DevSecOps — интеграция безопасности в DevOps процессы для раннего выявления уязвимостей.",
        "context": ["Web search: DevSecOps definition and importance"],
        "category": "concepts",
        "min_confidence": 0.75
    },
    {
        "id": "web_010",
        "query": "Статистика кибератак в 2025 году",
        "expected_tool": "web_search",
        "expected_answer_contains": ["статистик", "кибератак", "2025"],
        "ground_truth": "Статистика показывает рост числа кибератак в 2025 году по сравнению с предыдущими годами.",
        "context": ["Web search: cyber attacks statistics 2025"],
        "category": "statistics",
        "min_confidence": 0.8
    },
    {
        "id": "web_011",
        "query": "Новые AI-инструменты для кибербезопасности",
        "expected_tool": "web_search",
        "expected_answer_contains": ["AI", "искусственн", "инструмент", "кибербезопасность"],
        "ground_truth": "Новые AI-инструменты включают автоматизированный анализ угроз, предсказательную аналитику и интеллектуальные SIEM системы.",
        "context": ["Web search: AI tools for cybersecurity"],
        "category": "trends",
        "min_confidence": 0.8
    },
    {
        "id": "web_012",
        "query": "Ransomware тренды в 2025",
        "expected_tool": "web_search",
        "expected_answer_contains": ["ransomware", "тренд", "2025"],
        "ground_truth": "Ransomware атаки продолжают эволюционировать с новыми тактиками двойного вымогательства и таргетированными атаками.",
        "context": ["Web search: ransomware trends 2025"],
        "category": "threats",
        "min_confidence": 0.8
    }
]


# =============================================================================
# MULTIPLE QUERIES (10 кейсов) - Комбинированные запросы (SQL + RAG, SQL + Web и т.д.)
# =============================================================================

MULTIPLE_TEST_CASES = [
    {
        "id": "multi_001",
        "query": "Сколько человек работает в команде и какие продукты PT они могли бы использовать?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["10", "сотрудник", "PT Sandbox", "PT AI"],
        "ground_truth": "В команде работает 10 сотрудников. Для их работы подходят продукты PT: PT Sandbox для анализа файлов, PT Application Inspector для анализа кода, PT Network Attack Discovery для мониторинга сети.",
        "context": ["SQLite database + PT products documentation"],
        "category": "combined_info",
        "min_confidence": 0.8
    },
    {
        "id": "multi_002",
        "query": "Кто занимается безопасностью API и что такое PT Application Inspector?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["Bob", "безопасность API", "PT AI", "SAST"],
        "ground_truth": "Безопасностью API занимается Bob Johnson. PT Application Inspector (PT AI) — это инструмент статического анализа кода (SAST) для выявления уязвимостей в приложениях.",
        "context": ["SQLite database + PT AI documentation"],
        "category": "combined_info",
        "min_confidence": 0.8
    },
    {
        "id": "multi_003",
        "query": "Сколько аналитиков по кибербезопасности в команде и какие новости по этой теме есть сейчас?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "web_search"],
        "expected_answer_contains": ["аналитик", "кибербезопасность", "новост"],
        "ground_truth": "В команде 1 аналитик по кибербезопасности (Charlie Smith) и 1 аналитик по киберугрозам (Eve Davis). Последние новости по кибербезопасности включают...",
        "context": ["SQLite database + Web search"],
        "category": "combined_info",
        "min_confidence": 0.75
    },
    {
        "id": "multi_004",
        "query": "Какие области экспертизы есть в команде и как PT Sandbox может помочь в их работе?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["экспертиза", "PT Sandbox", "анализ", "безопасность"],
        "ground_truth": "В команде есть экспертиза в аналитике киберугроз, безопасности API, SIEM интеграции и др. PT Sandbox может помочь в динамическом анализе подозрительных файлов и выявлении угроз.",
        "context": ["SQLite database + PT Sandbox documentation"],
        "category": "combined_info",
        "min_confidence": 0.75
    },
    {
        "id": "multi_005",
        "query": "Есть ли в команде DevOps-инженер и какие тренды DevSecOps актуальны сейчас?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "web_search"],
        "expected_answer_contains": ["DevOps", "есть", "DevSecOps", "тренд"],
        "ground_truth": "Да, в команде есть DevOps-инженер — George Miller. Актуальные тренды DevSecOps включают shift-left security, автоматизацию тестирования безопасности в CI/CD и Infrastructure as Code security.",
        "context": ["SQLite database + Web search"],
        "category": "combined_info",
        "min_confidence": 0.75
    },
    {
        "id": "multi_006",
        "query": "Кто такая Alice Williams и какие PT продукты подходят для менеджеров продукта?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["Alice", "менеджер продукта", "PT"],
        "ground_truth": "Alice Williams — менеджер продукта с экспертизой в DLP и дорожной карте продукта. Для менеджеров подходят все PT продукты, особенно PT ISIM для управления инцидентами.",
        "context": ["SQLite database + PT products documentation"],
        "category": "combined_info",
        "min_confidence": 0.75
    },
    {
        "id": "multi_007",
        "query": "Сколько программистов в команде и поддерживает ли PT Sandbox анализ кода?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["программист", "2", "PT Sandbox", "анализ"],
        "ground_truth": "В команде 3 программиста (Bob Johnson, Jane Doe, John Brown). PT Sandbox специализируется на динамическом анализе файлов, а для статического анализа кода используется PT Application Inspector.",
        "context": ["SQLite database + PT Sandbox documentation"],
        "category": "combined_info",
        "min_confidence": 0.75
    },
    {
        "id": "multi_008",
        "query": "Кто занимается облачной безопасностью и какие новости по Cloud Security?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "web_search"],
        "expected_answer_contains": ["облачн", "безопасность", "новост"],
        "ground_truth": "Облачной безопасностью занимается Frank Lee (Главный инженер). Последние новости включают новые методы защиты облачных инфраструктур и zero-trust подходы.",
        "context": ["SQLite database + Web search"],
        "category": "combined_info",
        "min_confidence": 0.7
    },
    {
        "id": "multi_009",
        "query": "Какой средний опыт работы в команде и какие навыки наиболее востребованы в индустрии сейчас?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "web_search"],
        "expected_answer_contains": ["опыт", "лет", "навык", "индустри"],
        "ground_truth": "Средний опыт работы в команде составляет примерно 7-8 лет. Наиболее востребованы навыки в области cloud security, AI/ML for security и DevSecOps.",
        "context": ["SQLite database + Web search"],
        "category": "combined_info",
        "min_confidence": 0.7
    },
    {
        "id": "multi_010",
        "query": "Кто работает с SIEM и что такое PT NAD?",
        "expected_tool": "multiple",
        "expected_tools": ["sql", "rag"],
        "expected_answer_contains": ["SIEM", "Charlie", "PT NAD", "сет"],
        "ground_truth": "С SIEM работает Charlie Smith (интеграция). PT NAD (PT Network Attack Discovery) — система обнаружения сетевых атак и мониторинга безопасности сети.",
        "context": ["SQLite database + PT NAD documentation"],
        "category": "combined_info",
        "min_confidence": 0.75
    }
]


# =============================================================================
# NONE QUERIES (4 кейса) - Нерелевантные запросы (новая функция Router Agent)
# =============================================================================

NONE_TEST_CASES = [
    {
        "id": "none_001",
        "query": "Какая сейчас погода в Москве?",
        "expected_tool": "none",
        "expected_answer_contains": ["нерелевантн", "помочь", "не могу", "вопрос"],
        "ground_truth": "Извините, но я специализируюсь на вопросах о команде, продуктах PT Security и кибербезопасности. Я не могу помочь с вопросами о погоде.",
        "context": [],
        "category": "irrelevant",
        "min_confidence": 0.85
    },
    {
        "id": "none_002",
        "query": "Как приготовить борщ?",
        "expected_tool": "none",
        "expected_answer_contains": ["нерелевантн", "помочь", "не могу"],
        "ground_truth": "Извините, я не могу помочь с кулинарными рецептами. Я специализируюсь на вопросах о кибербезопасности и продуктах PT Security.",
        "context": [],
        "category": "irrelevant",
        "min_confidence": 0.9
    },
    {
        "id": "none_003",
        "query": "Когда будет следующее полнолуние?",
        "expected_tool": "none",
        "expected_answer_contains": ["нерелевантн", "помочь", "не могу"],
        "ground_truth": "Извините, астрономические вопросы не входят в мою область компетенции. Я могу помочь с вопросами о команде, PT продуктах и кибербезопасности.",
        "context": [],
        "category": "irrelevant",
        "min_confidence": 0.9
    },
    {
        "id": "none_004",
        "query": "Сколько будет 2+2?",
        "expected_tool": "none",
        "expected_answer_contains": ["4", "математик"],
        "ground_truth": "2+2 = 4. Однако, я специализируюсь на вопросах о кибербезопасности, команде и продуктах PT. Могу я помочь вам с чем-то в этой области?",
        "context": [],
        "category": "irrelevant",
        "min_confidence": 0.8
    }
]


# =============================================================================
# Объединение всех тестовых кейсов
# =============================================================================

ALL_TEST_CASES = (
    SQL_TEST_CASES +
    RAG_TEST_CASES +
    WEB_SEARCH_TEST_CASES +
    MULTIPLE_TEST_CASES +
    NONE_TEST_CASES
)


# =============================================================================
# Утилиты для работы с датасетом
# =============================================================================

def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Получить тесты по категории."""
    return [tc for tc in ALL_TEST_CASES if tc["category"] == category]


def get_test_cases_by_tool(tool: str) -> List[Dict[str, Any]]:
    """Получить тесты по типу инструмента."""
    return [tc for tc in ALL_TEST_CASES if tc["expected_tool"] == tool]


def get_test_case_by_id(test_id: str) -> Dict[str, Any]:
    """Получить тест по ID."""
    for tc in ALL_TEST_CASES:
        if tc["id"] == test_id:
            return tc
    raise ValueError(f"Test case with id '{test_id}' not found")


def print_dataset_statistics():
    """Вывести статистику датасета."""
    print("=" * 80)
    print("EVALUATION DATASET STATISTICS")
    print("=" * 80)
    print(f"Total test cases: {len(ALL_TEST_CASES)}")
    print()
    print("By tool type:")
    print(f"  SQL:        {len(SQL_TEST_CASES)}")
    print(f"  RAG:        {len(RAG_TEST_CASES)}")
    print(f"  Web Search: {len(WEB_SEARCH_TEST_CASES)}")
    print(f"  Multiple:   {len(MULTIPLE_TEST_CASES)}")
    print(f"  None:       {len(NONE_TEST_CASES)}")
    print()
    print("Categories:")
    categories = {}
    for tc in ALL_TEST_CASES:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("=" * 80)


if __name__ == "__main__":
    print_dataset_statistics()
