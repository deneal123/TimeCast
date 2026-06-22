# Наблюдения по качеству кода

Сводка слабых мест, выявленных при изучении кода (2026-06-20). Конкретные шаги —
в [../TODO.md](../TODO.md).

## Backend

| # | Наблюдение | Где | Серьёзность |
|---|-----------|-----|-------------|
| B1 | ✅ *Решено:* БД/ORM не использовались — вырезаны (`database/`, `repository/`, `create_sql.py`, `TimeCast.sql`, `pymysql`) | — | — |
| B2 | ✅ *Решено:* общий init-код 5 пайплайнов вынесен в `pipelines/_retail.py::load_retail_dataset` (попутно fix typo `sen_len`→`seq_len`) | `timecast/pipelines/` | — |
| B3 | ✅ *Решено:* библиотека синхронна, сервисы вызывают её через `asyncio.to_thread`; заодно исправлен баг `await torch.load` | `services/*`, `timecast/*` | — |
| B4 | ✅ *Решено:* `requirements.txt` прорежен (~150 → ~30); канон — `pyproject.toml` | `requirements.txt` | — |
| B5 | ✅ *Решено:* чужой `config.yaml` и `path_to_config` удалены | — | — |
| B6 | ✅ *Решено:* CORS ограничен через `CORS_ORIGINS` | `server.py` | — |
| B7 | ✅ *Решено:* `upload_csv` проверяет расширение `.csv`, размер (≤50 МБ), пустоту; защита от path traversal (`basename`) | `services/file_services.py` | — |
| B8 | ✅ *Частично:* сезонные константы → `config.DEFAULT_SEASONAL`; HF-репозитории всё ещё захардкожены | `timecast/*Inference.py` | низкая |
| B9 | ✅ *Решено:* `ClassicProccess` → `ClassicProcess` (wire-алиас `proccess` сохранён) | `timecast/` | — |
| B10 | ✅ *Решено:* в `timecast` 17 тестов (unit + e2e classic/neiro на CPU, маркер `slow`); `pytest -m "not slow"` для быстрого прогона | `timecast/tests/` | — |
| B11 | `LogStreamHandler.emit` смешивает `asyncio.create_task` и `run_until_complete` — хрупко | `server.py:69-83` | средняя |
| B12 | `GPU`-память (`clear_gpu_memory`) объявлена, но не вызывается в обучении/инференсе | `library/utils.py` | средняя |

## Frontend

| # | Наблюдение | Где | Серьёзность |
|---|-----------|-----|-------------|
| F1 | Базовый URL API захардкожен, без `.env` | `API/apiConsts.js` | высокая |
| F2 | Два HTTP-клиента: axios-инстанс создан, но сервисы используют `fetch` | `API/instance.js`, `API/services/*` | средняя |
| F3 | Выбор эндпоинта по форме JSON на фронте — хрупко, легко ошибиться | `pages/query_page.jsx` | средняя |
| F4 | Документация — длинные захардкоженные строки markdown | `pages/documentation_page.jsx` | средняя |
| F5 | Много неиспользуемых зависимостей (formik, yup, tesseract.js, react-pdf, ...) | `package.json` | средняя |
| F6 | Нет error boundary, loading-состояний, индикаторов прогресса | страницы | средняя |
| F7 | `private_Routes` не подключён; auth-инфраструктура мёртвая | `common/private_Routes.jsx` | низкая |
| F8 | Хардкод цветов/размеров вместо темы | компоненты | косметика |
| F9 | Нет тестов; имя пакета `sklad_front` (наследие) | `package.json` | низкая |

## Кросс-секционные

- **Нет единого контракта API** между фронтом и бэком (формы запросов описаны только в
  Pydantic-моделях и в захардкоженной документации фронта). Источник рассинхронизации.
- **Нет контейнеризации / CI** (Dockerfile, docker-compose, pipeline отсутствуют).
- **Нет `.env.example`** ни на фронте, ни на бэке — порог входа высокий.
