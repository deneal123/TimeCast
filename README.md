# TimeCast

Платформа для **обучения моделей предсказания временных рядов и инференса на них**
(прогноз спроса в розничной сети). Состоит из FastAPI-бэкенда (классические и
нейросетевые модели прогнозирования) и React-фронтенда.

```
TimeCast/
├── timecast/   # переиспользуемая библиотека (src-layout, pip-пакет) — ЯДРО
├── backend/    # FastAPI-приложение поверх timecast (обучение/инференс по HTTP)
├── frontend/   # React 18 + Chakra UI + Plotly — веб-дашборд
├── docs/        # документация по проекту
└── TODO.md      # план рефакторинга и улучшений
```

> `backend/` и `frontend/` — отдельные git-репозитории (ветки `backend` и `frontend`).
> Библиотека `timecast/` устанавливается в backend как editable-зависимость (`uv`/pip).

## Документация

- [docs/overview.md](docs/overview.md) — обзор, доменная модель, поток данных
- [docs/backend.md](docs/backend.md) — архитектура бэкенда, API, ML-пайплайны
- [docs/frontend.md](docs/frontend.md) — архитектура фронтенда
- [docs/observations.md](docs/observations.md) — наблюдения по качеству кода
- [TODO.md](TODO.md) — этапы рефакторинга

## Быстрый старт

### Библиотека (timecast)

```bash
cd timecast
uv sync                        # или: pip install -e .
uv run pytest                  # тесты
```

### Backend

```bash
cd backend
cp .env.example .env           # затем при необходимости отредактируйте .env
uv venv && uv pip install -r requirements.txt   # ставит web-зависимости + editable ../timecast
# запуск:
uv run python -m uvicorn server:app --app-dir src/pipeline   # или main.bat / main.sh
```

API поднимается на `http://<HOST>:<SERVER_PORT>` (по умолчанию `:8080`), маршруты —
под префиксом `/server`, Swagger — `/server/docs`. Переменные окружения описаны в
[backend/.env.example](backend/.env.example).

> БД не требуется (вырезана); данные обрабатываются из CSV через DuckDB.
> Backend импортирует библиотеку как пакет `timecast`. См. [docs/backend.md](docs/backend.md).

### Frontend

```bash
cd frontend
cp .env.example .env          # задайте REACT_APP_API_BASE_URL при необходимости
npm install
npm start                      # dev-сервер на http://localhost:3000
```

Базовый URL API настраивается через `REACT_APP_API_BASE_URL`
(по умолчанию `http://localhost:8080/server`).

## Основные сценарии

| Операция | Эндпоинт | Описание |
|----------|----------|----------|
| Сезонная аналитика | `POST /server/season_analytic/` | Декомпозиция ряда (тренд/сезон/остаток) |
| Обучение (classic) | `POST /server/classic_graduate/` | Статистические модели (sktime) |
| Обучение (neiro) | `POST /server/neiro_graduate/` | Нейросеть iTransformer (PyTorch) |
| Инференс (classic) | `POST /server/classic_inference/` | Прогноз статистическими моделями |
| Инференс (neiro) | `POST /server/neiro_inference/` | Прогноз нейросетью |
| Загрузка данных | `POST /server/upload_csv/` | Входные CSV |
| Результаты | `GET /server/get_zip/` | Графики и отчёты архивом |
| Лог обучения | `GET /server/stream-logs` | SSE-стрим в реальном времени |
