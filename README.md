# TimeCast

Платформа для **обучения моделей предсказания временных рядов и инференса на них**
(прогноз спроса в розничной сети): переиспользуемая библиотека `timecast`, тонкий
FastAPI-бэкенд поверх неё и React-дашборд (Plotly).

```text
TimeCast/  (ветка dev — корень монорепозитория)
├── timecast/   # библиотека (src-layout, pip-пакет) — ЯДРО        [сабмодуль → ветка library]
├── backend/    # FastAPI-приложение поверх timecast               [сабмодуль → ветка backend]
├── frontend/   # React 18 + Chakra UI + Plotly — веб-дашборд      [сабмодуль → ветка frontend]
├── docker/     # docker-compose (dev/prod), build.sh, run.sh
├── infra/      # nginx, postgres, redis, minio
├── docs/       # документация
├── Makefile    # build / test / lint / run / stop
└── TODO.md
```

> Один репозиторий, четыре ветки: `backend`, `frontend`, `library`, **`dev`**.
> Ветка `dev` (корень) подключает `backend`/`frontend`/`timecast` как git-**сабмодули**.

## Клонирование

```bash
git clone --recurse-submodules -b dev https://github.com/deneal123/TimeCast.git
cd TimeCast
# если клонировали без сабмодулей:
git submodule update --init --recursive
```

## Запуск через Docker (рекомендуется)

```bash
cp docker/.env.example docker/.env.dev    # подставьте секреты
make build MODE=dev                        # сборка образов
make run   MODE=dev                        # backend :8000 + frontend :3000 + postgres/redis/minio
make stop
```

> Полный стек: backend, frontend, postgres, redis, minio (+ nginx в prod).
> Celery-воркеры — за профилем `celery` (требуют реализации `celery_app`).
> Для prod нужны TLS-сертификаты в `infra/nginx/certs/`.

## Локальный запуск (без Docker)

**Библиотека:**

```bash
cd timecast && uv sync && uv run pytest
```

**Backend** (зависит от `timecast`):

```bash
cd backend
uv venv && uv pip install -e ../timecast \
  fastapi uvicorn python-multipart aiofiles python-dotenv pyyaml rich pydantic
cp ../docker/.env.example .env             # HOST/SERVER_PORT/DEBUG/пути/CORS
uv run uvicorn src.app.server:app --host 0.0.0.0 --port 8000
```
API — под префиксом `/server`, Swagger — `/server/docs`, health — `/server/health`.
Конфигурация только через `.env` (`python-dotenv`); БД не требуется (CSV + DuckDB).

**Frontend:**

```bash
cd frontend
echo "REACT_APP_API_BASE_URL=http://localhost:8000/server" > .env
npm install && npm start                   # http://localhost:3000
```

## Документация

- [docs/overview.md](docs/overview.md) — обзор, доменная модель, поток данных
- [docs/architecture-target.md](docs/architecture-target.md) — целевая архитектура
- [docs/backend.md](docs/backend.md) · [docs/frontend.md](docs/frontend.md) · [docs/observations.md](docs/observations.md)
- [TODO.md](TODO.md) — этапы рефакторинга · [timecast/README.md](timecast/README.md) — API библиотеки

## Эндпоинты API

| Операция | Эндпоинт | Описание |
|----------|----------|----------|
| Health | `GET /server/health` | Проверка живости (Docker/nginx) |
| Сезонная аналитика | `POST /server/season_analytic/` | Декомпозиция ряда (тренд/сезон/остаток) |
| Обучение (classic) | `POST /server/classic_graduate/` | Статистические модели (sktime) |
| Обучение (neiro) | `POST /server/neiro_graduate/` | Нейросеть iTransformer (PyTorch) |
| Инференс (classic) | `POST /server/classic_inference/` | Прогноз статистическими моделями (+ структурный результат) |
| Инференс (neiro) | `POST /server/neiro_inference/` | Прогноз нейросетью (+ структурный результат) |
| Загрузка данных | `POST /server/upload_csv/` | Входные CSV (валидация типа/размера) |
| Результаты | `GET /server/get_zip/` | Графики и отчёты архивом |
| Лог обучения | `GET /server/stream-logs` | SSE-стрим в реальном времени |
