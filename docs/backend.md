# Backend — TimeCast

FastAPI-приложение для обучения и инференса моделей прогнозирования временных рядов.
Версия API: **1.3.2** ([server.py:30](../backend/src/pipeline/server.py#L30)).

## Технологический стек

| Категория | Используется в коде |
|-----------|---------------------|
| Web | FastAPI, Uvicorn, Starlette (CORS, StaticFiles, SSE) |
| ML — classic | `sktime` (AutoARIMA / AutoREG / AutoETS / Prophet / TBATS), `statsmodels` |
| ML — neiro | `torch`, `iTransformer`, кастомный `CustomLoss` |
| Данные | `pandas`, `numpy`, `duckdb`, `scikit-learn` (MinMaxScaler, метрики) |
| Веса моделей | HuggingFace Hub (`huggingface-hub`) |
| Логи | `rich` + YAML-конфиг (`logging.yaml`) |
| БД | `pymysql` (прямое подключение, **без ORM**) |

> **Важно:** `requirements.txt` содержит ~150 пакетов (autots, darts, tensorflow/keras,
> xgboost, optuna, prophet и т.д.), но реально импортируется лишь малая часть.
> Большинство — транзитивные или неиспользуемые зависимости. См. [TODO.md](../TODO.md).

## Структура каталогов

> **ML-логика вынесена в отдельный пакет [`timecast/`](../timecast/) на уровне репозитория**
> (src-layout, pip-устанавливаемый). Backend импортирует его как зависимость `timecast`.

```text
backend/                   # FastAPI-приложение поверх timecast
├── pyproject.toml         # зависит от timecast (editable, ../timecast)
├── env.py                 # класс Env — чтение/запись .env
├── setup/                 # кастомный установщик окружения (legacy)
├── requirements*.txt
└── src/
    ├── pipeline/server.py # точка входа FastAPI, маршруты, SSE-логи, перевод доменных ошибок
    ├── services/          # тонкие адаптеры маршрут → публичный API timecast (через to_thread)
    └── utils/             # логирование, хэширование, загрузка файлов, ссылки
```

> БД, `database/`, `repository/`, `create_sql.py`, `TimeCast.sql`, чужой `config.yaml`
> удалены (см. история). Логика прогнозирования — в пакете `timecast` (см.
> [architecture-target.md](architecture-target.md)).

## API-эндпоинты

Все маршруты смонтированы под префиксом `/server`
([server.py:35](../backend/src/pipeline/server.py#L35)). CORS открыт для всех источников.

| Метод | Путь | Тег | Назначение |
|-------|------|-----|------------|
| GET | `/stream-logs` | Stream | SSE-стрим логов приложения |
| GET | `/generate-log` | Stream | Тестовая генерация лога |
| POST | `/upload_csv/` | File | Загрузка входных CSV |
| GET | `/get_zip/` | File | Скачивание результатов архивом |
| POST | `/season_analytic/` | Analytic | Сезонная декомпозиция ряда |
| POST | `/classic_graduate/` | Graduate | Обучение статистических моделей |
| POST | `/neiro_graduate/` | Graduate | Обучение нейросети (iTransformer) |
| POST | `/classic_inference/` | Inference | Прогноз статистическими моделями |
| POST | `/neiro_inference/` | Inference | Прогноз нейросетью |

Все POST принимают Pydantic-модели `Entry*Pipeline`
([pydantic_models.py](../backend/src/library/pydantic_models.py)) и возвращают `Dict`.

## Слои архитектуры

```
Route (server.py)
  → Service (services/*.py)        # async-обёртка + validate_with_pydantic
    → Pipeline (library/*Pipeline.py)   # @dataclass-оркестратор: пути, загрузка данных
      → Dataset + Model/Graduate/Inference   # ядро ML
        → (DuckDB / CSV / HuggingFace)
```

- **Service** — почти всегда один вызов пайплайна (`classic_services.py` и др.).
- **Pipeline** — 5 dataclass-классов. Каждый сам собирает пути к трём CSV, проверяет
  их наличие и поднимает `HTTPException` при отсутствии. **Код инициализации
  дублируется на ~95 % между четырьмя пайплайнами.**
- **`repository/` фактически пуст**, ORM-слоя нет.

## Данные

> **БД вырезана** (2026-06-20). MySQL/`pymysql`, `database/`, `repository/`,
> `create_sql.py`, `TimeCast.sql` удалены — они не использовались в рантайме.
> Продукту БД не нужна.

- Хранилище — **CSV-файлы + DuckDB** в памяти; артефакты (веса, графики, zip) лежат на
  диске по путям из `.env` (`DATA_PATH`, `PLOTS_PATH`, `ZIP_PATH`,
  `WEIGHTS_CLASSIC_PATH`, `WEIGHTS_NEIRO_PATH`).
- Целевое направление: пути и параметры передаются в библиотеку **явно**, а не через
  `env`/глобальные хелперы. См. [architecture-target.md](architecture-target.md).

## ML-пайплайны

### Classic (статистика)
1. `ClassicDataset` — загрузка 3 CSV через DuckDB, склейка, группировка `store → item`.
2. `ClassicGraduate` — для каждого товара и каждого сезонного периода (неделя=7,
   месяц=30, квартал=90) обучает 5 моделей `sktime`, выбирает лучшую по RMSE/R².
   Веса сохраняются (pickle) + JSON-отчёт.
3. `ClassicInference` — грузит веса с HuggingFace, режимы `estimate` (на тесте) и
   `future` (вперёд), рисует графики.

### Neiro (нейросеть)
1. `NeiroDataset` — PyTorch `Dataset`: декомпозиция ряда (тренд/сезон/остаток через
   statsmodels), нормализация `MinMaxScaler`, скользящее окно, `DataLoader`.
2. `NeiroGraduate` — `iTransformer` + `Adam` + `CustomLoss` (SmoothL1 + штрафы за
   отрицательные значения / cosine-embedding), `ReduceLROnPlateau`, авто-выбор
   CUDA/CPU. Чекпойнт = `model_state_dict` + `optimizer_state_dict` + epoch.
3. `NeiroInference` — грузит чекпойнт, прямой проход, обратное масштабирование,
   графики.

### SeasonAnalytic
`ClassicDataset → ClassicProccess` (sic, опечатка в имени): `seasonal_decompose`/STL,
удаление выбросов по IQR, тесты стационарности (ADF), ACF/Ljung-Box, визуализация.

## Конфигурация и логирование

- **Конфиг** — только через `.env` (`env.py`, класс `Env` с записью обратно в файл).
  Schema-валидации настроек нет. `config_parser.py` есть, но не используется.
- **Логи** — `logging.yaml` (root=INFO, file+console, приглушены cmdstanpy/statsmodels/
  pandas) + `rich` в `custom_logging.py`. Для фронтенда логи дублируются в
  `asyncio.Queue` и отдаются как SSE через `/stream-logs`.

## Запуск

- Установка окружения: `setup.bat` / `setup.sh` (кастомный установщик в `setup/`).
- Запуск сервера: `main.bat` / `main.sh` → `run_server()` (uvicorn,
  host/port/reload из `.env`).
- При `OFF_DATABASE=FALSE` перед стартом выполняется `create_sql.py` (но таблиц в
  SQL нет — фактически no-op).

См. также [observations.md](observations.md) — слабые места и техдолг.
