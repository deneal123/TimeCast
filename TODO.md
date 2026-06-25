# TODO — TimeCast: библиотека-ядро + дашборд

> **Северная звезда:** переиспользуемая **Python-библиотека** для обучения и инференса
> моделей прогнозирования временных рядов, поверх которой работает **быстрый дашборд**
> (визуализация обучения и инференса). База данных в продукте не нужна.

## Целевая архитектура

```text
┌──────────────────────────────────────────────┐
│  timecast/  (pip-пакет, чистый Python API)     │  ← ЯДРО И ПРОДУКТ
│  data · models · training · inference · season │
│  без FastAPI, без env, без хардкода путей       │
└──────────────────────────────────────────────┘
        ▲ import                    ▲ import
        │                           │
┌───────────────┐          ┌──────────────────┐
│ FastAPI (API) │          │ ноутбуки/скрипты │  ← библиотеку можно
│ тонкий слой   │          │ др. сервисы      │     использовать напрямую
└───────────────┘
        ▲ HTTP / SSE
        │
┌──────────────────────────┐
│ React-дашборд + Plotly    │  ← визуализация прогнозов, метрик, live-логов обучения
└──────────────────────────┘
```

**Принятые решения:**

- Библиотека → отдельный pip-устанавливаемый пакет с чистым API.
- Дашборд → развиваем существующий React + Plotly; FastAPI остаётся тонким API-слоем.
- БД → **вырезана** (см. «Сделано»).

Подробные находки — в [docs/observations.md](docs/observations.md),
целевая архитектура — в [docs/architecture-target.md](docs/architecture-target.md).

---

## ✅ Сделано

- Вырезана БД целиком: удалены `src/database/`, `src/repository/`, `create_sql.py`,
  `TimeCast.sql`, DB-блок в `server.py`, `pymysql` из requirements, DB-переменные из `.env.example`.
- `.env.example` для backend и frontend; базовый URL фронта вынесен в `REACT_APP_API_BASE_URL`.
- CORS ограничен через `CORS_ORIGINS` (вместо `*`).
- Удалён чужой `src/config.yaml` и мёртвый `path_to_config()`.
- Прорежены неиспользуемые npm-зависимости; пакет переименован в `timecast-frontend`.
- README верхнего уровня.

---

## Этап A. Библиотека: отвязка от веб-слоя (ядро вектора)

Цель: `library` ничего не знает про HTTP, env и фиксированные пути.

- [x] Доменные исключения (`exceptions.py`: `TimeCastError`, `DataNotFoundError`,
      `ValidationFailedError`) вместо `fastapi.HTTPException` внутри библиотеки. *(7 файлов, 26 raise)*
- [x] Центральный перевод доменных ошибок → HTTP-ответы на уровне FastAPI (exception handler).
- [x] Убрать `env`/`path_to_project()` из библиотеки: пути инжектятся через
      `src/library/config.py` (`get_paths`/`configure_paths`); FastAPI подставляет пути из `.env`
      при старте. *(12 файлов)*
- [x] Вынести имена входных CSV и сезонные константы в конфиг
      (`config.py`: `sales_csv`/`sales_dates_csv`/`sales_prices_csv`, `DEFAULT_SEASONAL`).
- [x] Заменить «фейковый» `async` на синхронные методы (обучение — CPU-bound):
      все методы библиотеки синхронны; `save_plot_into_server`/HF-загрузка переведены на
      sync (open/requests); сервисы FastAPI вызывают их через `asyncio.to_thread`.
      Заодно исправлен баг `await torch.load(...)`. Загрузка CSV (`write_file_into_server`)
      остаётся async (реальный поток UploadFile).

> ✅ Этап A проверен запуском (uv venv, Python 3.12): `import src.library` без FastAPI/env;
> сервер собирается; end-to-end через `TestClient` — пайплайн исполняется в worker-потоке
> (`to_thread`), `DataNotFoundError` корректно переводится в HTTP 404 центральным обработчиком.

## Этап B. Библиотека: оформление как pip-пакет

Цель: `pip install timecast` и `from timecast import ...` из любого проекта.

- [x] `pyproject.toml` (uv): метаданные, реальные зависимости, ruff/mypy, torch CUDA-индекс.
- [x] Переместить `src/library/` → пакет `timecast/`, переписать импорты `from src.*`.
      Библиотека самодостаточна: своё логирование (`_logging`), I/O (`_io`), каталоги (`_dirs`).
- [x] Публичный API в `timecast/__init__.py` (`timecast/api.py`): `season_analytic`,
      `train_classic`, `infer_classic`, `train_neiro`, `infer_neiro` + `configure_paths`/`get_paths`
      и доменные исключения.
- [x] Прорядить `requirements.txt` (~150 → ~30 реально используемых); `requirements_external.txt`
      без неиспользуемого tensorflow.
- [x] Исправить опечатку имён `ClassicProccess` → `ClassicProcess` (класс/модуль/`EntryClassicProcess`,
      метод `.process()`). Wire-алиас `"proccess"` сохранён (контракт с фронтом).

> ✅ Этап B проверен запуском: `import timecast` (без `src.*`), `timecast.season_analytic({...})`
> работает как библиотека; сервер на новом пакете — end-to-end 404 через `to_thread`.
>
> **Реструктуризация (современные стандарты):** библиотека вынесена из `backend/` на уровень
> репозитория — `timecast/` (src-layout `src/timecast/`, свой `pyproject.toml`, `tests/`,
> `README.md`, `py.typed`). Backend зависит от неё как от editable-пакета
> (`[tool.uv.sources] timecast = { path = "../timecast" }`, `package = false`).
> Проверено: editable-install, `import timecast` из нового пути, **pytest 3 passed**, сервер end-to-end 404.
> Остаётся опционально: разбиение пакета на подмодули (`data`/`models`/`training`/`inference`).

## Этап C. FastAPI как тонкий слой над библиотекой

Цель: сервер только принимает запрос, вызывает библиотеку, отдаёт результат/стрим.

- [x] Сервисы вызывают публичный API библиотеки (`timecast.train_classic`/`infer_*`/…),
      стали тонкими адаптерами.
- [x] CPU-bound вызовы в `asyncio.to_thread` (сделано в Этапе A).
- [x] Эндпоинты инференса возвращают структурированные результаты
      (`timecast.collect_results` / `serialize_inference_results`): `{item_id: {period: {rmse, r2, pred}}}`,
      JSON-безопасно (модели/NaN отброшены). Юнит-тест на синтетике — PASS.
- [x] Валидация и безопасность загрузки CSV: расширение `.csv`, лимит 50 МБ, пустой файл,
      защита от path traversal (basename) — проверено `TestClient` (400/413/200).

> ✅ Этап C проверен запуском: сервисы через публичный API (end-to-end 404 сохранился);
> загрузка CSV — 400 на плохое расширение/traversal, 200 на валидный; сериализатор результатов — unit PASS.
> Полный happy-path инференса (реальные веса с HF + данные) в этой среде не прогонялся.

## Этап D. Дашборд (React + Plotly)

Цель: видеть обучение и инференс, а не только текстовый лог.

- [x] Подключить Plotly (`react-plotly.js` + `plotly.js-dist-min` через фабрику — обход
      несовместимости с webpack-5).
- [x] График инференса: прогноз по `item_id`/периодам + метрики (rmse/r²)
      ([ForecastChart.jsx](../frontend/src/components/ForecastChart.jsx)), привязан к структурному
      ответу `/classic_inference/`·`/neiro_inference/`; в `query_page` рисуется под ответом.
- [x] График: прогноз **vs факт** — `actual` добавлен в `serialize_inference_results`; classic и neiro
      возвращают тест-период; `ForecastChart` рисует серую пунктирную линию факта (estimate-режим).
- [x] Панель обучения — `TrainingResults` компонент подключён; все graduate-эндпоинты возвращают
      `serialize_training_results`; routing в `query_page` определяет тип по `best_model`.
- [x] Сезонная декомпозиция — `DecompositionChart` подключён; `/season_analytic/` возвращает
      `collect_decomposition_results` (trend/seasonal/resid); routing по `trend`-ключу.
- [x] Унифицировать HTTP-клиент — все сервисы переведены на `Instance` (axios); интерсептор
      разворачивает `response.data` и нормализует FastAPI-detail в читаемый `Error.message`.
- [x] Явный выбор операции в UI — `detectOp` показывает бейдж с именем операции; retail-шаблоны
      (4 кнопки) заполняют JSON; `GenericSeriesForm` расширен режимом `decompose`.
- [x] Вынести контент документации — `public/docs/api.md` (Markdown с таблицами + code blocks);
      `documentation_page.jsx` 710→55 строк, фетчит файл через `fetch` + показывает спиннер.
- [x] Error boundary + loading — `ErrorBoundary` оборачивает `RouterProvider`; `isLoading` блокирует
      кнопку "Send Query" и показывает индикатор; ошибки API попадают в responseText.

> ✅ Старт Этапа D проверен сборкой: `CI=true npm run build` → *Compiled successfully* с Plotly.
> График потребляет контракт `collect_results` из Этапа C. Остальные графики (факт, обучение,
> декомпозиция) требуют, чтобы соответствующие эндпоинты возвращали структурные данные.

## Этап E. Качество, упаковка, эксплуатация

- [x] Тесты библиотеки (pytest): 18 тестов — unit на data/metrics/serialize + e2e classic/neiro на CPU (маркер `slow`).
- [x] Линт: `ruff` проходит полным конфигом (E,F,I,UP,B,SIM) в library и backend; `eslint` во frontend.
- [x] CI (GitHub Actions): per-branch workflow'ы — library (ruff+pytest), backend (ruff), frontend (lint+build), dev (интеграция через сабмодули: линт+тесты+smoke бэкенда+сборка фронта).
- [ ] Артефакты (веса/графики) — в объектное хранилище (S3/Selectel, `script/selectel_cloud.py`).
- [x] Dockerfile (api + статика) + `docker-compose`; healthcheck-эндпоинт — реализовано
      (`backend/docker/`, `frontend/docker/`, `docker/docker-compose*.yaml`, `/server/health`).
- [ ] Асинхронные задачи обучения (очередь arq/RQ) со статусом/прогрессом для дашборда.

---

### Порядок и зависимости

- **A → B → C** — основной поток: сначала отвязать ядро, потом упаковать, потом
  сделать сервер тонким. **D** (дашборд) можно вести параллельно с C по согласованному
  контракту ответов API. **E** — сквозной (тесты/CI желательно подключить уже на A–B).
- Каждый шаг A–C проверяется запуском (нужны установленные Python и Node — в текущей
  среде их нет, поэтому правки идут срезами, которые верифицируются чтением).
