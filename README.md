# timecast

Переиспользуемая Python-библиотека для **обучения и инференса моделей прогнозирования
временных рядов** (прогноз спроса в розничной сети). Ядро платформы TimeCast — не
зависит от веб-слоя (FastAPI), окружения и фиксированных путей.

## Возможности

- **Classic** — статистические модели `sktime` (AutoARIMA, AutoREG, AutoETS, Prophet, TBATS).
- **Neiro** — нейросеть `iTransformer` (PyTorch) с кастомным лоссом.
- **Season** — сезонная декомпозиция (тренд / сезон / остаток).
- Загрузка/склейка данных через DuckDB; веса моделей — с Hugging Face Hub.

## Установка

```bash
# из каталога timecast/
uv sync                 # или: pip install -e .
```

> GPU-сборка torch берётся из индекса в `pyproject.toml` (`[[tool.uv.index]]`).
> Для CPU/CI уберите индекс или поставьте CPU-колесо torch.

## Использование

```python
from timecast import configure_paths, train_classic, infer_classic, season_analytic

# Пути к данным/весам/графикам (по умолчанию относительные).
configure_paths(data_dir="/data/store1")

# Сезонная декомпозиция
season_analytic({
    "dataset": {"store_id": "STORE_1"},
    "proccess": {
        "dictdecompose": {"week": 7, "month": 30, "quarter": 90},
        "remove_bound": {"lower_bound_factor": 5, "upper_bound_factor": 5},
    },
})

# Инференс → структурированный результат
pipeline = infer_classic({...})
from timecast import collect_results
results = collect_results(pipeline)   # {item_id: {period: {rmse, r2, pred}}}
```

## Публичный API

| Функция | Назначение |
|---------|------------|
| `season_analytic(entry)` | сезонная декомпозиция |
| `train_classic(entry)` / `infer_classic(entry)` | статистические модели |
| `train_neiro(entry)` / `infer_neiro(entry)` | нейросеть |
| `collect_results(pipeline)` / `serialize_inference_results(...)` | JSON-безопасные результаты |
| `configure_paths(...)` / `get_paths()` | конфигурация путей |
| `TimeCastError` / `DataNotFoundError` / `ValidationFailedError` | доменные исключения |

## Структура

```text
timecast/
├── pyproject.toml
├── src/timecast/         # пакет (src-layout), разбит по слоям
│   ├── __init__.py       # публичный API (re-export)
│   ├── api.py            # высокоуровневые функции (train_*/infer_*/season_analytic)
│   ├── config.py         # пути/константы · exceptions.py · results.py · schemas.py
│   ├── data/             # загрузка/датасеты (classic, neiro)
│   ├── models/           # модели (classic + реестр, loss)
│   ├── training/         # обучение (classic, neiro)
│   ├── inference/        # инференс (classic, neiro)
│   ├── season/           # сезонная декомпозиция (process)
│   ├── pipelines/        # оркестраторы (classic/neiro graduate·inference, season_analytic)
│   └── _internal/        # io, logging, dirs, utils
└── tests/
```

## Разработка

```bash
uv run pytest        # тесты
uv run ruff check    # линт
uv run mypy          # типы
```
