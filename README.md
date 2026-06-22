# timecast

Переиспользуемая Python-библиотека для **обучения и инференса моделей прогнозирования
**любого** временного ряда с фичами**. Ядро платформы TimeCast — не зависит от веб-слоя
(FastAPI), окружения и фиксированных путей.

## Возможности

- **Обобщённый ряд** — прогноз ЛЮБОГО временного ряда из tidy-CSV (`time, target,
  [series_id], feature_*`); не привязан к домену. Розничный формат (магазины/товары) —
  частный случай, распознаётся автоматически.
- **Classic** — статистические модели `sktime` (AutoARIMA, AutoREG, AutoETS, Prophet, TBATS).
- **Neiro** — нейросеть `iTransformer` (PyTorch) с кастомным лоссом; число каналов
  (`num_variates`) выводится автоматически из числа фич.
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

### Обобщённый ряд (любой временной ряд)

Достаточно tidy-CSV: колонка времени, целевая колонка, опционально `series_id` и фичи.
Колонки можно указать явно или положиться на автоопределение.

```python
from timecast import (configure_paths, train_classic_series, infer_classic_series,
                      train_neiro_series, infer_neiro_series,
                      serialize_training_results, serialize_inference_results)

configure_paths(weights_classic_dir="./weights_classic")

dataset = {
    "source": "series.csv",       # tidy-CSV
    "time_col": "time",
    "target_col": "target",
    "series_id_col": "id",          # необязательно (один ряд — можно опустить)
    "feature_cols": ["promo"],      # необязательно (иначе выведутся автоматически)
}

# Classic: обучение и инференс
cg = train_classic_series(dataset, {
    "dictseasonal": {"week": 7},                       # любые ключи периодов
    "models_params": {"AUTOARIMA": [3, 3, 0, 0, 1, 1, "week"]},
})
print(serialize_training_results(cg.results))          # {id: {week: {best_model, best_rmse, ...}}}

ci = infer_classic_series(dataset, {"dictseasonal": {"week": 7}, "future_or_estimate": "estimate"})
print(serialize_inference_results(ci.results))         # {id: {week: {rmse, r2, pred}}}

# Neiro: симметрично (num_variates вычисляется сам)
# train_neiro_series(dataset, {"dictseasonal": {...}, "dictmodels": {...}, "seq_len": 30, ...})
# infer_neiro_series(dataset, {"dictseasonal": {...}, "dictmodels": {...}, ...})
```

### Розничный домен (магазины/товары)

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
| `train_classic_series(dataset, graduate)` / `infer_classic_series(dataset, inference)` | **обобщённый ряд**, classic |
| `train_neiro_series(dataset, graduate)` / `infer_neiro_series(dataset, inference)` | **обобщённый ряд**, нейросеть |
| `season_analytic(entry)` | сезонная декомпозиция |
| `train_classic(entry)` / `infer_classic(entry)` | retail-домен, статистические модели |
| `train_neiro(entry)` / `infer_neiro(entry)` | retail-домен, нейросеть |
| `collect_results(pipeline)` / `serialize_inference_results(...)` | JSON-безопасные результаты |
| `serialize_training_results(...)` / `serialize_decomposition_results(...)` | результаты обучения / декомпозиции |
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
│   ├── data/             # датасеты: timeseries (обобщённый tidy-CSV), classic, neiro
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
