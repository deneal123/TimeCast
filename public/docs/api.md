# TimeCast API Reference

---

## SeasonAnalytic — пример запроса

```json
{
  "dataset": { "store_id": "STORE_1" },
  "proccess": {
    "dictdecompose": { "week": 7, "month": 30, "quater": 90 },
    "remove_bound": { "lower_bound_factor": 5, "upper_bound_factor": 5 }
  }
}
```

---

## ClassicGraduate — пример запроса

```json
{
  "dataset": { "store_id": "STORE_1" },
  "graduate": {
    "dictseasonal": { "week": 7, "month": 30, "quater": 90 }
  },
  "models_params": { "AUTOREG": [7, "week"] }
}
```

---

## ClassicInference — пример запроса

```json
{
  "dataset": { "store_id": "STORE_1" },
  "inference": {
    "dictseasonal": { "week": 7, "month": 30, "quater": 90 },
    "future_or_estimate": "estimate"
  }
}
```

---

## NeiroGraduate — пример запроса

```json
{
  "dataset": { "store_id": "STORE_1" },
  "graduate": {
    "dictseasonal": { "week": 7, "month": 30, "quater": 90, "year": 365 },
    "dictmodels": {
      "IF": {
        "depth": 6, "dim": 512, "dim_head": 64, "heads": 8,
        "num_tokens_per_variate": 1, "num_variates": 7,
        "use_reversible_instance_norm": true
      }
    },
    "use_device": "cuda"
  }
}
```

---

## NeiroInference — пример запроса

```json
{
  "dataset": { "store_id": "STORE_1" },
  "inference": {
    "dictseasonal": { "week": 7, "month": 30, "quater": 90 },
    "dictmodels": {
      "IFFT": {
        "depth": 6, "dim": 256, "dim_head": 64, "heads": 8,
        "num_tokens_per_variate": 1, "num_variates": 7,
        "use_reversible_instance_norm": true
      }
    },
    "future_or_estimate": "estimate",
    "use_device": "cuda"
  }
}
```

---

## ClassicDataset

```python
ClassicDataset(entry: EntryClassicDataset(
    store_id: str,               # Store ID (STORE_1)
    shop_sales: pd.DataFrame,    # Sales data per store
    shop_sales_dates: pd.DataFrame,   # Holiday data
    shop_sales_prices: pd.DataFrame,  # Product price data
    plots: bool,                 # Plot graphs
    save_plots: bool,            # Save graphs
    save_path_plots: str,        # Path to graph directory
)) -> dict[str, pd.DataFrame]    # 15 datasets, one per time series
```

Загружает и готовит датафреймы для каждого продукта.

| Метод | Описание |
|-------|----------|
| `dataset` | Метод инициализации |
| `fetch_data` | Загружает CSV-файлы и строит таблицы через duckdb |
| `merge_data` | SQL-запрос для слияния трёх таблиц |
| `split_merge` | Разбивает общую таблицу на 15 рядов (по item_id) |
| `merge` | Возвращает merged dataset со всеми item |
| `dictidx` | Словарь param2idx / idx2param |
| `dictmerge` | Словарь датасетов, по одному на каждый ряд |
| `visualise` | Графики продаж и цены по времени |

---

## ClassicProccess

```python
ClassicProccess(entry: EntryClassicProccess(
    dictmerge: dict[str, pd.DataFrame],  # Merged datasets per series
    dictdecompose: dict[str, int],       # Seasonal periods
    removebound: dict[str, float],       # Outlier boundary factors
    plots: bool,
    saveplots: bool,
    savepathplots: str,
)) -> dict[str, dict[str, pd.DataFrame]]
```

Выполняет предобработку, удаление выбросов, декомпозицию и визуализацию.

| Метод | Описание |
|-------|----------|
| `proccess` | Основной метод: декомпозиция + визуализация по каждому ряду |
| `remove_outliers` | Удаление выбросов (IQR) |
| `decompose` | Декомпозиция на тренд / сезонность / остатки (аддитивная + мультипликативная) |
| `visualise` | Графики исходных и обработанных данных + ACF |

---

## ClassicModel (ABC)

```python
class ClassicModel(ABC):
    name_model: str = None
    _registered_models: dict = {}
```

Зарегистрированные модели: **AUTOARIMA**, **AUTOREG**, **AUTOETS**, **PROPHET**, **TBATS**.

| Метод | Описание |
|-------|----------|
| `fit_pred` | Обучение + предсказание + метрика |
| `score` | Метрика на тест-выборке |
| `param` | Параметры модели |
| `pred` | Предсказание на тест |
| `save` | Сохранение модели на диск |
| `register_model` | Регистрация в `_registered_models` |
| `get_model_classes` | Список зарегистрированных классов |
| `create_model` | Создание объекта по имени |
| `from_pretrained` | Загрузка предобученной модели |

---

## ClassicGraduate

```python
ClassicGraduate(entry: EntryClassicGraduate(
    dictidx: dict[str, pd.Series],
    dictmerge: dict[str, pd.DataFrame],
    dictseasonal: dict[str, int],   # Горизонты предсказания
    modelsparams: dict[str, tuple], # Параметры моделей
))
```

| Метод | Описание |
|-------|----------|
| `graduate` | Основной метод отбора лучших моделей |
| `train_model` | Подбор параметров, модели и окна для каждого ряда |
| `calc_optimum` | Оценка предсказания + метрики |

---

## ClassicInference

```python
ClassicInference(entry: EntryClassicInference(
    dictidx, dictmerge, dictseasonal,
    plots, save_plots, save_path_plots,
    save_path_weights: str,
))
```

| Метод | Описание |
|-------|----------|
| `inference` | Предсказание, оценка, визуализация |
| `load_models` | Загрузка весов из файловой системы |
| `visualise` | Графики факт/прогноз |
| `evaluate` | Оценка качества + сохранение метрик |
| `calc_feature` | Предсказание + проверка данных |

---

## NeiroDataset / get_datasets

```python
NeiroDataset(dictidx, metadata) -> torch.Dataset
get_datasets(dictidx, dictmerge, item_id, test_size,
             period, seq_len, step_length, seed,
             future_or_estimate_or_train) -> torch.Dataset
```

| Метод | Описание |
|-------|----------|
| `proccess_data` | Обработка train/test данных |
| `__len__` | Число срезов в датасете |
| `__getitem__` | Итерация по данным |
| `get_datasets` | Разбивка данных (estimate / future / train) |

---

## NeiroGraduate / collate_fn / CustomLoss

```python
NeiroGraduate(entry: EntryNeiroGraduate(
    dictidx, dictmerge, dictseasonal,
    dictmodels: dict[str, dict],
    seq_len: int, test_size: float, step_length: int,
    path_to_weights: str, use_device: str,
    start_learning_rate: float, batch_size: int,
    num_workers: int, pin_memory: bool,
    num_epochs: int, name_optimizer: str, seed: int,
))

collate_fn(batch, minmax_resid, minmax_trend, minmax_season,
           minmax_sellprice, minmax_series,
           pdata: bool, without_test: bool) -> dict

class CustomLoss:
    beta: float = 1.0       # SmoothL1Loss
    delta: float = 0.5      # Штраф за отрицательные значения
    gamma: float = 0.1      # CosineEmbeddingLoss
    cosine_margin: float = 0.0
    special_penalty: float = 1.0  # Штраф для тренда
```

| Метод | Описание |
|-------|----------|
| `graduate` | Инициализация обучения |
| `get_loaders` | Инициализация DataLoader |
| `get_models` | Инициализация архитектур (IFFT / IF) |
| `get_opt_crit_sh` | Оптимизатор + функция потерь |
| `load_checkpoint` | Загрузка чекпоинта |
| `train_models` | Обучение |
| `evaluate_models` | Оценка на тест-выборке |

---

## NeiroInference

```python
NeiroInference(entry: EntryNeiroInference(
    dictidx, dictmerge, dictseasonal, dictmodels,
    future_or_estimate: str,   # 'estimate' | 'future'
    seq_len: int,
    path_to_weights: str,
    plots: bool, save_plots: bool, save_path_plots: str,
    use_device: str,           # 'cpu' | 'cuda'
    num_workers: int, pin_memory: bool,
))
```

| Метод | Описание |
|-------|----------|
| `inference` | Предсказание, оценка, визуализация |
| `load_models` | Загрузка весов (.pt) из файловой системы |
| `visualise` | Графики факт/прогноз |
| `evaluate` | Оценка метрик (MAE, RMSE, R²) |

---

## Generic TimeSeries — произвольный временной ряд

Эндпоинты для работы с CSV без привязки к розничному домену.

### POST /server/timeseries_graduate/

```json
{
  "dataset": {
    "source": "path/to/data.csv",
    "time_col": "date",
    "target_col": "sales",
    "series_id_col": "store"
  },
  "graduate": {
    "dictseasonal": { "week": 7, "month": 30 },
    "models_params": { "AUTOREG": [7, "week"] }
  }
}
```

### POST /server/timeseries_neiro_graduate/

```json
{
  "dataset": { "source": "data.csv", "time_col": "date", "target_col": "value" },
  "graduate": {
    "dictseasonal": { "week": 7 },
    "dictmodels": { "IF": { "depth": 4, "dim": 256, "dim_head": 64, "heads": 4,
                             "num_tokens_per_variate": 1, "num_variates": 3 } }
  }
}
```

### POST /server/timeseries_inference/  и  /server/timeseries_neiro_inference/

```json
{
  "dataset": { "source": "data.csv", "time_col": "date", "target_col": "value" },
  "inference": {
    "dictseasonal": { "week": 7 },
    "future_or_estimate": "estimate"
  }
}
```

---

## Асинхронная очередь обучения

Каждый `*/graduate/` эндпоинт имеет парный `*/graduate/queue/`, который возвращает `task_id`
немедленно и выполняет обучение фоном.

### POST /server/classic_graduate/queue/

Тело запроса — идентично `POST /server/classic_graduate/`. Ответ:

```json
{ "task_id": "a3f2b1c04d8e", "status": "pending" }
```

Аналогично для `/neiro_graduate/queue/`, `/timeseries_graduate/queue/`,
`/timeseries_neiro_graduate/queue/`.

### GET /server/tasks/

Список всех задач (новые первыми):

```json
[
  {
    "task_id": "a3f2b1c04d8e",
    "operation": "classic_graduate",
    "status": "done",
    "created_at": "2026-06-25T12:00:00Z",
    "updated_at": "2026-06-25T12:03:00Z",
    "result": { "results": { ... } },
    "error": null
  }
]
```

### GET /server/tasks/{task_id}

Статус и результат конкретной задачи. Поля `status`: `pending` → `running` → `done` | `failed`.

Если настроено S3-хранилище (переменные `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`,
`S3_SECRET_KEY`), поле `result` дополняется массивом артефактов:

```json
{
  "result": {
    "results": { ... },
    "artifacts": [
      { "key": "a3f2b1c04d8e/weights/classic/model.pkl", "size": 24576 },
      { "key": "a3f2b1c04d8e/plots/forecast.png",         "size": 102400 }
    ]
  }
}
```
