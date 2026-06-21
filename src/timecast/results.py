"""Сериализация результатов инференса в JSON-безопасный вид для дашборда.

`*.results` пайплайнов содержат несериализуемые объекты (модели, pandas-серии).
Здесь они приводятся к спискам/числам: метрики (rmse, r2) и предсказания (pred).
"""
import math

try:
    import numpy as np
except Exception:  # numpy всегда есть, но не делаем модуль хрупким
    np = None

try:
    import pandas as pd
except Exception:
    pd = None


def _to_jsonable(value):
    """Приводит число/серию/массив к JSON-безопасному значению (или None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None and isinstance(value, np.integer):
        return int(value)
    if np is not None and isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if pd is not None and isinstance(value, pd.Series):
        return [_to_jsonable(x) for x in value.tolist()]
    if np is not None and isinstance(value, np.ndarray):
        return [_to_jsonable(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(x) for x in value]
    # модели и прочие объекты не сериализуем
    return None


def serialize_inference_results(results: dict) -> dict:
    """results[item_id][period] = {rmse, r2?, pred, model} → JSON-безопасный dict.

    Дефолтные значения `dictseasonal` (int) и объекты моделей отбрасываются.
    """
    out = {}
    for item_id, periods in (results or {}).items():
        if not isinstance(periods, dict):
            continue
        item_out = {}
        for period, r in periods.items():
            if not isinstance(r, dict):
                continue  # незаполненный период (int из dictseasonal)
            item_out[str(period)] = {
                "rmse": _to_jsonable(r.get("rmse")),
                "r2": _to_jsonable(r.get("r2")),
                "pred": _to_jsonable(r.get("pred")),
            }
        if item_out:
            out[str(item_id)] = item_out
    return out


def collect_results(pipeline) -> dict:
    """Достаёт и сериализует результаты инференса из пайплайна (classic/neiro)."""
    inference = getattr(pipeline, "classic_inference", None) or getattr(pipeline, "neiro_inference", None)
    results = getattr(inference, "results", None) if inference is not None else None
    return serialize_inference_results(results)
