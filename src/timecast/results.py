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
    if isinstance(value, str):
        return value
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
                "actual": _to_jsonable(r.get("actual")),
            }
        if item_out:
            out[str(item_id)] = item_out
    return out


def serialize_training_results(results: dict) -> dict:
    """results[item_id][period] = {best_model, best_param, best_rmse, best_r2} → JSON-безопасный dict.

    Для дашборда обучения: лучшая модель и её метрики по каждому периоду.
    """
    out = {}
    for item_id, periods in (results or {}).items():
        if not isinstance(periods, dict):
            continue
        item_out = {}
        for period, r in periods.items():
            if not isinstance(r, dict):
                continue
            item_out[str(period)] = {
                "best_model": _to_jsonable(r.get("best_model")),
                "best_param": _to_jsonable(r.get("best_param")),
                "best_rmse": _to_jsonable(r.get("best_rmse")),
                "best_r2": _to_jsonable(r.get("best_r2")),
            }
        if item_out:
            out[str(item_id)] = item_out
    return out


def serialize_decomposition_results(results: dict) -> dict:
    """results[item_id][period] = {trend, seasonal, resid} (pandas-серии) → списки.

    Для дашборда сезонной декомпозиции (аддитивная модель).
    """
    out = {}
    for item_id, periods in (results or {}).items():
        if not isinstance(periods, dict):
            continue
        item_out = {}
        for period, comp in periods.items():
            if not isinstance(comp, dict):
                continue
            item_out[str(period)] = {
                "trend": _to_jsonable(comp.get("trend")),
                "seasonal": _to_jsonable(comp.get("seasonal")),
                "resid": _to_jsonable(comp.get("resid")),
            }
        if item_out:
            out[str(item_id)] = item_out
    return out


def collect_results(pipeline) -> dict:
    """Достаёт и сериализует результаты инференса из пайплайна (classic/neiro)."""
    inference = getattr(pipeline, "classic_inference", None) or getattr(pipeline, "neiro_inference", None)
    results = getattr(inference, "results", None) if inference is not None else None
    return serialize_inference_results(results)


def collect_training_results(pipeline) -> dict:
    """Достаёт и сериализует результаты обучения из пайплайна (classic/neiro graduate)."""
    graduate = getattr(pipeline, "classic_graduate", None) or getattr(pipeline, "neiro_graduate", None)
    results = getattr(graduate, "results", None) if graduate is not None else None
    return serialize_training_results(results)


def collect_decomposition_results(pipeline) -> dict:
    """Достаёт и сериализует декомпозицию из пайплайна сезонной аналитики."""
    process = getattr(pipeline, "classic_process", None)
    results = getattr(process, "results", None) if process is not None else None
    return serialize_decomposition_results(results)
