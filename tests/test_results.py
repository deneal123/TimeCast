"""Юнит-тесты сериализации результатов инференса."""
import numpy as np
import pandas as pd

from timecast.results import serialize_inference_results, collect_results


class _FakeModel:
    pass


def _sample_results():
    return {
        "STORE_1_FOODS_1_001": {
            # classic-style: rmse + r2 + pandas-серия с NaN + несериализуемая модель
            "week": {"rmse": np.float64(1.5), "r2": 0.8,
                     "pred": pd.Series([1.0, 2.0, np.nan]), "model": _FakeModel()},
            # neiro-style: без r2, ndarray-предсказание, model — строка
            "month": {"rmse": 2.0, "pred": np.array([3.0, 4.0]), "model": "AUTOARIMA"},
            # незаполненный период — int из dictseasonal → отбрасывается
            "quater": 90,
        }
    }


def test_serialize_is_json_safe():
    out = serialize_inference_results(_sample_results())
    item = out["STORE_1_FOODS_1_001"]
    # NaN → None, pandas/ndarray → списки
    assert item["week"]["pred"] == [1.0, 2.0, None]
    assert item["week"]["rmse"] == 1.5
    assert item["week"]["r2"] == 0.8
    # отсутствующий r2 → None; модель не попадает в вывод
    assert item["month"]["r2"] is None
    assert item["month"]["pred"] == [3.0, 4.0]
    assert "model" not in item["month"]
    # незаполненный период отброшен
    assert "quater" not in item


def test_serialize_handles_empty():
    assert serialize_inference_results(None) == {}
    assert serialize_inference_results({}) == {}


def test_collect_results_from_pipeline():
    class _Inference:
        results = _sample_results()

    class _Pipeline:
        classic_inference = _Inference()

    out = collect_results(_Pipeline())
    assert "STORE_1_FOODS_1_001" in out
    # неизвестный пайплайн → пустой результат, без падения
    assert collect_results(object()) == {}
