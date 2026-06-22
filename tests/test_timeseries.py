"""Юнит-тесты обобщённого загрузчика временных рядов (без привязки к домену)."""
import pandas as pd
import pytest

from timecast import TimeSeriesDataset, EntryTimeSeriesDataset
from timecast.exceptions import DataNotFoundError


def _csv(tmp_path, df, name="series.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_multi_series_with_features(tmp_path):
    path = _csv(tmp_path, pd.DataFrame({
        "time": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        "series_id": ["A", "A", "B", "B"],
        "target": [10.0, 12.0, 20.0, 22.0],
        "price": [1.5, 1.6, 2.0, 2.1],     # числовая фича
        "promo": ["no", "yes", "yes", "no"],  # категориальная → кодируется
    }))
    ds = TimeSeriesDataset(entry=EntryTimeSeriesDataset(source=path, series_id_col="series_id"))
    ds.dataset()

    dm = ds.dictmerge
    assert set(dm.keys()) == {"A", "B"}
    a = dm["A"]
    assert list(a["target"]) == [10.0, 12.0]
    assert {"target", "date_id", "date", "price", "promo"}.issubset(a.columns)
    assert list(a["date_id"]) == [0, 1]            # глобальный индекс дат
    assert str(a["promo"].dtype).startswith(("int", "float"))  # категория закодирована

    di = ds.dictidx
    assert di["feature_cols"] == ["price", "promo"]
    assert "promo" in di["feature_encoders"] and "price" not in di["feature_encoders"]
    assert len(di["date2idx"]) == 2
    assert set(di["series2idx"]) == {"A", "B"}


def test_single_series_inferred_features(tmp_path):
    path = _csv(tmp_path, pd.DataFrame({
        "time": ["2024-01-01", "2024-01-02"], "target": [1.0, 2.0], "f": [5, 6],
    }))
    ds = TimeSeriesDataset(entry=EntryTimeSeriesDataset(source=path))
    ds.dataset()
    assert list(ds.dictmerge.keys()) == ["default"]   # нет series_id → один ряд
    assert ds.dictidx["feature_cols"] == ["f"]


def test_explicit_feature_subset(tmp_path):
    path = _csv(tmp_path, pd.DataFrame({
        "time": ["2024-01-01", "2024-01-02"], "target": [1.0, 2.0],
        "a": [1, 2], "b": [3, 4],
    }))
    ds = TimeSeriesDataset(entry=EntryTimeSeriesDataset(source=path, feature_cols=["a"]))
    ds.dataset()
    assert ds.dictidx["feature_cols"] == ["a"]
    assert "b" not in ds.dictmerge["default"].columns


def test_missing_target_raises(tmp_path):
    path = _csv(tmp_path, pd.DataFrame({"time": ["2024-01-01"], "value": [1.0]}))
    ds = TimeSeriesDataset(entry=EntryTimeSeriesDataset(source=path))
    with pytest.raises(DataNotFoundError):
        ds.dataset()
