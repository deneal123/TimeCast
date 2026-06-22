"""Тест унифицированной подготовки данных в NeiroDataset (generic vs retail).

Проверяется только извлечение (date_id, series, exogenous) — обучение нейросети
требует GPU/данных и здесь не запускается.
"""
import pandas as pd

from timecast.data.neiro import NeiroDataset


def test_process_data_generic():
    df = pd.DataFrame({
        "target": [1.0, 2.0, 3.0],
        "date_id": [0, 1, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "price": [10, 11, 12],
        "promo": [0, 1, 0],
    })
    ds = NeiroDataset(dictidx={"feature_cols": ["price", "promo"]}, metadata=[])
    assert ds.feature_cols == ["price", "promo"]

    date_id, series, exog = ds.process_data(df)
    assert list(series.values) == [1.0, 2.0, 3.0]
    assert list(exog.columns) == ["price", "promo"]   # произвольные фичи
    assert list(series.index) == list(df["date"])
    assert len(exog) == 3


def test_process_data_retail():
    df = pd.DataFrame({
        "cnt": [5.0, 6.0],
        "date_id": [1, 2],
        "sell_price": [1, 2],
        "event_name": [0, 1],
        "event_type": [0, 0],
        "cashback": [0, 1],
    })
    ds = NeiroDataset(dictidx={"idx2date": {0: "2024-01-01", 1: "2024-01-02"}}, metadata=[])
    assert ds.feature_cols == ["sell_price", "event_name", "event_type", "cashback"]

    date_id, series, exog = ds.process_data(df)
    assert list(series.values) == [5.0, 6.0]
    assert list(exog.columns) == ["sell_price", "event_name", "event_type", "cashback"]
    assert list(series.index) == ["2024-01-01", "2024-01-02"]  # idx2date[date_id-1]
