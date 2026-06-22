"""Тест унифицированной подготовки данных в ClassicGraduate (generic vs retail)."""
import pandas as pd

from timecast.training.classic import ClassicGraduate


def test_prepare_generic_format():
    # Обобщённый формат TimeSeriesDataset: target + feature_cols + date.
    params = pd.DataFrame({
        "target": [1.0, 2.0, 3.0],
        "date_id": [0, 1, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "price": [10, 11, 12],
        "promo": [0, 1, 0],
    })
    dictidx = {"feature_cols": ["price", "promo"]}
    series, exog = ClassicGraduate._prepare(params, dictidx)

    assert list(series) == [1.0, 2.0, 3.0]
    assert list(exog.columns) == ["price", "promo"]
    assert len(exog) == 3
    assert list(series.index) == list(params["date"])
    assert list(exog.index) == list(params["date"])


def test_prepare_retail_format():
    # Доменный retail-формат: cnt + фикс. фичи + date_id (1-based).
    params = pd.DataFrame({
        "cnt": [5.0, 6.0],
        "date_id": [1, 2],
        "sell_price": [1, 2],
        "event_name": [0, 1],
        "event_type": [0, 0],
        "cashback": [0, 1],
    })
    dictidx = {"idx2date": {0: "2024-01-01", 1: "2024-01-02"}}
    series, exog = ClassicGraduate._prepare(params, dictidx)

    assert list(series) == [5.0, 6.0]
    assert list(exog.columns) == ["sell_price", "event_name", "event_type", "cashback"]
    assert list(series.index) == ["2024-01-01", "2024-01-02"]  # idx2date[date_id-1]
