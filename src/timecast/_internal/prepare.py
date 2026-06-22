"""Унифицированное извлечение (series, exogenous) из записи dictmerge.

Поддерживает два формата:
- обобщённый (TimeSeriesDataset): target + feature_cols + date;
- доменный retail (ClassicDataset): cnt + фикс. фичи + date_id (1-based).
Формат определяется наличием 'feature_cols' в dictidx.
"""
import pandas as pd

from timecast._internal.features import RETAIL_FEATURES


def prepare_series_exog(params, dictidx):
    if "feature_cols" in dictidx:
        cols = list(dictidx["feature_cols"])
        index = list(params["date"])
        series = params["target"].copy()
        series.index = index
        exogenous = params[cols].copy()
        exogenous.index = index
        return series, exogenous

    index = [dictidx["idx2date"][i - 1] for i in params["date_id"]]
    series = params["cnt"].copy()
    series.index = index
    cols = {}
    for name in RETAIL_FEATURES:
        col = params[name].copy()
        col.index = index
        cols[name] = col
    return series, pd.DataFrame(cols)
