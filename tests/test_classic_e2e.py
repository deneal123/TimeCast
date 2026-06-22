"""End-to-end проверка обобщённого classic-обучения на синтетическом ряду (CPU).

Ловит регрессии доменной привязки: произвольные ключи dictseasonal (не только
week/month/quater) и произвольные фичи (не sell_price). Реальный fit sktime —
тест помечен медленным; инференс не запускается (ProcessPoolExecutor нестабилен в CI).
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import timecast


@pytest.mark.slow
def test_train_classic_series_generic_arbitrary_period_and_features():
    d = tempfile.mkdtemp()
    n = 70
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    target = 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 7) + np.random.RandomState(0).normal(0, 0.5, n)
    df = pd.DataFrame({
        "time": dates,
        "id": "A",
        "target": target,
        "promo": (t % 7 == 0).astype(int),   # произвольная фича (не sell_price)
    })
    csv = os.path.join(d, "series.csv")
    df.to_csv(csv, index=False)

    timecast.configure_paths(weights_classic_dir=os.path.join(d, "w"))
    cg = timecast.train_classic_series(
        {"source": csv, "time_col": "time", "target_col": "target",
         "series_id_col": "id", "feature_cols": ["promo"]},
        # произвольный ключ периода "biweek" — не из week/month/quater
        {"dictseasonal": {"biweek": 7}, "models_params": {"AUTOARIMA": [2, 2, 0, 0, 1, 1, "week"]}},
    )
    results = timecast.serialize_training_results(getattr(cg, "results", {}))
    assert "A" in results
    assert set(results["A"].keys()) == {"biweek"}            # ключ периода сохранён как есть
    entry = results["A"]["biweek"]
    assert entry["best_model"] == "AUTOARIMA"
    assert isinstance(entry["best_rmse"], float)
    # веса сохранены под произвольным ключом периода
    assert any("biweek" in f for f in os.listdir(os.path.join(d, "w")))
