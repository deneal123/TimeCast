"""End-to-end проверка обобщённого neiro-конвейера на синтетике (CPU).

Гоняет train+infer iTransformer на CPU с крошечной конфигурацией. Ловит регрессии:
авто-`num_variates` (3 + число фич), произвольные фичи (не sell_price), совместимость
с torch (ReduceLROnPlateau без verbose). Помечен slow — реальное обучение сети.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import timecast

_IFFT = {"depth": 1, "dim": 16, "dim_head": 8, "heads": 2,
         "num_tokens_per_variate": 1, "use_reversible_instance_norm": True}


def _make_csv(d, n=160, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n)
    target = 20 + 0.05 * t + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "time": dates, "id": "S0", "target": target,
        "promo": (t % 7 == 0).astype(int),   # произвольная фича (не retail)
    })
    csv = os.path.join(d, "series.csv")
    df.to_csv(csv, index=False)
    return csv


@pytest.mark.slow
def test_neiro_series_generic_cpu_train_and_infer():
    d = tempfile.mkdtemp()
    csv = _make_csv(d)
    timecast.configure_paths(weights_neiro_dir=os.path.join(d, "wn"),
                             plots_dir=os.path.join(d, "p"))
    dataset = {"source": csv, "time_col": "time", "target_col": "target",
               "series_id_col": "id", "feature_cols": ["promo"]}
    common = {"dictseasonal": {"week": 7}, "dictmodels": {"IFFT": _IFFT},
              "seq_len": 30, "use_device": "cpu", "num_workers": 0}

    # Успешное обучение само проверяет авто-num_variates: при неверном числе
    # каналов (3 decomposition + 1 фича) cat/forward упали бы на несоответствии формы.
    ng = timecast.train_neiro_series(dataset, {**common, "test_size": 0.3,
                                              "batch_size": 4, "num_epochs": 1})
    assert "IFFT" in ng.models
    assert any(f.endswith(".pt") for f in os.listdir(os.path.join(d, "wn")))

    ni = timecast.infer_neiro_series(dataset, {**common, "future_or_estimate": "estimate",
                                              "save_plots": False, "plots": False})
    res = timecast.serialize_inference_results(getattr(ni, "results", {}))
    assert "S0" in res
    assert "week" in res["S0"]
