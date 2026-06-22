"""Высокоуровневый публичный API библиотеки TimeCast.

Тонкие синхронные обёртки над пайплайнами: принимают тело запроса (dict в форме
соответствующей `Entry*`-модели), валидируют его и запускают пайплайн. Возвращают
экземпляр пайплайна — у него доступны промежуточные данные/результаты
(`dictmerge`, `dictidx` и т.п.).

Пути к данным/весам/графикам настраиваются через `configure_paths(...)`
(см. `timecast.config`). Пример::

    from timecast import configure_paths, season_analytic

    configure_paths(data_dir="/data/store1")
    season_analytic({
        "dataset": {"store_id": "STORE_1"},
        "proccess": {
            "dictdecompose": {"week": 7, "month": 30, "quarter": 90},
            "remove_bound": {"lower_bound_factor": 5, "upper_bound_factor": 5},
        },
    })
"""
from typing import Any

from timecast.data.timeseries import TimeSeriesDataset
from timecast.inference.classic import ClassicInference
from timecast.inference.neiro import NeiroInference
from timecast.pipelines.classic_graduate import ClassicGraduatePipeline
from timecast.pipelines.classic_inference import ClassicInferencePipeline
from timecast.pipelines.neiro_graduate import NeiroGraduatePipeline
from timecast.pipelines.neiro_inference import NeiroInferencePipeline
from timecast.pipelines.season_analytic import SeasonAnalyticPipeline
from timecast.schemas import (
    EntryClassicGraduate,
    EntryClassicGraduatePipeline,
    EntryClassicInference,
    EntryClassicInferencePipeline,
    EntryNeiroGraduate,
    EntryNeiroGraduatePipeline,
    EntryNeiroInference,
    EntryNeiroInferencePipeline,
    EntrySeasonAnalyticPipeline,
    EntryTimeSeriesDataset,
    validate_with_pydantic,
)
from timecast.training.classic import ClassicGraduate
from timecast.training.neiro import NeiroGraduate


def season_analytic(entry: dict[str, Any]) -> SeasonAnalyticPipeline:
    """Сезонная декомпозиция рядов (тренд/сезон/остаток)."""
    pipeline = validate_with_pydantic(EntrySeasonAnalyticPipeline)(SeasonAnalyticPipeline)(entry=entry)
    pipeline.analyze()
    return pipeline


def train_classic(entry: dict[str, Any]) -> ClassicGraduatePipeline:
    """Обучение классических (статистических) моделей sktime."""
    pipeline = validate_with_pydantic(EntryClassicGraduatePipeline)(ClassicGraduatePipeline)(entry=entry)
    pipeline.graduate()
    return pipeline


def infer_classic(entry: dict[str, Any]) -> ClassicInferencePipeline:
    """Инференс классическими моделями."""
    pipeline = validate_with_pydantic(EntryClassicInferencePipeline)(ClassicInferencePipeline)(entry=entry)
    pipeline.inference()
    return pipeline


def train_neiro(entry: dict[str, Any]) -> NeiroGraduatePipeline:
    """Обучение нейросетевой модели (iTransformer)."""
    pipeline = validate_with_pydantic(EntryNeiroGraduatePipeline)(NeiroGraduatePipeline)(entry=entry)
    pipeline.graduate()
    return pipeline


def infer_neiro(entry: dict[str, Any]) -> NeiroInferencePipeline:
    """Инференс нейросетевой моделью."""
    pipeline = validate_with_pydantic(EntryNeiroInferencePipeline)(NeiroInferencePipeline)(entry=entry)
    pipeline.inference()
    return pipeline


def train_classic_series(dataset: dict[str, Any], graduate: dict[str, Any]) -> ClassicGraduate:
    """Обучение классических моделей на ПРОИЗВОЛЬНОМ временном ряду (tidy CSV, без привязки к домену).

    dataset: поля EntryTimeSeriesDataset — source, [time_col, target_col, series_id_col, feature_cols].
    graduate: dictseasonal, models_params, [save_path_weights].
    Пример::

        train_classic_series(
            {"source": "series.csv", "series_id_col": "id"},
            {"dictseasonal": {"week": 7, "month": 30, "quater": 90},
             "models_params": {"AUTOARIMA": (3, 3, 0, 0, 1, 1, "week")}},
        )
    """
    ds = validate_with_pydantic(EntryTimeSeriesDataset)(TimeSeriesDataset)(entry=dataset)
    ds.dataset()
    cg = validate_with_pydantic(EntryClassicGraduate)(ClassicGraduate)(entry={
        "dictidx": ds.dictidx,
        "dictmerge": ds.dictmerge,
        "dictseasonal": graduate["dictseasonal"],
        "models_params": graduate["models_params"],
        "save_path_weights": graduate.get("save_path_weights"),
    })
    cg.graduate()
    return cg


def infer_classic_series(dataset: dict[str, Any], inference: dict[str, Any]) -> ClassicInference:
    """Инференс классическими моделями на ПРОИЗВОЛЬНОМ временном ряду (tidy CSV, без привязки к домену).

    dataset: поля EntryTimeSeriesDataset — source, [time_col, target_col, series_id_col, feature_cols].
    inference: dictseasonal, [future_or_estimate, save_path_weights, plots, save_plots, save_path_plots].
    Веса должны быть обучены на ряду с той же структурой (см. train_classic_series).
    """
    ds = validate_with_pydantic(EntryTimeSeriesDataset)(TimeSeriesDataset)(entry=dataset)
    ds.dataset()
    ci = validate_with_pydantic(EntryClassicInference)(ClassicInference)(entry={
        "dictidx": ds.dictidx,
        "dictmerge": ds.dictmerge,
        "dictseasonal": inference["dictseasonal"],
        "future_or_estimate": inference.get("future_or_estimate", "estimate"),
        "save_path_weights": inference.get("save_path_weights"),
        "plots": inference.get("plots", False),
        "save_plots": inference.get("save_plots", True),
        "save_path_plots": inference.get("save_path_plots"),
    })
    ci.inference()
    return ci


def train_neiro_series(dataset: dict[str, Any], graduate: dict[str, Any]) -> NeiroGraduate:
    """Обучение нейросети (iTransformer) на ПРОИЗВОЛЬНОМ временном ряду (tidy CSV).

    dataset: поля EntryTimeSeriesDataset — source, [time_col, target_col, series_id_col, feature_cols].
    graduate: параметры EntryNeiroGraduate (dictseasonal, dictmodels, seq_len, test_size, ...) КРОМЕ
    dictidx/dictmerge — они выводятся из dataset. num_variates вычисляется автоматически (3 + число фич).
    """
    ds = validate_with_pydantic(EntryTimeSeriesDataset)(TimeSeriesDataset)(entry=dataset)
    ds.dataset()
    entry = {**graduate, "dictidx": ds.dictidx, "dictmerge": ds.dictmerge}
    ng = validate_with_pydantic(EntryNeiroGraduate)(NeiroGraduate)(entry=entry)
    ng.graduate()
    return ng


def infer_neiro_series(dataset: dict[str, Any], inference: dict[str, Any]) -> NeiroInference:
    """Инференс нейросетью на ПРОИЗВОЛЬНОМ временном ряду (tidy CSV).

    dataset: поля EntryTimeSeriesDataset.
    inference: параметры EntryNeiroInference (dictseasonal, dictmodels, seq_len, future_or_estimate, ...)
    КРОМЕ dictidx/dictmerge — они выводятся из dataset.
    """
    ds = validate_with_pydantic(EntryTimeSeriesDataset)(TimeSeriesDataset)(entry=dataset)
    ds.dataset()
    entry = {**inference, "dictidx": ds.dictidx, "dictmerge": ds.dictmerge}
    ni = validate_with_pydantic(EntryNeiroInference)(NeiroInference)(entry=entry)
    ni.inference()
    return ni
