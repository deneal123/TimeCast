"""Сервис обобщённого временного ряда (tidy CSV, без привязки к домену)."""
import asyncio

import timecast

from src.utils.custom_logging import setup_logging

log = setup_logging()


async def timeseries_graduate_pipeline(dataset: dict, graduate: dict) -> dict:
    # Обучение classic-моделей на произвольном ряду; CPU-bound — в отдельном потоке.
    cg = await asyncio.to_thread(timecast.train_classic_series, dataset, graduate)
    results = timecast.serialize_training_results(getattr(cg, "results", {}))
    log.info("Success timeseries graduate")
    return {"message": "Success graduate", "results": results}


async def timeseries_inference_pipeline(dataset: dict, inference: dict) -> dict:
    # Инференс classic-моделей на произвольном ряду; CPU-bound — в отдельном потоке.
    ci = await asyncio.to_thread(timecast.infer_classic_series, dataset, inference)
    results = timecast.serialize_inference_results(getattr(ci, "results", {}))
    log.info("Success timeseries inference")
    return {"message": "Success inference", "results": results}


async def timeseries_neiro_graduate_pipeline(dataset: dict, graduate: dict) -> dict:
    # Обучение нейросети на произвольном ряду (GPU-bound — в отдельном потоке).
    ng = await asyncio.to_thread(timecast.train_neiro_series, dataset, graduate)
    results = timecast.serialize_training_results(getattr(ng, "results", {}))
    log.info("Success timeseries neiro graduate")
    return {"message": "Success graduate", "results": results}


async def timeseries_neiro_inference_pipeline(dataset: dict, inference: dict) -> dict:
    # Инференс нейросети на произвольном ряду (GPU-bound — в отдельном потоке).
    ni = await asyncio.to_thread(timecast.infer_neiro_series, dataset, inference)
    results = timecast.serialize_inference_results(getattr(ni, "results", {}))
    log.info("Success timeseries neiro inference")
    return {"message": "Success inference", "results": results}
