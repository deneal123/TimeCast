"""Сервис обобщённого временного ряда (tidy CSV, без привязки к домену)."""
import asyncio
import timecast
from typing import Dict
from src.utils.custom_logging import setup_logging
log = setup_logging()


async def timeseries_graduate_pipeline(dataset: Dict, graduate: Dict) -> Dict:
    # Обучение classic-моделей на произвольном ряду; CPU-bound — в отдельном потоке.
    cg = await asyncio.to_thread(timecast.train_classic_series, dataset, graduate)
    results = timecast.serialize_training_results(getattr(cg, "results", {}))
    log.info("Success timeseries graduate")
    return {"message": "Success graduate", "results": results}


async def timeseries_inference_pipeline(dataset: Dict, inference: Dict) -> Dict:
    # Инференс classic-моделей на произвольном ряду; CPU-bound — в отдельном потоке.
    ci = await asyncio.to_thread(timecast.infer_classic_series, dataset, inference)
    results = timecast.serialize_inference_results(getattr(ci, "results", {}))
    log.info("Success timeseries inference")
    return {"message": "Success inference", "results": results}
