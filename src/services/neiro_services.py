import asyncio
import timecast
from timecast.pydantic_models import EntryNeiroGraduatePipeline, EntryNeiroInferencePipeline
from typing import Dict
from src.utils.custom_logging import setup_logging
log = setup_logging()


async def neiro_graduate_pipeline(entry: EntryNeiroGraduatePipeline) -> Dict:
    await asyncio.to_thread(timecast.train_neiro, entry)
    log.info("Success graduate")
    return {"message": "Success graduate"}


async def neiro_inference_pipeline(entry: EntryNeiroInferencePipeline) -> Dict:
    pipeline = await asyncio.to_thread(timecast.infer_neiro, entry)
    results = timecast.collect_results(pipeline)
    log.info("Success inference")
    return {"message": "Success inference", "results": results}
