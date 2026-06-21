import asyncio
import timecast
from timecast.pydantic_models import EntryClassicGraduatePipeline, EntryClassicInferencePipeline
from typing import Dict
from src.utils.custom_logging import setup_logging
log = setup_logging()


async def classic_graduate_pipeline(entry: EntryClassicGraduatePipeline) -> Dict:
    await asyncio.to_thread(timecast.train_classic, entry)
    log.info("Success graduate")
    return {"message": "Success graduate"}


async def classic_inference_pipeline(entry: EntryClassicInferencePipeline) -> Dict:
    pipeline = await asyncio.to_thread(timecast.infer_classic, entry)
    results = timecast.collect_results(pipeline)
    log.info("Success inference")
    return {"message": "Success inference", "results": results}
