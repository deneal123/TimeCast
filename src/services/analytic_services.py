import asyncio
import timecast
from timecast import EntrySeasonAnalyticPipeline
from typing import Dict
from src.utils.custom_logging import setup_logging
log = setup_logging()


async def season_analytic_pipeline(entry: EntrySeasonAnalyticPipeline) -> Dict:
    # CPU-bound работа библиотеки выполняется в отдельном потоке, чтобы не блокировать loop.
    pipeline = await asyncio.to_thread(timecast.season_analytic, entry)
    results = timecast.collect_decomposition_results(pipeline)
    log.info("Success analyse")
    return {"message": "Success analyse", "results": results}
