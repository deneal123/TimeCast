from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntrySeasonAnalyticPipeline
from src.library.SeasonAnalyticPipeline import SeasonAnalyticPipeline
from typing import Dict
from src.utils.custom_logging import setup_logging
log = setup_logging()


async def season_analytic_pipeline(
        entry: EntrySeasonAnalyticPipeline
) -> Dict:

    season_analytic_pipeline = validate_with_pydantic(EntrySeasonAnalyticPipeline)(SeasonAnalyticPipeline)(
        entry=entry
    )
    await season_analytic_pipeline.analyze()

    log.info("Success analyse")
    return {"message": "Success analyse"}
