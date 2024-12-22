from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntrySeasonAnalyticPipeline
from src.library.SeasonAnalyticPipeline import SeasonAnalyticPipeline



def season_analytic_pipeline(
        entry: EntrySeasonAnalyticPipeline
) -> None:

    season_analytic_pipeline = validate_with_pydantic(EntrySeasonAnalyticPipeline)(SeasonAnalyticPipeline)(
        entry=entry
    )
