from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntryNeiroGraduatePipeline, EntryNeiroInferencePipeline
from src.library.NeiroGraduatePipeline import NeiroGraduatePipeline
from src.library.NeiroInferencePipeline import NeiroInferencePipeline
from typing import Dict


async def neiro_graduate_pipeline(
        entry: EntryNeiroGraduatePipeline
) -> Dict:

    neiro_graduate_pipeline = validate_with_pydantic(EntryNeiroGraduatePipeline)(NeiroGraduatePipeline)(
        entry=entry
    )
    await neiro_graduate_pipeline.graduate()

    return {"message": "Success graduate"}


async def neiro_inference_pipeline(
        entry: EntryNeiroInferencePipeline
) -> Dict:

    neiro_inference_pipeline = validate_with_pydantic(EntryNeiroInferencePipeline)(NeiroInferencePipeline)(
        entry=entry
    )
    await neiro_inference_pipeline.inference()

    return {"message": "Success inference"}
