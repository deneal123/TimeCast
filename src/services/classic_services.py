from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntryClassicGraduatePipeline, EntryClassicInferencePipeline
from src.library.ClassicGraduatePipeline import ClassicGraduatePipeline
from src.library.ClassicInferencePipeline import ClassicInferencePipeline
from typing import Dict


async def classic_graduate_pipeline(
        entry: EntryClassicGraduatePipeline
) -> Dict:

    classic_graduate_pipeline = validate_with_pydantic(EntryClassicGraduatePipeline)(ClassicGraduatePipeline)(
        entry=entry
    )
    await classic_graduate_pipeline.graduate()

    return {"message": "Success graduate"}


async def classic_inference_pipeline(
        entry: EntryClassicInferencePipeline
) -> Dict:

    classic_inference_pipeline = validate_with_pydantic(EntryClassicInferencePipeline)(ClassicInferencePipeline)(
        entry=entry
    )
    await classic_inference_pipeline.inference()

    return {"message": "Success inference"}
