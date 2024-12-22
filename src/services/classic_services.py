from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntryClassicGraduatePipeline, EntryClassicInferencePipeline
from src.library.ClassicGraduatePipeline import ClassicGraduatePipeline
from src.library.ClassicInferencePipeline import ClassicInferencePipeline



def classic_graduate_pipeline(
        entry: EntryClassicGraduatePipeline
) -> None:

    classic_graduate_pipeline = validate_with_pydantic(EntryClassicGraduatePipeline)(ClassicGraduatePipeline)(
        entry=entry
    )


def classic_inference_pipeline(
        entry: EntryClassicInferencePipeline
) -> None:

    classic_inference_pipeline = validate_with_pydantic(EntryClassicInferencePipeline)(ClassicInferencePipeline)(
        entry=entry
    )
