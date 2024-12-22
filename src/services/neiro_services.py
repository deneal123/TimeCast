from src.library.pydantic_models import validate_with_pydantic
from src.library.pydantic_models import EntryNeiroGraduatePipeline, EntryNeiroInferencePipeline
from src.library.NeiroGraduatePipeline import NeiroGraduatePipeline
from src.library.NeiroInferencePipeline import NeiroInferencePipeline



def neiro_graduate_pipeline(
        entry: EntryNeiroGraduatePipeline
) -> None:

    neiro_graduate_pipeline = validate_with_pydantic(EntryNeiroGraduatePipeline)(NeiroGraduatePipeline)(
        entry=entry
    )


def neiro_inference_pipeline(
        entry: EntryNeiroInferencePipeline
) -> None:

    neiro_inference_pipeline = validate_with_pydantic(EntryNeiroInferencePipeline)(NeiroInferencePipeline)(
        entry=entry
    )
