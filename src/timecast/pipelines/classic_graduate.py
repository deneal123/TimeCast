from dataclasses import dataclass

from timecast.pipelines._retail import load_retail_dataset
from timecast.schemas import (
    EntryClassicGraduate,
    EntryClassicGraduatePipeline,
    validate_with_pydantic,
)
from timecast.training.classic import ClassicGraduate


@dataclass
class ClassicGraduatePipeline:
    entry: EntryClassicGraduatePipeline

    def __post_init__(self):
        pass

    def graduate(self):

        self.classic_dataset, dictidx, dictmerge = load_retail_dataset(self.entry.Dataset)

        graduate = self.entry.Graduate

        self.classic_graduate = validate_with_pydantic(EntryClassicGraduate)(ClassicGraduate)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": graduate.DictSeasonal,
                "models_params": graduate.ModelsParams,
                "save_path_weights": graduate.SavePathWeights
            }
        )

        self.classic_graduate.graduate()
