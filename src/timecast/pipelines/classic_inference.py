from dataclasses import dataclass

from timecast.inference.classic import ClassicInference
from timecast.pipelines._retail import load_retail_dataset
from timecast.schemas import (
    EntryClassicInference,
    EntryClassicInferencePipeline,
    validate_with_pydantic,
)


@dataclass
class ClassicInferencePipeline:
    entry: EntryClassicInferencePipeline

    def __post_init__(self):
        pass

    def inference(self):

        self.classic_dataset, dictidx, dictmerge = load_retail_dataset(self.entry.Dataset)

        inference = self.entry.Inference

        self.classic_inference = validate_with_pydantic(EntryClassicInference)(ClassicInference)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": inference.DictSeasonal,
                "future_or_estimate": inference.FutureOrEstimate,
                "plots": inference.Plots,
                "save_plots": inference.SavePlots,
                "save_path_plots": inference.SavePathPlots,
                "save_path_weights": inference.SavePathWeights
            }
        )
        
        self.classic_inference.inference()
