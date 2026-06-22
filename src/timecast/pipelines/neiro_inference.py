from dataclasses import dataclass

from timecast.inference.neiro import NeiroInference
from timecast.pipelines._retail import load_retail_dataset
from timecast.schemas import (
    EntryNeiroInference,
    EntryNeiroInferencePipeline,
    validate_with_pydantic,
)


@dataclass
class NeiroInferencePipeline:
    entry: EntryNeiroInferencePipeline

    def __post_init__(self):
        pass

    def inference(self):

        self.classic_dataset, dictidx, dictmerge = load_retail_dataset(self.entry.Dataset)

        inference = self.entry.Inference

        self.neiro_inference = validate_with_pydantic(EntryNeiroInference)(NeiroInference)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": inference.DictSeasonal,
                "dictmodels": inference.DictModels,
                "future_or_estimate": inference.FutureOrEstimate,
                "seq_len": inference.SeqLen,
                "path_to_weights": inference.PathWeights,
                "plots": inference.Plots,
                "save_plots": inference.SavePlots,
                "save_path_plots": inference.SavePathPlots,
                "pin_memory": inference.PinMemory,
                "num_workers": inference.NumWorkers,
                "use_device": inference.UseDevice
            }
        )
        
        self.neiro_inference.inference()
