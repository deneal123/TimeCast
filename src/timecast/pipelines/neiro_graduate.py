from timecast.schemas import validate_with_pydantic, EntryNeiroGraduatePipeline, EntryNeiroGraduate
from timecast.training.neiro import NeiroGraduate
from timecast.pipelines._retail import load_retail_dataset
from dataclasses import dataclass


@dataclass
class NeiroGraduatePipeline:
    entry: EntryNeiroGraduatePipeline

    def __post_init__(self):
        pass

    def graduate(self):

        self.classic_dataset, dictidx, dictmerge = load_retail_dataset(self.entry.Dataset)

        graduate = self.entry.Graduate

        self.neiro_graduate = validate_with_pydantic(EntryNeiroGraduate)(NeiroGraduate)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": graduate.DictSeasonal,
                "dictmodels": graduate.DictModels,
                "seq_len": graduate.SeqLen,
                "test_size": graduate.TestSize,
                "step_length": graduate.StepLen,
                "path_to_weights": graduate.PathWeights,
                "use_device": graduate.UseDevice,
                "start_learning_rate": graduate.StartLerningRate,
                "batch_size": graduate.BatchSize,
                "num_workers": graduate.NumWorkers,
                "pin_memory": graduate.PinMemory,
                "num_epochs": graduate.NumEpochs,
                "name_optimizer": graduate.NameOptimizer,
                "seed": graduate.Seed
            }
        )
        
        self.neiro_graduate.graduate()
