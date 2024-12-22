from .pydantic_models import validate_with_pydantic, EntryNeiroGraduatePipeline, EntryClassicDataset, EntryNeiroGraduate
from .ClassicDataset import ClassicDataset
from .NeiroGraduate import NeiroGraduate
from dataclasses import dataclass



@dataclass
class NeiroGraduatePipeline:
    entry: EntryNeiroGraduatePipeline

    def __post_init__(self):

        dataset = self.entry.Dataset

        self.classic_dataset = validate_with_pydantic(EntryClassicDataset)(ClassicDataset)(
            entry={
                "store_id": dataset.StoreID,
                "shop_sales": dataset.ShopSales,
                "shop_sales_dates": dataset.ShopSalesDates,
                "shop_sales_prices": dataset.ShopSalesPrices,
                "plots": False,
                "save_plots": dataset.SavePlots,
                "save_path_plots": dataset.SavePathPlots
            }
        )

        self.classic_dataset.dataset()
        dictidx = self.classic_dataset.dictidx
        dictmerge = self.classic_dataset.dictmerge

        graduate = self.entry.Graduate

        self.neiro_graduate = validate_with_pydantic(EntryNeiroGraduate)(NeiroGraduate)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": graduate.DictSeasonal,
                "dictmodels": graduate.DictModels,
                "sen_len": graduate.SeqLen,
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