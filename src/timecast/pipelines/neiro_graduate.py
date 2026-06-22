from timecast.schemas import validate_with_pydantic, EntryNeiroGraduatePipeline, EntryClassicDataset, EntryNeiroGraduate
from timecast.exceptions import DataNotFoundError
from timecast.data.classic import ClassicDataset
from timecast.training.neiro import NeiroGraduate
from dataclasses import dataclass
from timecast.config import get_paths
import os




@dataclass
class NeiroGraduatePipeline:
    entry: EntryNeiroGraduatePipeline

    def __post_init__(self):
        pass

    def graduate(self):

        dataset = self.entry.Dataset

        shop_sales = os.path.join(get_paths().data_dir, get_paths().sales_csv)
        shop_sales_dates = os.path.join(get_paths().data_dir, get_paths().sales_dates_csv)
        shop_sales_prices = os.path.join(get_paths().data_dir, get_paths().sales_prices_csv)

        if not os.path.exists(shop_sales) or not os.path.exists(shop_sales_dates) or not os.path.exists(
                shop_sales_prices):
            raise DataNotFoundError("CSV files not found")

        self.classic_dataset = validate_with_pydantic(EntryClassicDataset)(ClassicDataset)(
            entry={
                "store_id": dataset.StoreID,
                "shop_sales": shop_sales,
                "shop_sales_dates": shop_sales_dates,
                "shop_sales_prices": shop_sales_prices,
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
