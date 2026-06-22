from timecast.schemas import validate_with_pydantic, EntryNeiroInferencePipeline, EntryClassicDataset, EntryNeiroInference
from timecast.exceptions import DataNotFoundError
from timecast.data.classic import ClassicDataset
from timecast.inference.neiro import NeiroInference
from dataclasses import dataclass
from timecast.config import get_paths
import os





@dataclass
class NeiroInferencePipeline:
    entry: EntryNeiroInferencePipeline

    def __post_init__(self):
        pass

    def inference(self):

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

        inference = self.entry.Inference

        self.neiro_inference = validate_with_pydantic(EntryNeiroInference)(NeiroInference)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": inference.DictSeasonal,
                "dictmodels": inference.DictModels,
                "future_or_estimate": inference.FutureOrEstimate,
                "sen_len": inference.SeqLen,
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
