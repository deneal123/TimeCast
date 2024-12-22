from .pydantic_models import validate_with_pydantic, EntryNeiroInferencePipeline, EntryClassicDataset, EntryNeiroInference
from .ClassicDataset import ClassicDataset
from .NeiroInference import NeiroInference
from dataclasses import dataclass




@dataclass
class NeiroInferencePipeline:
    entry: EntryNeiroInferencePipeline

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

        inference = self.entry.Inference

        self.neiro_inference = validate_with_pydantic(EntryNeiroInference)(NeiroInference)(
            entry={
                "dictidx": dictidx,
                "dictmerge": dictmerge,
                "dictseasonal": inference.DictSeasonal,
                "dictmodels":inference.DictModels,
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