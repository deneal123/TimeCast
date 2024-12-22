from .pydantic_models import validate_with_pydantic, EntryClassicInferencePipeline, EntryClassicDataset, EntryClassicInference
from .ClassicDataset import ClassicDataset
from .ClassicInference import ClassicInference
from dataclasses import dataclass



@dataclass
class ClassicInferencePipeline:
    entry: EntryClassicInferencePipeline

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