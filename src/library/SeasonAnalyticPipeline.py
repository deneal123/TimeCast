from .pydantic_models import validate_with_pydantic, EntrySeasonAnalyticPipeline, EntryClassicDataset, EntryClassicProccess
from .ClassicDataset import ClassicDataset
from .ClassicProccess import ClassicProccess
from dataclasses import dataclass



@dataclass
class SeasonAnalyticPipeline:
    entry: EntrySeasonAnalyticPipeline

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

        proccess = self.entry.Proccess

        self.classic_proccess = validate_with_pydantic(EntryClassicProccess)(ClassicProccess)(
            entry={
                "dictmerge": dictmerge,
                "dictdecompose": proccess.DictDecompose,
                "remove_bound": proccess.RemoveBound,
                "plots": proccess.Plots,
                "save_plots": proccess.SavePlots,
                "save_path_plots": proccess.SavePathPlots
            }
        )
        
        self.classic_proccess.proccess()        