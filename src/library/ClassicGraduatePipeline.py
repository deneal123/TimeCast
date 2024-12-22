from .pydantic_models import validate_with_pydantic, EntryClassicGraduatePipeline, EntryClassicDataset, EntryClassicGraduate
from .ClassicDataset import ClassicDataset
from .ClassicGraduate import ClassicGraduate
from dataclasses import dataclass


@dataclass
class ClassicGraduatePipeline:
    entry: EntryClassicGraduatePipeline

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