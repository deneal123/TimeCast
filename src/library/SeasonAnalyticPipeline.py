from src.library.pydantic_models import validate_with_pydantic, EntrySeasonAnalyticPipeline, EntryClassicDataset, EntryClassicProccess
from fastapi import HTTPException, status
from src.library.ClassicDataset import ClassicDataset
from src.library.ClassicProccess import ClassicProccess
from dataclasses import dataclass
from src import path_to_project
from env import Env, log
import os

env = Env()


@dataclass
class SeasonAnalyticPipeline:
    entry: EntrySeasonAnalyticPipeline

    def __post_init__(self):
        pass

    async def analyze(self):

        dataset = self.entry.Dataset

        shop_sales = os.path.join(path_to_project(), env.__getattr__("DATA_PATH"), "shop_sales.csv")
        shop_sales_dates = os.path.join(path_to_project(), env.__getattr__("DATA_PATH"), "shop_sales_dates.csv")
        shop_sales_prices = os.path.join(path_to_project(), env.__getattr__("DATA_PATH"), "shop_sales_prices.csv")

        if not os.path.exists(shop_sales) or not os.path.exists(shop_sales_dates) or not os.path.exists(
                shop_sales_prices):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CSV files not found")

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

        await self.classic_dataset.dataset()
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
        
        await self.classic_proccess.proccess()
