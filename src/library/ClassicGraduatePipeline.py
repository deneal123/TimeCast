from src.library.pydantic_models import validate_with_pydantic, EntryClassicGraduatePipeline, EntryClassicDataset, \
    EntryClassicGraduate
from fastapi import HTTPException, status
from src.library.ClassicDataset import ClassicDataset
from src.library.ClassicGraduate import ClassicGraduate
from dataclasses import dataclass
from src import path_to_project
from env import Env
import os

env = Env()


@dataclass
class ClassicGraduatePipeline:
    entry: EntryClassicGraduatePipeline

    def __post_init__(self):
        pass

    async def graduate(self):

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

        await self.classic_graduate.graduate()
