from timecast.schemas import validate_with_pydantic, EntrySeasonAnalyticPipeline, EntryClassicDataset, EntryClassicProcess
from timecast.exceptions import DataNotFoundError
from timecast.data.classic import ClassicDataset
from timecast.season.process import ClassicProcess
from dataclasses import dataclass
from timecast.config import get_paths
from timecast._internal.logging import setup_logging
import os

log = setup_logging()


@dataclass
class SeasonAnalyticPipeline:
    entry: EntrySeasonAnalyticPipeline

    def __post_init__(self):
        pass

    def analyze(self):

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

        proccess = self.entry.Process

        self.classic_process = validate_with_pydantic(EntryClassicProcess)(ClassicProcess)(
            entry={
                "dictmerge": dictmerge,
                "dictdecompose": proccess.DictDecompose,
                "remove_bound": proccess.RemoveBound,
                "plots": proccess.Plots,
                "save_plots": proccess.SavePlots,
                "save_path_plots": proccess.SavePathPlots
            }
        )
        
        self.classic_process.process()
