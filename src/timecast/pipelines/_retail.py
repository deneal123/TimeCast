"""Общая загрузка retail-данных для пайплайнов (3 CSV → ClassicDataset).

Раньше этот блок (~25 строк) дублировался во всех четырёх retail-пайплайнах
(classic/neiro × graduate/inference).
"""
import os

from timecast.config import get_paths
from timecast.exceptions import DataNotFoundError
from timecast.data.classic import ClassicDataset
from timecast.schemas import validate_with_pydantic, EntryClassicDataset


def load_retail_dataset(dataset):
    """Строит ClassicDataset из 3 retail-CSV и возвращает (instance, dictidx, dictmerge).

    dataset — EntryClassicDataset-секция запроса (StoreID/SavePlots/SavePathPlots).
    Бросает DataNotFoundError, если какого-то CSV нет.
    """
    paths = get_paths()
    shop_sales = os.path.join(paths.data_dir, paths.sales_csv)
    shop_sales_dates = os.path.join(paths.data_dir, paths.sales_dates_csv)
    shop_sales_prices = os.path.join(paths.data_dir, paths.sales_prices_csv)

    if not (os.path.exists(shop_sales) and os.path.exists(shop_sales_dates)
            and os.path.exists(shop_sales_prices)):
        raise DataNotFoundError("CSV files not found")

    classic_dataset = validate_with_pydantic(EntryClassicDataset)(ClassicDataset)(
        entry={
            "store_id": dataset.StoreID,
            "shop_sales": shop_sales,
            "shop_sales_dates": shop_sales_dates,
            "shop_sales_prices": shop_sales_prices,
            "plots": False,
            "save_plots": dataset.SavePlots,
            "save_path_plots": dataset.SavePathPlots,
        }
    )
    classic_dataset.dataset()
    return classic_dataset, classic_dataset.dictidx, classic_dataset.dictmerge
