"""Конфигурация путей библиотеки TimeCast.

Библиотека не читает переменные окружения и не знает про структуру проекта.
Базовые директории для данных и артефактов берутся отсюда; по умолчанию они
**относительные** (от текущей рабочей директории), что удобно для автономного
использования (ноутбук, скрипт).

Веб-слой (FastAPI) при старте вызывает `configure_paths(...)` и подставляет
абсолютные пути из `.env`, сохраняя прежнее поведение сервера.
"""
from dataclasses import dataclass


@dataclass
class LibraryPaths:
    #: Директория с входными CSV (shop_sales*.csv).
    data_dir: str = "data"
    #: Куда сохранять графики.
    plots_dir: str = "plots"
    #: Веса классических (статистических) моделей.
    weights_classic_dir: str = "weights_classic"
    #: Веса нейросетевых моделей.
    weights_neiro_dir: str = "weights_neiro"
    #: Имя CSV с продажами (внутри data_dir).
    sales_csv: str = "shop_sales.csv"
    #: Имя CSV с календарём/датами.
    sales_dates_csv: str = "shop_sales_dates.csv"
    #: Имя CSV с ценами.
    sales_prices_csv: str = "shop_sales_prices.csv"


#: Период декомпозиции по умолчанию (в днях). Ключи — идентификаторы периодов.
#: NB: ключ "quater" сохранён с исторической опечаткой — он используется как ключ
#: в других местах кода; переименование в "quarter" — отдельная задача.
DEFAULT_SEASONAL = {"week": 7, "month": 30, "quater": 90}


_PATHS = LibraryPaths()


def get_paths() -> LibraryPaths:
    """Текущая конфигурация путей библиотеки."""
    return _PATHS


def configure_paths(*, data_dir: str = None, plots_dir: str = None,
                    weights_classic_dir: str = None, weights_neiro_dir: str = None) -> None:
    """Переопределить базовые пути (None — оставить прежнее значение)."""
    if data_dir is not None:
        _PATHS.data_dir = data_dir
    if plots_dir is not None:
        _PATHS.plots_dir = plots_dir
    if weights_classic_dir is not None:
        _PATHS.weights_classic_dir = weights_classic_dir
    if weights_neiro_dir is not None:
        _PATHS.weights_neiro_dir = weights_neiro_dir
