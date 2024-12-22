from dataclasses import dataclass # Для классов
from .pydantic_models import EntryClassicProccess
import pandas as pd # Работа с таблицами
import statsmodels.api as sm
import os # Для работы с файловой системой
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.stattools as ts
from scipy import stats, signal
from statsmodels.graphics import tsaplots
import duckdb as db
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
import numpy as np # Для работы с массивами
import matplotlib.pyplot as plt
from .create_dir import create_directories_if_not_exist
from tqdm import tqdm # Для красивого отображения процесса работы метода
from copy import deepcopy # Позволяет делать полную копию данных (не ссылаясь на обьект)
from .custom_logging import setup_logging
log = setup_logging(debug=False)




@dataclass
class ClassicProccess:
    entry: EntryClassicProccess

    def __post_init__(self):

        self.dictmerge = self.entry.DictMerge
        self.dictdecompose = self.entry.DictDecompose
        self.lower_bound_factor = self.entry.RemoveBound['lower_bound_factor']
        self.upper_bound_factor = self.entry.RemoveBound['upper_bound_factor']
        self.plots = self.entry.Plots
        self.save_plots = self.entry.SavePlots
        self.save_path_plots = self.entry.SavePathPlots

        if self.save_plots:
            if self.save_path_plots is None:
                self.save_path_plots = "./plots"
            create_directories_if_not_exist([self.save_path_plots])
        
        self.dictstructparam = {
            "series": None,
            "redis": None,
            "trend": None,
            "seasonal": None
        }

    def proccess(self):
        for item_id, item in self.dictmerge.items():
            date_id = item['date_id']
            series = item['cnt']
            self.decompose(series, item_id)
            self.visualise(item_id,
                           series,
                           self.dictstructparam["redis"],
                           self.dictstructparam["trend"],
                           self.dictstructparam["seasonal"],
                           date_id)

    def remove_outliers(self, series: pd.DataFrame):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if series.std() < 0.1 or (series.max() - series.min()) < 0.1 or Q1 == 0 and Q3 == 0 or IQR <= 1:
            return series
        lower_bound = Q1 - self.lower_bound_factor * IQR
        upper_bound = Q3 + self.upper_bound_factor * IQR
        series = series.apply(lambda x: series.median() if x < lower_bound or x > upper_bound else x)
        return series

    def decompose_series(self, series: pd.Series, period: int, model: str):
        return seasonal_decompose(
            series,
            model=model,
            filt=None,
            period=period,
            two_sided=True,
            extrapolate_trend=1
        )

    def decompose(self, series: pd.DataFrame, item_id):
        dictredis = {}
        dicttrend = {}
        dictseasonal = {}
        
        for index, (seasonality, value) in enumerate(self.dictdecompose.items()):
            sales_add = self.decompose_series(self.remove_outliers(series), value, 'additive')
            sales_mult = self.decompose_series(self.remove_outliers(series), value, 'multiplicative')
    
            dictredis[f'{seasonality}'] = [self.remove_outliers(sales_add.resid.fillna(0)),
                                           self.remove_outliers(sales_mult.resid.fillna(0))]
            dicttrend[f'{seasonality}'] = [sales_add.trend.fillna(0), sales_mult.trend.fillna(0)]
            dictseasonal[f'{seasonality}'] = [sales_add.seasonal.fillna(0), sales_mult.seasonal.fillna(0)]
            
        self.dictstructparam["redis"] = dictredis
        self.dictstructparam["trend"] = dicttrend
        self.dictstructparam["seasonal"] = dictseasonal
        self.dictstructparam["series"] = series

    def visualise(self, item: str, series: pd.DataFrame, redis: dict, trend: dict, seasonal: dict, date_id: pd.DataFrame):
        fixed_fontsize = 10  # Фиксированный размер шрифта для заголовков
        for seasonality, _ in self.dictdecompose.items():
            # Получение данных для текущей сезонности
            redis_add, redis_mult = redis[seasonality]
            trend_add, trend_mult = trend[seasonality]
            seasonal_add, seasonal_mult = seasonal[seasonality]

            # Рассчитываем минимальные и максимальные значения для оси ординат
            min_value = min(series.min(), redis_add.min(), redis_mult.min(), trend_add.min(), trend_mult.min(), seasonal_add.min(), seasonal_mult.min())
            max_value = max(series.max(), redis_add.max(), redis_mult.max(), trend_add.max(), trend_mult.max(), seasonal_add.max(), seasonal_mult.max())

            # Создаем фигуру с 6 сабплотами
            fig, ax = plt.subplots(3, 2, figsize=(16, 12))

            # 1. Исходный временной ряд
            ax[0, 0].plot(date_id, series, label='Original Series', color='blue')
            ax[0, 0].set_title(f'{item} - Original Series ({seasonality})', fontsize=fixed_fontsize)
            ax[0, 0].set_xlabel('Date', fontsize=fixed_fontsize)
            ax[0, 0].set_ylabel('Count', fontsize=fixed_fontsize)
            ax[0, 0].grid(True)
            ax[0, 0].legend()

            # 2. Остатки (add и mult на одном графике)
            ax[0, 1].plot(date_id, redis_add, label='Residuals (Additive)', color='green')
            ax[0, 1].plot(date_id, redis_mult, label='Residuals (Multiplicative)', color='orange')
            ax[0, 1].set_title(f'{item} - Residuals ({seasonality})', fontsize=fixed_fontsize)
            ax[0, 1].set_xlabel('Date', fontsize=fixed_fontsize)
            ax[0, 1].set_ylabel('Residuals', fontsize=fixed_fontsize)
            ax[0, 1].grid(True)
            ax[0, 1].legend()

            # 3. Тренд (add и mult на одном графике)
            ax[1, 0].plot(date_id, trend_add, label='Trend', color='blue')
            ax[1, 0].set_title(f'{item} - Trend ({seasonality})', fontsize=fixed_fontsize)
            ax[1, 0].set_xlabel('Date', fontsize=fixed_fontsize)
            ax[1, 0].set_ylabel('Trend', fontsize=fixed_fontsize)
            ax[1, 0].grid(True)
            ax[1, 0].legend()

            # 4. Сезонность (add и mult на одном графике)
            ax[1, 1].plot(date_id, seasonal_add, label='Seasonality (Additive)', color='red')
            ax[1, 1].plot(date_id, seasonal_mult, label='Seasonality (Multiplicative)', color='brown')
            ax[1, 1].set_title(f'{item} - Seasonality ({seasonality})', fontsize=fixed_fontsize)
            ax[1, 1].set_xlabel('Date', fontsize=fixed_fontsize)
            ax[1, 1].set_ylabel('Seasonality', fontsize=fixed_fontsize)
            ax[1, 1].grid(True)
            ax[1, 1].legend()

            # 5. Автокорреляция (add и mult на одном графике)
            tsaplots.plot_acf(redis_add, ax=ax[2, 0], lags=40, alpha=0.05, color='red', label='ACF (Additive)')
            tsaplots.plot_acf(redis_mult, ax=ax[2, 0], lags=40, alpha=0.05, color='orange', label='ACF (Multiplicative)')
            ax[2, 0].set_ylim(-0.75, 0.75)
            ax[2, 0].legend()
            
            # 6. Частичная автокорреляция (add и mult на одном графике)
            tsaplots.plot_pacf(redis_add, ax=ax[2, 1], lags=40, alpha=0.05, color='red', label='ACF (Additive)')
            tsaplots.plot_pacf(redis_mult, ax=ax[2, 1], lags=40, alpha=0.05, color='orange', label='ACF (Multiplicative)')
            ax[2, 1].set_ylim(-0.75, 0.75)
            ax[2, 1].legend()
        
            # Показать графики
            plt.tight_layout()
            if self.plots:
                plt.show()
            if self.save_plots:
                fig.savefig(os.path.join(self.save_path_plots, f"classic_proccess_{item}_{seasonality}"), dpi=100)
            plt.close(fig)