from dataclasses import dataclass
from src.library.pydantic_models import EntryClassicGraduate
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.frequencies import to_offset
from itertools import product
from src.library.utils import check_fit, metrics_report
from sktime.forecasting.model_selection import SlidingWindowSplitter, ExpandingWindowSplitter
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from src.library.ClassicModel import ClassicModel
from sklearn.preprocessing import MinMaxScaler
from src.utils.create_dir import create_directories_if_not_exist
import json
from src.library.utils import dec_series
from pathlib import Path
from copy import deepcopy
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging
log, log_stream_handler = setup_logging()
env = Env()



@dataclass
class ClassicGraduate:
    entry: EntryClassicGraduate

    def __post_init__(self):

        self.dictidx = self.entry.DictIdx
        self.dictmerge = self.entry.DictMerge
        self.dictseasonal = self.entry.DictSeasonal
        self.models_params = self.entry.ModelsParams
        self.path_to_weights = self.entry.SavePathWeights

        if self.path_to_weights is None:
            self.path_to_weights = Path(os.path.join(path_to_project(), env.__getattr__("WEIGHTS_CLASSIC_PATH")))
        else:
            self.path_to_weights = Path(os.path.join(self.path_to_weights))
        create_directories_if_not_exist([self.path_to_weights])

        self.minmax_resid = MinMaxScaler()
        self.minmax_trend = MinMaxScaler()
        self.minmax_season = MinMaxScaler()
        self.minmax_sellprice = MinMaxScaler()
        self.minmax_series = MinMaxScaler()

        self.results = {}

    async def graduate(self):
        for item_id, params in self.dictmerge.items():

            series = params['cnt']
            date_id = params['date_id']
            sell_price = params['sell_price']
            event_name = params['event_name']
            event_type = params['event_type']
            cashback = params['cashback']
            series.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            sell_price.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            event_name.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            event_type.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            cashback.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            exogenous = pd.DataFrame({
                "sell_price": sell_price,
                "event_name": event_name,
                "event_type": event_type,
                "cashback": cashback
            })

            best_models, best_params, best_rmses, best_r2s = self.train_model(series, exogenous, item_id)

            self.results[item_id] = {

                'week': {'best_model': best_models['week'],
                         'best_param': best_params['week'],
                         'best_rmse': best_rmses['week'],
                         'best_r2': best_r2s['week']},
                'month': {'best_model': best_models['month'],
                          'best_param': best_params['month'],
                          'best_rmse': best_rmses['month'],
                          'best_r2': best_r2s['month']},
                'quater': {'best_model': best_models['quater'],
                           'best_param': best_params['quater'],
                           'best_rmse': best_rmses['quater'],
                           'best_r2': best_r2s['quater']}

            }

            log.info(f"ItemID {item_id}, {self.results[item_id]}")

            for key, value in self.results[item_id].items():
                for n, p in value.items():
                    if n == "best_model":
                        model = p
                        self.results[item_id][key]['best_model'] = model.name_model
                        try:
                            model.save(dir_path=self.path_to_weights,
                                       prefix=f"{item_id}_{key}_{model.name_model}",
                                       results=self.results[item_id][key])
                        except Exception as ex:
                            log.exception("", exc_info=ex)

    def train_model(self, series, exogenous, item_id):
        best_params = {}
        best_models = {}
        best_rmses = {}
        best_r2s = {}

        for period, val in self.dictseasonal.items():

            best_param = None
            best_model = None
            best_rmse = float('inf')
            best_r2 = float('-inf')

            for model, param in self.models_params.items():
                log.info(f"Model train: {model} for period {period}")

                if model == "AUTOARIMA" or model == "TBATS":
                    splitter = ExpandingWindowSplitter(fh=[i for i in range(val)],
                                                       initial_window=(len(series) - val - 1), step_length=1)
                elif model == "AUTOETS":
                    splitter = ExpandingWindowSplitter(fh=[i for i in range(val)],
                                                       initial_window=(len(series) - val - 5), step_length=1)
                else:
                    splitter = ExpandingWindowSplitter(fh=[i for i in range(val)],
                                                       initial_window=(len(series) - val - 31), step_length=1)

                rmse_values = []
                r2_values = []
                best_tss = None

                with tqdm(total=splitter.get_n_splits(series), unit="split") as pbar:
                    for index, (train_indices, test_indices) in enumerate(splitter.split(series)):

                        train = series.iloc[train_indices]
                        test = series.iloc[test_indices]

                        try:
                            rmse, r2, pred, tss = self.calc_optimum(train, test, exogenous, model, param, period,
                                                                    item_id)
                        except ValueError as ve:
                            continue

                        rmse = round(rmse, 3)
                        rmse_values.append(rmse)
                        r2 = round(r2, 3)
                        r2_values.append(r2)

                        if rmse < best_rmse:
                            best_tss = tss

                        # Обновляем прогресс-бар
                        pbar.update(1)
                        pbar.set_description(f"Processing {period} splits, RMSE: {rmse}, R2: {r2}")

                if rmse_values:
                    avg_rmse = np.mean(rmse_values)
                    avg_r2 = np.mean(r2_values)

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_r2 = avg_r2
                        best_model = best_tss
                        best_param = best_tss.param()

                    log.info(f"Model {model}, period {period}:\n"
                             f"Average -> RMSE: {avg_rmse} / R2: {avg_r2}")

            best_params[f'{period}'] = best_param
            best_models[f'{period}'] = best_model
            best_rmses[f'{period}'] = best_rmse
            best_r2s[f'{period}'] = best_r2

        return best_models, best_params, best_rmses, best_r2s

    @staticmethod
    def calc_optimum(train, test, exogenous, model, param, ses, item_id):

        # train_len = len(train)
        # series = pd.concat([train, test])

        # resid, trend, season = dec_series(series, 7, 'additive') # additive

        # resid[resid == 0.01] = 0.00001
        # trend[trend == 0.01] = 0.00001
        # season[season == 0.01] = 0.00001

        # Нормализуем на объединенных данных
        # resid = pd.Series(
        #     self.minmax_resid.fit_transform(resid.values.reshape(-1, 1)).flatten(),
        #     name='resid', index=resid.index
        # )
        # trend = pd.Series(
        #     self.minmax_trend.fit_transform(trend.values.reshape(-1, 1)).flatten(),
        #     name='trend', index=trend.index
        # )
        # season = pd.Series(
        #     self.minmax_season.fit_transform(season.values.reshape(-1, 1)).flatten(),
        #     name='season', index=season.index
        # )
        # exogenous['sell_price'] = self.minmax_sellprice.fit_transform(
        #          exogenous['sell_price'].values.reshape(-1, 1)
        #      ).flatten()

        # resid_train, resid_test = resid.iloc[:train_len], resid.iloc[train_len:]
        # trend_train, trend_test = trend.iloc[:train_len], trend.iloc[train_len:]
        # season_train, season_test = season.iloc[:train_len], season.iloc[train_len:]

        # train = pd.concat([resid_train, trend_train, season_train], axis=1)
        # test = pd.concat([resid_test, trend_test, season_test], axis=1)

        # series = pd.Series(
        #     self.minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(),
        #     name='series', index=series.index
        # )
        # train, test = series.iloc[:train_len], series.iloc[train_len:]     

        train[train == 0.01] = 0.000001
        test[test == 0.01] = 0.000001
        tss = ClassicModel.create_model(model, train, test, exogenous)
        tss.fit(*param)
        pred = tss.pred()
        rmse = np.sqrt(mean_squared_error(test, pred))
        r2 = r2_score(test, pred)

        return rmse, r2, pred, tss
