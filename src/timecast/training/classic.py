import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from tqdm import tqdm

from timecast._internal.dirs import create_directories_if_not_exist
from timecast._internal.logging import setup_logging
from timecast._internal.prepare import prepare_series_exog
from timecast.config import get_paths
from timecast.models.classic import ClassicModel
from timecast.schemas import EntryClassicGraduate

log = setup_logging()



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
            self.path_to_weights = Path(get_paths().weights_classic_dir)
        else:
            self.path_to_weights = Path(os.path.join(self.path_to_weights))
        create_directories_if_not_exist([self.path_to_weights])

        self.minmax_resid = MinMaxScaler()
        self.minmax_trend = MinMaxScaler()
        self.minmax_season = MinMaxScaler()
        self.minmax_sellprice = MinMaxScaler()
        self.minmax_series = MinMaxScaler()

        self.results = {}

    _prepare = staticmethod(prepare_series_exog)

    def graduate(self):
        for item_id, params in self.dictmerge.items():
            log.info(f"Process {item_id}")
            series, exogenous = self._prepare(params, self.dictidx)

            log.info("Training")
            best_models, best_params, best_rmses, best_r2s = self.train_model(series, exogenous, item_id)

            # Произвольные периоды сезонности — берём фактические ключи dictseasonal,
            # а не жёстко week/month/quater (поддержка любого ряда).
            self.results[item_id] = {
                period: {'best_model': best_models[period],
                         'best_param': best_params[period],
                         'best_rmse': best_rmses[period],
                         'best_r2': best_r2s[period]}
                for period in self.dictseasonal
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
                    for _index, (train_indices, test_indices) in enumerate(splitter.split(series)):

                        train = series.iloc[train_indices]
                        test = series.iloc[test_indices]

                        try:
                            rmse, r2, pred, tss = self.calc_optimum(train, test, exogenous, model, param, period,
                                                                    item_id)
                        except ValueError:
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

        train[train == 0.01] = 0.000001
        test[test == 0.01] = 0.000001
        tss = ClassicModel.create_model(model, train, test, exogenous)
        tss.fit(*param)
        pred = tss.pred()
        rmse = np.sqrt(mean_squared_error(test, pred))
        r2 = r2_score(test, pred)

        return rmse, r2, pred, tss
