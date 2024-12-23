from dataclasses import dataclass
from src.library.pydantic_models import EntryClassicInference
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from sktime.split import SingleWindowSplitter
from src.library.ClassicModel import ClassicModel
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
from src.library.utils import dec_series
from src.utils.create_dir import create_directories_if_not_exist
from pathlib import Path
from src.utils.write_file_into_server import save_plot_into_server, download_all_files_rep_hugging_face
from sklearn.metrics import mean_squared_error, r2_score
from aiofiles import open as aio_open
from tqdm import tqdm
from copy import deepcopy
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging
from threading import Lock
log = setup_logging()
env = Env()


@dataclass
class ClassicInference:
    entry: EntryClassicInference

    def __post_init__(self):

        self.dictidx = self.entry.DictIdx
        self.dictmerge = self.entry.DictMerge
        self.dictseasonal = self.entry.DictSeasonal
        self.future_or_estimate = self.entry.FutureOrEstimate
        self.plots = self.entry.Plots
        self.save_plots = self.entry.SavePlots
        self.save_path_plots = self.entry.SavePathPlots
        self.path_to_weights = self.entry.SavePathWeights

        if self.path_to_weights is None:
            self.path_to_weights = Path(os.path.join(path_to_project(), env.__getattr__("WEIGHTS_CLASSIC_PATH")))
        else:
            self.path_to_weights = Path(os.path.join(self.path_to_weights))
        create_directories_if_not_exist([self.path_to_weights])

        self.results = {}
        self.dictmodels = {}

        self.minmax_resid = MinMaxScaler()
        self.minmax_trend = MinMaxScaler()
        self.minmax_season = MinMaxScaler()
        self.minmax_sellprice = MinMaxScaler()
        self.minmax_series = MinMaxScaler()

        if self.future_or_estimate == 'estimate':
            self.without_test = False
        elif self.future_or_estimate == 'future':
            self.without_test = True

        if self.save_plots:
            if self.save_path_plots is None:
                self.save_path_plots = os.path.join(path_to_project(), env.__getattr__("PLOTS_PATH"))
            else:
                self.save_path_plots = os.path.join(self.save_path_plots)
            create_directories_if_not_exist([self.save_path_plots])

        self.lock = Lock()

    async def download_weights(self):
        if len(os.listdir(self.path_to_weights)) == 0:
            await download_all_files_rep_hugging_face(
                model_name="GrafTrahula/STORE_CLASSIC",
                save_dir=self.path_to_weights,
                token=None
            )

    async def inference(self):
        await self.download_weights()
        self.load_models()
        with tqdm(total=len(self.dictmerge.items()), unit="ItemID") as pbar:
            for index, (item_id, params) in enumerate(self.dictmerge.items()):
                self.results[f"{item_id}"] = deepcopy(self.dictseasonal)
                series = params['cnt']
                date_id = params['date_id']
                sell_price = params['sell_price']
                event_name = params['event_name']
                event_type = params['event_type']
                cashback = params['cashback']
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
                series.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
                self.evaluate(series, exogenous, item_id)
                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_description(f"Processing {item_id}")

        await self.visualise()

    def load_models(self) -> dict[str, list[tuple[object, dict]]]:
        """
        Загрузить модели и их JSON-файлы из указанной директории.
    
        :param path_to_weights: Директория, где хранятся .zip и .json файлы моделей.
        :return: Словарь {период прогноза: [(модель, JSON-данные), ...]}
        """
        models_dict = {"week": [], "month": [], "quater": []}

        # Проходим по всем файлам в директории
        for filename in os.listdir(self.path_to_weights):
            # Получаем полный путь к файлу
            full_path = os.path.join(self.path_to_weights, filename)

            # Извлекаем имя модели и период из имени файла
            if filename.endswith(".zip"):

                filename = filename.split(".zip")[0]
                # Разделяем строку по символу "_"
                parts = filename.split("_")
                # Извлекаем компоненты
                model_name = parts[0]
                item_id = "_".join(parts[1:-2])
                key = parts[-2]
                prefix = f"{item_id}_{key}_{model_name}"

                # Путь к JSON-файлу
                json_filename = f"{filename}.json"
                json_path = os.path.join(self.path_to_weights, json_filename)

                # Загружаем модель
                model = ClassicModel.from_pretrained_(
                    model_name=f"{model_name}",
                    dir_path=self.path_to_weights,
                    prefix=f"{item_id}_{key}_{model_name}"
                )

                # # Загружаем JSON-данные
                # json_data = await self.load_json_file(json_path)

                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as json_file:
                        json_data = json.loads(json_file.read())
                else:
                    return {}

                # Добавляем модель и JSON в соответствующий период
                if key in models_dict:
                    models_dict[key].append({f"{item_id}": (model, json_data)})

        self.dictmodels = models_dict

    # @staticmethod
    # async def load_json_file(json_path):
    #     """
    #     Асинхронно загружает данные из JSON-файла.
    #
    #     :param json_path: Путь к JSON-файлу.
    #     :return: Данные из JSON-файла в виде Python-объекта.
    #     """
    #     if os.path.exists(json_path):
    #         async with aio_open(json_path, "r", encoding="utf-8") as json_file:
    #             content = await json_file.read()  # Асинхронно читаем содержимое
    #             return json.loads(content)       # Загружаем JSON из строки
    #     else:
    #         return {}  # Возвращаем пустой объект, если файл не существует

    async def visualise(self):
        for item_id, periods in self.results.items():
            ncols = 3  # Количество столбцов
            nrows = -(-len(periods) // ncols)  # Округляем вверх количество строк
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
            axes = axes.flatten()  # Упрощаем перебор осей

            # Получаем данные из словаря
            actual = self.dictmerge[item_id]['cnt']
            sell_price = self.dictmerge[f'{item_id}']['sell_price']
            date_id = self.dictmerge[f'{item_id}']['date_id']
            actual.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            sell_price.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]

            for idx, (period, result) in enumerate(periods.items()):
                ax = axes[idx]

                # Восстановленный предсказанный ряд
                pred = result['pred']
                # pred['resid'] = self.minmax_resid.inverse_transform(pred['resid'].values.reshape(-1, 1)).flatten()
                # pred['trend'] = self.minmax_trend.inverse_transform(pred['trend'].values.reshape(-1, 1)).flatten()
                # pred['season'] = self.minmax_season.inverse_transform(pred['season'].values.reshape(-1, 1)).flatten()
                pred_indexes = pred.index
                # pred = self.minmax_series.inverse_transform(pred.values.reshape(-1, 1)).flatten()
                pred = pd.Series(pred, name='pred', index=pred_indexes).clip(lower=0)

                # Суммируем компоненты для получения восстановленного ряда
                # pred = pd.Series(pred[['resid', 'trend', 'season']].sum(axis=1).clip(lower=0), name='series', index=pred.index)

                if self.future_or_estimate == 'estimate':
                    # Оценка
                    rmse = result['rmse']

                    # Устанавливаем границы интервала
                    start = max(pred.index[0] - pd.Timedelta(days=2 * self.dictseasonal[period]), actual.index[0])
                    end = min(pred.index[-1] + pd.Timedelta(days=self.dictseasonal[period]), actual.index[-1])

                    # Обрезаем данные для отображения
                    actual_trimmed = actual.loc[start:end]
                    pred_trimmed = pred.loc[start:end]

                    # Построение графиков
                    ax.plot(actual_trimmed.index, actual_trimmed.values, label='Actual', color='blue', alpha=0.5)
                    ax.plot(pred_trimmed.index, pred_trimmed.values, label=f'Estimate Prediction ({period})',
                            color='red', linewidth=2)

                elif self.future_or_estimate == 'future':

                    # Обрезаем данные actual до двух последних периодов предсказания
                    period_length = len(pred)  # Длина одного периода предсказания
                    actual_trimmed = actual.iloc[-2 * period_length:]  # Берем последние 2 периода

                    # Построение графиков
                    ax.plot(actual_trimmed.index, actual_trimmed.values, label='Actual (last 2 periods)', color='blue',
                            alpha=0.5)
                    ax.plot(pred.index, pred.values, label=f'Future Prediction ({period})', color='green', linewidth=2)

                # Оформление
                ax.set_title(f"{item_id} - {period}")
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True)

                # Поворот подписей на оси X
                ax.tick_params(axis='x', rotation=45)

                if not self.without_test:
                    # Добавляем текст с оценкой score
                    ax.text(0.95, 0.95, f'RMSE: {rmse:.3f}', transform=ax.transAxes,
                            fontsize=12, color='black', ha='right', va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

            # Убираем лишние оси, если их больше, чем графиков
            for extra_ax in axes[len(periods):]:
                extra_ax.axis('off')

            # Настройка и отображение всех графиков
            plt.tight_layout()
            if self.plots:
                plt.show()
            if self.save_plots:
                path = os.path.join(self.save_path_plots, f"classic_inference_{item_id}.png")
                await save_plot_into_server(fig, path)
            plt.close(fig)

    def evaluate(self, series, exogenous, item_id):

        for index, ((period, val), (_, item_list)) in enumerate(
                zip(self.dictseasonal.items(), self.dictmodels.items())):
            item_json = [s for s in item_list if str(item_id) in s.keys()][0]
            model, json = item_json[f'{item_id}']
            splitter = SingleWindowSplitter(fh=[i for i in range(val)],
                                            window_length=len(series) - val)
            rmse_values = []
            r2_values = []

            indices = list(splitter.split(series))
            train_indices = indices[0][0]
            test_indices = indices[0][1]
            train = series.iloc[train_indices]
            test = series.iloc[test_indices]

            if self.without_test:
                train = series
                test = val

            rmse, r2, pred, model = self.calc_feature(train, test, exogenous, model, period, item_id)

            self.results[f"{item_id}"][f'{period}'] = {
                "rmse": rmse,
                "r2": r2,
                "pred": pred,
                "model": model
            }

    def calc_feature(self, train, test, exogenous, model, period, item_id):

        if not self.without_test:

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
            exogenous['sell_price'] = self.minmax_sellprice.fit_transform(
                exogenous['sell_price'].values.reshape(-1, 1)
            ).flatten()

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

            train.loc[train == 0.01] = 0.000001
            test.loc[test == 0.01] = 0.000001
            pred = model.fit_pred_async(train, test, exogenous, self.lock)
            rmse = np.sqrt(mean_squared_error(test, pred))
            r2 = r2_score(test, pred)
            return rmse, r2, pred, model
        else:
            # series = train

            # resid, trend, season = dec_series(series, 7, 'additive') # additive
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
            #         exogenous['sell_price'].values.reshape(-1, 1)
            #     ).flatten()
            # train = pd.concat([resid, trend, season], axis=1)

            # series = pd.Series(
            #     self.minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(),
            #     name='series', index=series.index
            # )

            # train = series

            train.loc[train == 0.01] = 0.000001
            pred = model.fit_pred_async(train, test, exogenous, self.lock, 'future')
            return None, None, pred, model
