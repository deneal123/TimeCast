import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sktime.split import SingleWindowSplitter
from tqdm import tqdm

from timecast._internal.dirs import create_directories_if_not_exist
from timecast._internal.io import download_all_files_rep_hugging_face, save_plot_into_server
from timecast._internal.logging import setup_logging
from timecast._internal.prepare import prepare_series_exog
from timecast.config import get_paths
from timecast.models.classic import ClassicModel
from timecast.schemas import EntryClassicInference

log = setup_logging()


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
            self.path_to_weights = Path(get_paths().weights_classic_dir)
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
                self.save_path_plots = get_paths().plots_dir
            else:
                self.save_path_plots = os.path.join(self.save_path_plots)
            create_directories_if_not_exist([self.save_path_plots])

        self.lock = Lock()

    def download_weights(self):
        if len(os.listdir(self.path_to_weights)) == 0:
            download_all_files_rep_hugging_face(
                model_name="GrafTrahula/STORE_CLASSIC",
                save_dir=self.path_to_weights,
                token=None
            )

    def inference(self):
        log.info("Checking weights")
        self.download_weights()
        log.info("Initializing models")
        self.load_models()
        with tqdm(total=len(self.dictmerge.items()), unit="ItemID") as pbar:
            for _index, (item_id, params) in enumerate(self.dictmerge.items()):
                log.info(f"Process {item_id}")
                self.results[f"{item_id}"] = deepcopy(self.dictseasonal)
                series, exogenous = prepare_series_exog(params, self.dictidx)
                log.info("Evaluating")
                self.evaluate(series, exogenous, item_id)
                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_description(f"Processing {item_id}")
        log.info("Visualising")
        self.visualise()

    def load_models(self) -> dict[str, list[tuple[object, dict]]]:
        """
        Загрузить модели и их JSON-файлы из указанной директории.
    
        :param path_to_weights: Директория, где хранятся .zip и .json файлы моделей.
        :return: Словарь {период прогноза: [(модель, JSON-данные), ...]}
        """
        # Периоды берём из dictseasonal (поддержка произвольных ключей), а не жёстко.
        models_dict = {period: [] for period in self.dictseasonal}

        # Проходим по всем файлам в директории
        for filename in os.listdir(self.path_to_weights):
            # Получаем полный путь к файлу
            os.path.join(self.path_to_weights, filename)

            # Извлекаем имя модели и период из имени файла
            if filename.endswith(".zip"):

                filename = filename.split(".zip")[0]
                # Разделяем строку по символу "_"
                parts = filename.split("_")
                # Извлекаем компоненты
                model_name = parts[0]
                item_id = "_".join(parts[1:-2])
                key = parts[-2]

                # Путь к JSON-файлу
                json_filename = f"{filename}.json"
                json_path = os.path.join(self.path_to_weights, json_filename)

                # Загружаем модель
                model = ClassicModel.from_pretrained_(
                    model_name=f"{model_name}",
                    dir_path=self.path_to_weights,
                    prefix=f"{item_id}_{key}_{model_name}"
                )

                if os.path.exists(json_path):
                    with open(json_path, encoding="utf-8") as json_file:
                        json_data = json.loads(json_file.read())
                else:
                    return {}

                # Добавляем модель и JSON в соответствующий период
                if key in models_dict:
                    models_dict[key].append({f"{item_id}": (model, json_data)})

        self.dictmodels = models_dict

    def visualise(self):
        for item_id, periods in self.results.items():
            ncols = 3  # Количество столбцов
            nrows = -(-len(periods) // ncols)  # Округляем вверх количество строк
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
            axes = axes.flatten()  # Упрощаем перебор осей

            # Получаем фактический ряд: generic — 'target' + 'date', retail — 'cnt' + idx2date.
            generic = "feature_cols" in self.dictidx
            actual = self.dictmerge[item_id]['target' if generic else 'cnt']
            if generic:
                actual.index = list(self.dictmerge[item_id]['date'])
            else:
                date_id = self.dictmerge[f'{item_id}']['date_id']
                actual.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]

            for idx, (period, result) in enumerate(periods.items()):
                ax = axes[idx]

                # Восстановленный предсказанный ряд
                pred = result['pred']
                pred = pd.Series(pred, name='pred', index=pred.index).clip(lower=0)

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
                save_plot_into_server(fig, path)
            plt.close(fig)

    def evaluate(self, series, exogenous, item_id):

        for _index, ((period, val), (_, item_list)) in enumerate(
                zip(self.dictseasonal.items(), self.dictmodels.items(), strict=False)):
            item_json = [s for s in item_list if str(item_id) in s][0]
            model, json = item_json[f'{item_id}']
            splitter = SingleWindowSplitter(fh=[i for i in range(val)],
                                            window_length=len(series) - val)

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
                "actual": test if not self.without_test else None,
                "model": model
            }

    def calc_feature(self, train, test, exogenous, model, period, item_id):
        if not self.without_test:
            # Масштабируем ВСЕ фичи (а не только sell_price) — поддержка любого ряда.
            for _col in exogenous.columns:
                exogenous[_col] = self.minmax_sellprice.fit_transform(
                    exogenous[_col].values.reshape(-1, 1)
                ).flatten()

            train.loc[train == 0.01] = 0.000001
            test.loc[test == 0.01] = 0.000001
            pred = model.fit_pred_async(train, test, exogenous, self.lock)
            rmse = np.sqrt(mean_squared_error(test, pred))
            r2 = r2_score(test, pred)
            return rmse, r2, pred, model
        else:
            train.loc[train == 0.01] = 0.000001
            pred = model.fit_pred_async(train, test, exogenous, self.lock, 'future')
            return None, None, pred, model
