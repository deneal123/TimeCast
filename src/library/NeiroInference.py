import os
import re
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from functools import partial
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sktime.split import SingleWindowSplitter
from copy import deepcopy
from src.utils.write_file_into_server import save_plot_into_server, download_all_files_rep_hugging_face
from src.library.NeiroDataset import get_datasets, collate_fn
from src.library.utils import calculate_metrics_auto, convert_timeseries_to_dataframe
from src.utils.create_dir import create_directories_if_not_exist
from src.library.pydantic_models import EntryNeiroInference
from iTransformer import iTransformer, iTransformerFFT
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging
from fastapi import HTTPException, status

log = setup_logging()
env = Env()


@dataclass
class NeiroInference:
    entry: EntryNeiroInference

    def __post_init__(self):

        self.dictidx = self.entry.DictIdx
        self.dictmerge = self.entry.DictMerge
        self.dictseasonal = self.entry.DictSeasonal
        self.dictmodels = self.entry.DictModels
        self.future_or_estimate = self.entry.FutureOrEstimate
        self.seq_len = self.entry.SeqLen
        self.path_to_weights = self.entry.PathWeights
        self.plots = self.entry.Plots
        self.save_plots = self.entry.SavePlots
        self.save_path_plots = self.entry.SavePathPlots
        self.use_device = self.entry.UseDevice
        self.num_workers = self.entry.NumWorkers
        self.pin_memory = self.entry.PinMemory

        if self.path_to_weights is None:
            self.path_to_weights = Path(os.path.join(path_to_project(), env.__getattr__("WEIGHTS_NEIRO_PATH")))
        else:
            self.path_to_weights = Path(os.path.join(self.path_to_weights))
        create_directories_if_not_exist([self.path_to_weights])

        self.results = {}

        self.minmax_resid = MinMaxScaler()
        self.minmax_trend = MinMaxScaler()
        self.minmax_season = MinMaxScaler()
        self.minmax_sellprice = MinMaxScaler()
        self.minmax_series = MinMaxScaler()

        self.checkpoint = None
        self.test_dataset = None
        self.test_loader = None
        self.batch_size = 1

        self.models = {}

        self.dictloadmodels = {}

        if self.future_or_estimate == 'estimate':
            self.without_test = False
        elif self.future_or_estimate == 'future':
            self.without_test = True

        # Перемещение модели на GPU, если CUDA доступен
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")

        if self.save_plots:
            if self.save_path_plots is None:
                self.save_path_plots = os.path.join(path_to_project(), env.__getattr__("PLOTS_PATH"))
            else:
                self.save_path_plots = os.path.join(self.save_path_plots)
            create_directories_if_not_exist([self.save_path_plots])

    async def download_weights(self):
        if len(os.listdir(self.path_to_weights)) == 0:
            await download_all_files_rep_hugging_face(
                model_name="GrafTrahula/STORE_NEIRO",
                save_dir=self.path_to_weights,
                token=None
            )

    async def inference(self):
        log.info("Checking weights")
        await self.download_weights()
        log.info("Initialize models")
        self.load_models()
        with tqdm(total=len(self.dictmerge.items()), unit="ItemID") as pbar:
            for index, (item_id, params) in enumerate(self.dictmerge.items()):
                self.results[f"{item_id}"] = deepcopy(self.dictseasonal)
                self.evaluate(item_id)
                # Обновляем прогресс-бар
                pbar.update(1)
                log.info(f"Processing {item_id}")
                pbar.set_description(f"Processing {item_id}")
        log.info(f"Visialising")
        await self.visualise()

    def get_models(self, period: int):

        for model_name, model_params in self.dictmodels.items():
            if model_name == "IFFT":
                self.models[model_name] = iTransformerFFT(
                    num_variates=model_params["num_variates"],
                    lookback_len=self.seq_len,
                    num_tokens_per_variate=model_params["num_tokens_per_variate"],
                    dim=model_params["dim"],
                    depth=model_params["depth"],
                    heads=model_params["heads"],
                    dim_head=model_params["dim_head"],
                    pred_length=(period,),
                    use_reversible_instance_norm=model_params["use_reversible_instance_norm"]
                ).to(self.device)
            elif model_name == "IF":
                self.models[model_name] = iTransformer(
                    num_variates=model_params["num_variates"],
                    lookback_len=self.seq_len,
                    num_tokens_per_variate=model_params["num_tokens_per_variate"],
                    dim=model_params["dim"],
                    depth=model_params["depth"],
                    heads=model_params["heads"],
                    dim_head=model_params["dim_head"],
                    pred_length=(period,),
                    use_reversible_instance_norm=model_params["use_reversible_instance_norm"]
                ).to(self.device)

    # Функция для загрузки данных
    def get_loaders(self, item_id: str, period: int):

        self.test_dataset = get_datasets(
            dictidx=self.dictidx,
            dictmerge=self.dictmerge,
            item_id=item_id,
            period=period,
            seq_len=self.seq_len,
            future_or_estimate_or_train=self.future_or_estimate
        )

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn,
                                                         minmax_resid=self.minmax_resid,
                                                         minmax_trend=self.minmax_trend,
                                                         minmax_season=self.minmax_season,
                                                         minmax_sellprice=self.minmax_sellprice,
                                                         minmax_series=self.minmax_series,
                                                         without_test=self.without_test),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=False)

    def load_models(self) -> dict[str, list[tuple[object, dict]]]:
        models_dict = {"week": [], "month": [], "quater": []}

        # Проходим по всем файлам в директории
        for filename in os.listdir(self.path_to_weights):
            # Получаем полный путь к файлу
            full_path = os.path.join(self.path_to_weights, filename)

            # Извлекаем имя модели и период из имени файла
            if filename.endswith(".pt"):

                filename = filename.split(".pt")[0]
                # Разделяем строку по символу "_"
                parts = filename.split("_")
                # Извлекаем компоненты
                model_name = parts[0]
                item_id = "_".join(parts[1:-2])
                key = parts[-2]
                prefix = f"{item_id}_{key}_{model_name}"

                self.get_models(self.dictseasonal[f'{key}'])

                # Загружаем модель
                path = os.path.join(self.path_to_weights, f"{model_name}_{prefix}.pt")
                if os.path.isfile(path):
                    self.checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                    try:
                        self.models[f'{model_name}'].load_state_dict(self.checkpoint['model_state_dict'])
                    except Exception as ex:
                        log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)

                # Добавляем модель
                if key in models_dict:
                    models_dict[key].append({f"{item_id}": (self.models[f'{model_name}'], model_name)})

        self.dictloadmodels = models_dict

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
                pred = result['pred'][0]

                if len(pred) > 4:
                    pred = pred.resid + pred.trend + pred.season
                else:
                    pred = pred.series

                # Заменяем все значения ниже 0 на 0
                pred = pred.clip(lower=0.0)

                if self.future_or_estimate == 'estimate':
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

                if self.future_or_estimate == 'estimate':
                    # Добавляем текст с оценкой score
                    ax.text(0.95, 0.95, f'RMSE: {rmse:.2f}', transform=ax.transAxes,
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
                path = os.path.join(self.save_path_plots, f"neiro_inference_{item_id}.png")
                await save_plot_into_server(fig, path)
            plt.close(fig)

    def evaluate(self, item_id: str):

        for index, ((period, val), (_, item_list)) in enumerate(
                zip(self.dictseasonal.items(), self.dictloadmodels.items())):
            self.get_loaders(item_id, val)
            item_json = [s for s in item_list if str(item_id) in s.keys()][0]
            model, name_model = item_json[f'{item_id}']

            preds = []
            model.eval()
            all_y_true = []
            all_y_pred = []

            # Проходим по набору данных
            with torch.no_grad():
                for jndex, batch in enumerate(self.test_loader):

                    if len(batch['train']) > 4:
                        proccess = True
                    else:
                        proccess = False

                    # Распаковка тренировочных данных    
                    timestamp_test = batch['train']["timestamp"]
                    date_id_test = batch['train']["date_id"].to(self.device)
                    series_test = batch['train']["series"].to(self.device)
                    if proccess:
                        resid_test = batch['train']["resid"].to(self.device)
                        trend_test = batch['train']["trend"].to(self.device)
                        season_test = batch['train']["season"].to(self.device)
                        resid_test = resid_test.unsqueeze(-1)
                        trend_test = trend_test.unsqueeze(-1)
                        season_test = season_test.unsqueeze(-1)
                    exogenous_test = batch['train']["exogenous"].to(self.device)
                    # Добавляем третью ось (формат [batch, seq_length, 1])
                    date_id_test = date_id_test.unsqueeze(-1)
                    series_test = series_test.unsqueeze(-1)
                    # Объединяем вдоль третьей оси
                    if proccess:
                        timeseries_test = torch.cat((resid_test,  # date_id_test,
                                                     trend_test,
                                                     season_test,
                                                     exogenous_test), dim=-1)  # Формат [batch, seq_length, 8]
                    else:
                        timeseries_test = torch.cat((series_test,  # date_id_test,
                                                     exogenous_test), dim=-1)  # Формат [batch, seq_length, 6]

                    # Распаковка валидационных данных
                    if self.future_or_estimate == 'estimate':
                        timestamp_valid = batch['test']["timestamp"]
                        date_id_valid = batch['test']["date_id"].to(self.device)
                        series_valid = batch['test']["series"].to(self.device)
                        if proccess:
                            resid_valid = batch['test']["resid"].to(self.device)
                            trend_valid = batch['test']["trend"].to(self.device)
                            season_valid = batch['test']["season"].to(self.device)
                            resid_valid = resid_valid.unsqueeze(-1)
                            trend_valid = trend_valid.unsqueeze(-1)
                            season_valid = season_valid.unsqueeze(-1)
                        exogenous_valid = batch['test']["exogenous"].to(self.device)
                        # Добавляем третью ось (формат [batch, seq_length, 1])
                        date_id_valid = date_id_valid.unsqueeze(-1)
                        series_valid = series_valid.unsqueeze(-1)
                        # Объединяем вдоль третьей оси
                        if proccess:
                            timeseries_valid = torch.cat((resid_valid,  # date_id_valid,
                                                          trend_valid,
                                                          season_valid,
                                                          exogenous_valid), dim=-1)  # Формат [batch, seq_length, 8]
                        else:
                            timeseries_valid = torch.cat((series_valid,  # date_id_valid
                                                          exogenous_valid), dim=-1)  # Формат [batch, seq_length, 6]

                    # Тестируем модель
                    logits = torch.nan_to_num(model(timeseries_test)[val], nan=0.0)

                    # Определяем timestamp_valid для future
                    if self.future_or_estimate == 'future':
                        # Генерация будущих временных меток
                        last_timestamp = timestamp_test[0][-1]
                        timestamp_valid = pd.date_range(start=last_timestamp,
                                                        periods=val,
                                                        freq='D')
                        timestamp_valid = [timestamp_valid]

                    preds.append(convert_timeseries_to_dataframe(1, logits, timestamp_valid,
                                                                 self.minmax_resid,
                                                                 self.minmax_trend,
                                                                 self.minmax_season,
                                                                 self.minmax_series))

                    if self.future_or_estimate == 'estimate':
                        # Собираем предсказания и истинные значения
                        all_y_true.append(timeseries_valid.cpu().detach().numpy())
                        all_y_pred.append(logits.cpu().detach().numpy())

            if self.future_or_estimate == 'estimate':
                # После всех батчей вычисляем метрики
                all_y_true = np.concatenate(all_y_true, axis=0)
                all_y_pred = np.concatenate(all_y_pred, axis=0)
                num_components = all_y_true.shape[-1]
                if num_components == 7:
                    all_y_true = all_y_true[..., :3]
                    all_y_pred = all_y_pred[..., :3]
                elif num_components == 5:
                    all_y_true = all_y_true[..., :1]
                    all_y_pred = all_y_pred[..., :1]

                mae, rmse, r2 = calculate_metrics_auto(all_y_true, all_y_pred)

                self.results[f"{item_id}"][f'{period}'] = {
                    "rmse": rmse,
                    "pred": preds[0],
                    "model": name_model
                }
            else:
                self.results[f"{item_id}"][f'{period}'] = {
                    "pred": preds[0],
                    "model": name_model
                }
