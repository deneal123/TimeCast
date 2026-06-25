import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from iTransformer import iTransformer, iTransformerFFT
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from timecast._internal.dirs import create_directories_if_not_exist
from timecast._internal.features import num_variates
from timecast._internal.io import download_all_files_rep_hugging_face, save_plot_into_server
from timecast._internal.logging import setup_logging
from timecast._internal.neiro_channels import assemble_input, is_process_batch
from timecast._internal.utils import calculate_metrics_auto, convert_timeseries_to_dataframe
from timecast.config import get_paths
from timecast.data.neiro import collate_fn, get_datasets
from timecast.schemas import EntryNeiroInference

log = setup_logging()


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
        self.decompose_period = self.entry.DecomposePeriod
        self.decompose_model = self.entry.DecomposeModel

        if self.path_to_weights is None:
            self.path_to_weights = Path(get_paths().weights_neiro_dir)
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
                self.save_path_plots = get_paths().plots_dir
            else:
                self.save_path_plots = os.path.join(self.save_path_plots)
            create_directories_if_not_exist([self.save_path_plots])

    def download_weights(self):
        if len(os.listdir(self.path_to_weights)) == 0:
            download_all_files_rep_hugging_face(
                model_name="GrafTrahula/STORE_NEIRO",
                save_dir=self.path_to_weights,
                token=None
            )

    def inference(self):
        log.info("Checking weights")
        self.download_weights()
        log.info("Initialize models")
        self.load_models()
        with tqdm(total=len(self.dictmerge.items()), unit="ItemID") as pbar:
            for _index, (item_id, _params) in enumerate(self.dictmerge.items()):
                self.results[f"{item_id}"] = deepcopy(self.dictseasonal)
                self.evaluate(item_id)
                # Обновляем прогресс-бар
                pbar.update(1)
                log.info(f"Processing {item_id}")
                pbar.set_description(f"Processing {item_id}")
        log.info("Visialising")
        self.visualise()

    def get_models(self, period: int):
        # Архитектура должна совпадать с обученной (см. neiro_channels.num_variates).
        nvar = num_variates(self.dictidx)

        for model_name, model_params in self.dictmodels.items():
            if model_name == "IFFT":
                self.models[model_name] = iTransformerFFT(
                    num_variates=nvar,
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
                    num_variates=nvar,
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
                                                         without_test=self.without_test,
                                                         decompose_period=self.decompose_period,
                                                         decompose_model=self.decompose_model),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=False)

    def load_models(self) -> dict[str, list[tuple[object, dict]]]:
        # Периоды из dictseasonal (произвольные ключи), а не жёстко week/month/quater.
        models_dict = {period: [] for period in self.dictseasonal}

        # Проходим по всем файлам в директории
        for filename in os.listdir(self.path_to_weights):
            # Получаем полный путь к файлу
            os.path.join(self.path_to_weights, filename)

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

    def visualise(self):
        for item_id, periods in self.results.items():
            ncols = 3  # Количество столбцов
            nrows = -(-len(periods) // ncols)  # Округляем вверх количество строк
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
            axes = axes.flatten()  # Упрощаем перебор осей

            # Фактический ряд (унифицировано: generic target / retail cnt). sell_price был мёртв.
            generic = "feature_cols" in self.dictidx
            actual = self.dictmerge[item_id]['target' if generic else 'cnt']
            date_id = self.dictmerge[item_id]['date_id']
            if generic:
                actual.index = list(self.dictmerge[item_id]['date'])
            else:
                actual.index = [self.dictidx['idx2date'][idx - 1] for idx in date_id]

            for idx, (period, result) in enumerate(periods.items()):
                ax = axes[idx]

                # pred уже pd.Series (reconstructed в evaluate).
                pred = result['pred']

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
                save_plot_into_server(fig, path)
            plt.close(fig)

    def evaluate(self, item_id: str):

        for _index, ((period, val), (_, item_list)) in enumerate(
                zip(self.dictseasonal.items(), self.dictloadmodels.items(), strict=False)):
            self.get_loaders(item_id, val)
            item_json = [s for s in item_list if str(item_id) in s][0]
            model, name_model = item_json[f'{item_id}']

            preds = []
            model.eval()
            all_y_true = []
            all_y_pred = []

            # Проходим по набору данных
            with torch.no_grad():
                for _jndex, batch in enumerate(self.test_loader):

                    proccess = is_process_batch(batch)

                    timestamp_test = batch['train']["timestamp"]
                    timeseries_test = assemble_input(batch['train'], proccess, self.device)

                    if self.future_or_estimate == 'estimate':
                        timestamp_valid = batch['test']["timestamp"]
                        timeseries_valid = assemble_input(batch['test'], proccess, self.device)

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

            # Восстанавливаем прогноз как 1D Series (resid+trend+season или series).
            pred_df = preds[0][0]
            if 'resid' in pred_df.columns:
                pred_series = (pred_df['resid'] + pred_df['trend'] + pred_df['season']).clip(lower=0.0)
            else:
                pred_series = pred_df['series'].clip(lower=0.0)

            # actual: последние val точек оригинального ряда (только для estimate).
            actual_test = None
            if self.future_or_estimate == 'estimate':
                generic = "feature_cols" in self.dictidx
                actual_full = self.dictmerge[item_id]['target' if generic else 'cnt'].copy()
                if generic:
                    actual_full.index = list(self.dictmerge[item_id]['date'])
                else:
                    _date_id = self.dictmerge[item_id]['date_id']
                    actual_full.index = [self.dictidx['idx2date'][i - 1] for i in _date_id]
                actual_test = actual_full.iloc[-val:]

            if self.future_or_estimate == 'estimate':
                # После всех батчей вычисляем метрики
                all_y_true = np.concatenate(all_y_true, axis=0)
                all_y_pred = np.concatenate(all_y_pred, axis=0)
                # Целевые компоненты: декомпозиция (3) при proccess, иначе сам ряд (1).
                _k = 3 if proccess else 1
                all_y_true = all_y_true[..., :_k]
                all_y_pred = all_y_pred[..., :_k]

                mae, rmse, r2 = calculate_metrics_auto(all_y_true, all_y_pred)

                self.results[f"{item_id}"][f'{period}'] = {
                    "rmse": rmse,
                    "pred": pred_series,
                    "actual": actual_test,
                    "model": name_model
                }
            else:
                self.results[f"{item_id}"][f'{period}'] = {
                    "pred": pred_series,
                    "actual": None,
                    "model": name_model
                }
