import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from functools import partial
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from src.library.NeiroDataset import get_datasets, collate_fn
from src.library.CustomLoss import CustomLoss
from src.library.utils import calculate_metrics_auto, convert_timeseries_to_dataframe
from src.utils.create_dir import create_directories_if_not_exist
from src.library.pydantic_models import EntryNeiroGraduate
from src.library.utils import save_model
from iTransformer import iTransformer, iTransformerFFT
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging

log, log_stream_handler = setup_logging()
env = Env()


@dataclass
class NeiroGraduate:
    entry: EntryNeiroGraduate

    def __post_init__(self):
        self.date = datetime.now()

        self.dictidx = self.entry.DictIdx
        self.dictmerge = self.entry.DictMerge
        self.dictseasonal = self.entry.DictSeasonal
        self.dictmodels = self.entry.DictModels
        self.seq_len = self.entry.SeqLen
        self.test_size = float(self.entry.TestSize)
        self.step_length = self.entry.StepLen
        self.path_to_weights = self.entry.PathWeights
        self.use_device = self.entry.UseDevice
        self.start_learning_rate = float(self.entry.StartLerningRate)
        self.batch_size = self.entry.BatchSize
        self.num_workers = self.entry.NumWorkers
        self.pin_memory = self.entry.PinMemory
        self.num_epochs = self.entry.NumEpochs
        self.name_optimizer = self.entry.NameOptimizer
        self.seed = self.entry.Seed

        if self.path_to_weights is None:
            self.path_to_weights = Path(os.path.join(path_to_project(), env.__getattr__("WEIGHTS_NEIRO_PATH")))
        else:
            self.path_to_weights = Path(os.path.join(self.path_to_weights))
        create_directories_if_not_exist([self.path_to_weights])

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.scheduler = {}
        self.optimizer = {}
        self.models = {}
        self.checkpoint = None

        self.minmax_resid = MinMaxScaler()
        self.minmax_trend = MinMaxScaler()
        self.minmax_season = MinMaxScaler()
        self.minmax_sellprice = MinMaxScaler()
        self.minmax_series = MinMaxScaler()

        # Перемещение модели на GPU, если CUDA доступен
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")

        if self.device == "cpu":
            self.pin_memory = False

    async def graduate(self):
        for index, (item_id, _) in enumerate(self.dictmerge.items()):
            log.info(f"Обучаем модели для товара: {item_id}")
            for period, value in self.dictseasonal.items():
                log.info(f"Период: {period}")
                # Получаем генераторы обучения, валидации и теста
                self.get_loaders(item_id, value)
                # Загружаем модели
                self.get_models(value)
                # Определяем оптимизатор, функцию потерь и планировщик
                self.get_opt_crit_sh()
                # Загружаем чекпоинт
                await self.load_checkpoint(item_id, value)
                # Выводим информацию
                print(self.__str__())
                # Обучаем
                self.train_models(value)
                # Тестируем
                self.evaluate_models(item_id, value)

    def __str__(self):
        # log.info(f"Определенное устройство: {self.use_device}")
        # log.info(f"Количество эпох обучения {self.num_epochs}")
        # log.info(f"Размер пакета: {self.batch_size}")
        # log.info(f"Выбранный оптимизатор: {self.name_optimizer}")
        return """"""

    # Функция для загрузки данных
    def get_loaders(self, item_id: str, period: int):

        self.train_dataset, self.test_dataset = get_datasets(
            dictidx=self.dictidx,
            dictmerge=self.dictmerge,
            item_id=item_id,
            test_size=self.test_size,
            period=period,
            seq_len=self.seq_len,
            step_length=self.step_length
        )

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=partial(collate_fn,
                                                          minmax_resid=self.minmax_resid,
                                                          minmax_trend=self.minmax_trend,
                                                          minmax_season=self.minmax_season,
                                                          minmax_sellprice=self.minmax_sellprice,
                                                          minmax_series=self.minmax_series),
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory,
                                       drop_last=True)

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn,
                                                         minmax_resid=self.minmax_resid,
                                                         minmax_trend=self.minmax_trend,
                                                         minmax_season=self.minmax_season,
                                                         minmax_sellprice=self.minmax_sellprice,
                                                         minmax_series=self.minmax_series),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=True)

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

    def get_opt_crit_sh(self):
        # Определение функции потерь
        self.criterion = CustomLoss(beta=1.0, delta=1.0)
        # Оптимизаторы для каждой модели
        self.optimizers = {
            model_name: optim.__dict__[self.name_optimizer](
                model.parameters(), lr=self.start_learning_rate
            )
            for model_name, model in self.models.items()
        }

        # Планировщики
        self.schedulers = {
            model_name: ReduceLROnPlateau(
                self.optimizers[model_name], mode='min', patience=2, verbose=True
            )
            for model_name in self.models.keys()
        }

    async def load_checkpoint(self, item_id: str, period: int):
        for name_model, model in self.models.items():
            path = os.path.join(self.path_to_weights, f"{name_model}_{item_id}_{period}_{name_model}.pt")
            try:
                if os.path.isfile(path):
                    self.checkpoint = await torch.load(path, map_location=self.device, weights_only=True)
                    try:
                        model.load_state_dict(self.checkpoint['model_state_dict'])
                        self.optimizers[f'{name_model}'].load_state_dict(self.checkpoint['optimizer_state_dict'])
                        log.info("Веса успешно загружены")
                    except Exception as ex:
                        log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
                else:
                    log.info("Не найден файл с моделью")
            except Exception as ex:
                log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)

    # Функция для обучения модели с валидацией
    def train_models(self, period: int):

        for name_model, model in self.models.items():

            # Переводим модель в режим тренировки
            model.train()

            # Итерируемся по эпохам
            for epoch in range(self.num_epochs):
                train_loss = 0.0
                all_y_true = []
                all_y_pred = []

                # Проходим по набору данных
                with tqdm(total=len(self.train_loader)) as pbar_train:
                    for index, batch in enumerate(self.train_loader):

                        if len(batch['train']) > 4:
                            proccess = True
                        else:
                            proccess = False

                        # Распаковка тренировочных данных    
                        timestamp_train = batch['train']["timestamp"]
                        date_id_train = batch['train']["date_id"].to(self.device)
                        series_train = batch['train']["series"].to(self.device)
                        if proccess:
                            resid_train = batch['train']["resid"].to(self.device)
                            trend_train = batch['train']["trend"].to(self.device)
                            season_train = batch['train']["season"].to(self.device)
                            resid_train = resid_train.unsqueeze(-1)
                            trend_train = trend_train.unsqueeze(-1)
                            season_train = season_train.unsqueeze(-1)
                        exogenous_train = batch['train']["exogenous"].to(self.device)
                        # Добавляем третью ось (формат [batch, seq_length, 1])
                        date_id_train = date_id_train.unsqueeze(-1)
                        series_train = series_train.unsqueeze(-1)
                        # Объединяем вдоль третьей оси
                        if proccess:
                            timeseries_train = torch.cat((resid_train,  # date_id_train,
                                                          trend_train,
                                                          season_train,
                                                          exogenous_train), dim=-1)  # Формат [batch, seq_length, 8]
                        else:
                            timeseries_train = torch.cat((series_train,  # date_id_train,
                                                          exogenous_train), dim=-1)  # Формат [batch, seq_length, 6]

                        # Распаковка валидационных данных
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
                            timeseries_valid = torch.cat((series_valid,  # date_id_valid,
                                                          exogenous_valid), dim=-1)  # Формат [batch, seq_length, 6]

                        # Обучаем модель
                        logits = torch.nan_to_num(model(timeseries_train)[period], nan=0.0)

                        if proccess:
                            loss = self.criterion(logits[:, :, :3], timeseries_valid[:, :, :3])
                        else:
                            loss = self.criterion(logits[:, :, :1], timeseries_valid[:, :, :1])
                        self.optimizers[f'{name_model}'].zero_grad()

                        # Собираем предсказания и истинные значения
                        all_y_true.append(timeseries_valid.cpu().detach().numpy())
                        all_y_pred.append(logits.cpu().detach().numpy())

                        train_loss += loss.item() * self.batch_size
                        loss.backward()
                        self.optimizers[f'{name_model}'].step()

                        # Обновляем бар
                        pbar_train.set_description(f"(Train) / {name_model}")
                        pbar_train.unit = " sample"
                        pbar_train.set_postfix(epoch=(epoch + 1), loss=train_loss / ((index + 1) * self.batch_size))
                        pbar_train.update(1)

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
                log.info(
                    f"Epoch {epoch + 1} - Model {name_model} - Loss: {train_loss / len(self.train_loader.dataset)} - MAE: {mae:.2f} - RMSE: {rmse:.2f} - R²: {r2:.2f}")

        log.info("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_models(self, item_id: str, period: int):
        best_name = ""
        best_model = None
        best_metrics = {"MAE": float('inf'), "RMSE": float('inf'), "R2": float('-inf')}

        for name_model, model in self.models.items():

            # Переводим модель в режим инференса
            model.eval()
            all_y_true = []
            all_y_pred = []
            valid_loss = 0.0

            # Проходим по набору данных
            with torch.no_grad():
                with tqdm(total=len(self.test_loader)) as pbar_test:
                    for index, batch in enumerate(self.test_loader):

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
                        logits = torch.nan_to_num(model(timeseries_test)[period], nan=0.0)

                        if proccess:
                            loss = self.criterion(logits[:, :, :3], timeseries_valid[:, :, :3])
                        else:
                            loss = self.criterion(logits[:, :, :1], timeseries_valid[:, :, :1])
                        valid_loss += loss.item() * self.batch_size

                        # Собираем предсказания и истинные значения
                        all_y_true.append(timeseries_valid.cpu().detach().numpy())
                        all_y_pred.append(logits.cpu().detach().numpy())

                        # Обновляем бар
                        pbar_test.set_description(f"(Test) / {name_model}")
                        pbar_test.unit = " sample"
                        pbar_test.set_postfix(loss=valid_loss / ((index + 1) * self.batch_size))
                        pbar_test.update(1)

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

            # Сохраняем модель, если она показала лучшие результаты
            if mae < best_metrics["RMSE"]:
                best_metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
                key = next((k for k, v in self.dictseasonal.items() if v == period), None)
                best_name = f"{name_model}_{item_id}_{key}_{name_model}"
                best_model = model.state_dict()
                best_optimizer = self.optimizers[f'{name_model}'].state_dict()

        # Сохраняем лучшую модель
        if best_model:
            save_model(self.path_to_weights, best_name, best_model, best_optimizer, self.num_epochs)
            log.info(
                f"Лучшая модель: {best_name} - MAE: {best_metrics['MAE']:.2f} - RMSE: {best_metrics['RMSE']:.2f} - R²: {best_metrics['R2']:.2f}")
        log.info("Тестирование завершено!")
