import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from timecast.data.neiro import get_datasets, collate_fn
from timecast.models.loss import CustomLoss
from timecast._internal.utils import calculate_metrics_auto
from timecast._internal.features import num_variates
from timecast._internal.neiro_channels import is_process_batch, assemble_input
from timecast._internal.dirs import create_directories_if_not_exist
from timecast.schemas import EntryNeiroGraduate
from timecast._internal.utils import save_model
from iTransformer import iTransformer, iTransformerFFT
from timecast.config import get_paths
from timecast._internal.logging import setup_logging

log = setup_logging()


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
        self.decompose_period = self.entry.DecomposePeriod
        self.decompose_model = self.entry.DecomposeModel

        if self.path_to_weights is None:
            self.path_to_weights = Path(get_paths().weights_neiro_dir)
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

    def graduate(self):
        for index, (item_id, _) in enumerate(self.dictmerge.items()):
            log.info(f"Processing: {item_id}")
            for period, value in self.dictseasonal.items():
                log.info(f"Period: {period}")
                # Получаем генераторы обучения, валидации и теста
                self.get_loaders(item_id, value)
                # Загружаем модели
                self.get_models(value)
                # Определяем оптимизатор, функцию потерь и планировщик
                self.get_opt_crit_sh()
                # Загружаем чекпоинт
                self.load_checkpoint(item_id, value)
                # Выводим информацию
                print(self.__str__())
                # Обучаем
                log.info("Training")
                self.train_models(value)
                # Тестируем
                log.info("Testing")
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
                                                          minmax_series=self.minmax_series,
                                                          decompose_period=self.decompose_period,
                                                          decompose_model=self.decompose_model),
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
                                                         minmax_series=self.minmax_series,
                                                         decompose_period=self.decompose_period,
                                                         decompose_model=self.decompose_model),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory,
                                      drop_last=True)

    def get_models(self, period: int):
        # Число каналов модели вычисляется из числа фич (см. neiro_channels.num_variates).
        # num_variates из запроса игнорируется.
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
                self.optimizers[model_name], mode='min', patience=2
            )
            for model_name in self.models.keys()
        }

    def load_checkpoint(self, item_id: str, period: int):
        for name_model, model in self.models.items():
            path = os.path.join(self.path_to_weights, f"{name_model}_{item_id}_{period}_{name_model}.pt")
            try:
                if os.path.isfile(path):
                    self.checkpoint = torch.load(path, map_location=self.device, weights_only=True)
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

                        proccess = is_process_batch(batch)

                        # Каналы: decomposition (3) + фичи при proccess, иначе ряд + фичи.
                        timeseries_train = assemble_input(batch['train'], proccess, self.device)
                        timeseries_valid = assemble_input(batch['test'], proccess, self.device)

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
                # Целевые компоненты: декомпозиция (3) при proccess, иначе сам ряд (1).
                # Не зависит от числа фич (раньше было зашито num_components==7/5).
                _k = 3 if proccess else 1
                all_y_true = all_y_true[..., :_k]
                all_y_pred = all_y_pred[..., :_k]

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

                        proccess = is_process_batch(batch)

                        timeseries_test = assemble_input(batch['train'], proccess, self.device)
                        timeseries_valid = assemble_input(batch['test'], proccess, self.device)

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
            # Целевые компоненты: декомпозиция (3) при proccess, иначе сам ряд (1).
            _k = 3 if proccess else 1
            all_y_true = all_y_true[..., :_k]
            all_y_pred = all_y_pred[..., :_k]

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
