from abc import ABC, abstractmethod
import pandas as pd
import os
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import ARIMA
import matplotlib.pyplot as plt
import json
import numpy as np
from copy import deepcopy
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging

log = setup_logging()
env = Env()


class ClassicModel(ABC):
    name_model: str = None
    _registered_models: dict = {}

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, exogenous):
        self.train = train
        self.test = test
        self.exogenous = exogenous
        self.model = None
        self.dictseasonal = {
            "week": 7,
            "month": 30,
            "quater": 90
        }
        self.train_index = self.train.index
        self.train.index.freq = "D"
        self.test_index = self.test.index
        self.test.index.freq = "D"

    @abstractmethod
    def fit(self, **kwargs) -> None:
        pass

    def fit_pred(self, train, test, exogenous, future_or_estimate='estimate'):
        if isinstance(test, pd.DataFrame) or isinstance(test, pd.Series):
            period = len(test)
        elif isinstance(test, int):
            period = test
            exogenous_columns = exogenous.columns
            last_timestamp = exogenous.index[-2]
            future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=period, freq='D')
            future_exogenous = pd.DataFrame(0, index=future_timestamps, columns=exogenous_columns)
        self.model.fit(y=train)  # X=exogenous.loc[train.index].fillna(0))
        if future_or_estimate == 'estimate':
            pred = self.model.predict(fh=np.arange(0, period))
            # X=exogenous.loc[test.index].fillna(0))
        elif future_or_estimate == "future":
            pred = self.model.predict(fh=np.arange(0, period))
            # X=future_exogenous.fillna(0))
        return pred

    def param(self):
        return self.model.get_params(deep=True)

    def pred(self) -> pd.Series:
        pred = self.model.predict(fh=np.arange(0, len(self.test)))
        # X=self.exogenous.loc[self.test.index].fillna(0))
        return pred

    def save(self, dir_path: str = "./weights", prefix: str = None, results=None) -> None:
        if not os.path.exists(dir_path):
            create_directories_if_not_exist([dir_path])
        if prefix is not None:
            file_path = os.path.join(dir_path, f"{self.name_model}_{prefix}")
        else:
            file_path = os.path.join(dir_path, f"{self.name_model}")
        self.model.save(file_path, serialization_format='pickle')
        if results is not None:
            if prefix is not None:
                json_path = os.path.join(dir_path, f"{self.name_model}_{prefix}.json")
            else:
                json_path = os.path.join(dir_path, f"{self.name_model}.json")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, ensure_ascii=False, indent=4)
        log.info(f"Save model to {file_path}")

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls._registered_models[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_model_classes(cls):
        return {
            "AUTOARIMA": _AUTOARIMA_,
            "AUTOREG": _AUTOREG_,
            "AUTOETS": _AUTOETS_,
            "PROPHET": _PROPHET_,
            "TBATS": _TBATS_,
        }

    @classmethod
    def create_model(cls, model_name: str, train: pd.DataFrame, test: pd.DataFrame, exogenous: pd.DataFrame):
        # Получаем модельный словарь через get_model_classes()
        model_classes = cls.get_model_classes()
        if model_name not in model_classes:
            raise ValueError(f"Unknown model name: {model_name}")
        # Получаем нужный класс модели
        model_class = model_classes[model_name]
        return model_class(train, test, exogenous)

    @classmethod
    def from_pretrained(cls, dir_path: str = "./weights", prefix: str = None):
        if cls is ClassicModel:
            raise TypeError("Cannot instantiate abstract class ClassicModel directly.")
        if prefix is None:
            raise Exception("Set ItemID")
        if prefix is not None:
            file_path = os.path.join(dir_path, f"{cls.name_model}_{prefix}")
        else:
            file_path = os.path.join(dir_path, f"{cls.name_model}")
        instance = cls.__new__(cls)  # Создаем экземпляр без вызова __init__
        instance.model = instance.model.load_from_path(f"{file_path}.zip")
        return instance

    @classmethod
    def from_pretrained_(cls, model_name: str, dir_path: str, prefix: str):
        # Получаем модельный словарь через get_model_classes()
        model_classes = cls.get_model_classes()
        if model_name not in model_classes:
            raise ValueError(f"Unknown model name: {model_name}")
        model_class = model_classes[model_name]
        return model_class.from_pretrained(dir_path=dir_path, prefix=prefix)


@ClassicModel.register_model("AUTOARIMA")
class _AUTOARIMA_(ClassicModel):
    name_model = "AUTOARIMA"
    # model = AutoARIMA()
    model = ARIMA()

    def __init__(self, train: pd.Series, test: pd.Series, exogenous) -> None:
        super().__init__(train, test, exogenous)

    def fit(self, p: int, q: int, d: int, D: int, P: int, Q: int, m: str) -> None:
        # self.model = AutoARIMA(start_p=2, start_q=2, start_P=0, start_Q=0, d=None, D=None,
        #                        max_p=p, max_q=q, max_P=P, max_Q=Q, max_d=d, max_D=D,
        #                        sp=self.dictseasonal[f'{m}'],
        #                        seasonal=True,
        #                        stationary=False,
        #                        time_varying_regression=False,
        #                        mle_regression=False,
        #                        trace=False,
        #                        error_action='ignore',  
        #                        suppress_warnings=True,
        #                        n_jobs=8,
        #                        stepwise=True)
        self.model = ARIMA(order=(p, 0, q))
        self.model.fit(y=self.train)  # X=self.exogenous.loc[self.train.index].fillna(0))


@ClassicModel.register_model("AUTOREG")
class _AUTOREG_(ClassicModel):
    name_model = "AUTOREG"
    model = AutoREG()

    def __init__(self, train: pd.Series, test: pd.Series, exogenous) -> None:
        super().__init__(train, test, exogenous)

    def fit(self, lags: int, m: str) -> None:
        self.model = AutoREG(lags=lags, seasonal=True, trend='t')  # period=self.dictseasonal[f'{m}'])
        self.model.fit(y=self.train)  # X=self.exogenous.loc[self.train.index].fillna(0))


@ClassicModel.register_model("AUTOETS")
class _AUTOETS_(ClassicModel):
    name_model = "AUTOETS"
    model = AutoETS()

    def __init__(self, train: pd.Series, test: pd.Series, exogenous) -> None:
        super().__init__(train, test, exogenous)

    def fit(self, m: str) -> None:
        self.model = AutoETS(auto=True, freq='D', sp=self.dictseasonal[f'{m}'])
        self.model.fit(y=self.train)  # X=self.exogenous.loc[self.train.index].fillna(0))


@ClassicModel.register_model("PROPHET")
class _PROPHET_(ClassicModel):
    name_model = "PROPHET"
    model = Prophet()

    def __init__(self, train: pd.Series, test: pd.Series, exogenous) -> None:
        super().__init__(train, test, exogenous)

    def fit(self, n_changepoints: int) -> None:
        self.model = Prophet(freq="D", n_changepoints=n_changepoints)
        self.model.fit(y=self.train)  # X=self.exogenous.loc[self.train.index].fillna(0))


@ClassicModel.register_model("TBATS")
class _TBATS_(ClassicModel):
    name_model = "TBATS"
    model = TBATS()

    def __init__(self, train: pd.Series, test: pd.Series, exogenous) -> None:
        super().__init__(train, test, exogenous)

    def fit(self, none) -> None:
        self.model = TBATS(use_box_cox=False,
                           use_trend=True,
                           use_damped_trend=True,
                           use_arma_errors=False)
        self.model.fit(y=self.train)  # X=self.exogenous.loc[self.train.index].fillna(0))
