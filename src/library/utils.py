from sktime.performance_metrics.forecasting import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error, root_mean_squared_error
from scipy import stats, signal
from scipy.stats import shapiro
from scipy.signal import windows as wind
from scipy.stats import ttest_1samp, shapiro
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose as decompose
import statsmodels.api as sm
import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import MinMaxScaler
import random
import requests
from src.utils.custom_logging import setup_logging
log = setup_logging()


def clear_gpu_memory():
    """
    Очищает память графического процессора (GPU) в PyTorch.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Очищает кеш
        torch.cuda.ipc_collect()  # Высвобождает неиспользуемую память (полезно при многопоточности)
        print("GPU memory has been cleared.")
    else:
        print("CUDA is not available.")


def save_model(path_to_weights,
               name_model,
               model_state_dict,
               optimizer_state_dict,
               num_epochs):
    # Сохраняем модель
    path = os.path.join(path_to_weights, f"{name_model}.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict},
        path)


def convert_timeseries_to_dataframe(batch_size,
                                    timeseries,
                                    timestamp,
                                    minmax_resid: MinMaxScaler = None,
                                    minmax_trend: MinMaxScaler = None,
                                    minmax_season: MinMaxScaler = None,
                                    minmax_series: MinMaxScaler = None):
    if minmax_resid and minmax_trend and minmax_season and minmax_series is not None:
        proccess = True
    else:
        proccess = False

    dataframes = []

    for part in range(batch_size):

        if proccess:
            # Извлекаем данные для текущего сэмпла
            # date_id = timeseries[part, :, 0].cpu().detach().numpy()
            # series = timeseries[part, :, 0].cpu().detach().numpy()
            resid = timeseries[part, :, 0].cpu().detach().numpy()
            trend = timeseries[part, :, 1].cpu().detach().numpy()
            season = timeseries[part, :, 2].cpu().detach().numpy()
            sell_price = timeseries[part, :, 3].cpu().detach().numpy()
            event_name = timeseries[part, :, 4].cpu().detach().numpy()
            event_type = timeseries[part, :, 5].cpu().detach().numpy()
            cashback = timeseries[part, :, 6].cpu().detach().numpy()
        else:
            # Извлекаем данные для текущего сэмпла
            # date_id = timeseries[part, :, 0].cpu().detach().numpy()
            series = timeseries[part, :, 0].cpu().detach().numpy()
            sell_price = timeseries[part, :, 1].cpu().detach().numpy()
            event_name = timeseries[part, :, 2].cpu().detach().numpy()
            event_type = timeseries[part, :, 3].cpu().detach().numpy()
            cashback = timeseries[part, :, 4].cpu().detach().numpy()

        # Используем временные метки из `timestamp` для индекса
        timestamps = timestamp[part]

        # Формируем DataFrame для текущего сэмпла
        if proccess:
            data = {
                # 'series': series.flatten(),
                'resid': minmax_resid.inverse_transform(resid.flatten().reshape(-1, 1)).flatten(),
                'trend': minmax_trend.inverse_transform(trend.flatten().reshape(-1, 1)).flatten(),
                'season': minmax_season.inverse_transform(season.flatten().reshape(-1, 1)).flatten(),
                'sell_price': sell_price.flatten(),
                'event_name': event_name.flatten(),
                'event_type': event_type.flatten(),
                'cashback': cashback.flatten(),
            }
        else:
            data = {
                'series': minmax_series.inverse_transform(series.flatten().reshape(-1, 1)).flatten(),
                'sell_price': sell_price.flatten(),
                'event_name': event_name.flatten(),
                'event_type': event_type.flatten(),
                'cashback': cashback.flatten(),
            }

        df = pd.DataFrame(data)

        # Присваиваем временные метки как индекс
        df['timestamp'] = timestamps
        df.set_index('timestamp', inplace=True)

        # Добавляем DataFrame в список
        dataframes.append(df)

    return dataframes


def decompose_series(series: pd.Series, period: int, model: str):
    return decompose(
        series,
        model=model,
        filt=None,
        period=period,
        two_sided=True,
        extrapolate_trend=1
    )


def dec_series(series: pd.Series, period: int, model: str):
    series_indexes = series.index
    series_freq = series.index.freq
    series_not_zero = series.copy()
    series_not_zero[series_not_zero == 0] = 1e-8
    series_mult = decompose_series(series_not_zero, period, model)
    trend = series_mult.trend
    trend.index = series_indexes
    trend.index.freq = series_freq
    trend.name = 'trend'
    season = series_mult.seasonal
    season.index = series_indexes
    season.index.freq = series_freq
    season.name = 'season'
    resid = series_mult.resid
    resid.index = series_indexes
    resid.index.freq = series_freq
    resid.name = 'resid'
    return resid, trend, season


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_fit(resids):
    mean_t, mean_pval = ttest_1samp(resids, 0)
    stat_t, stat_pval = adfuller(resids)[:2]
    norm_t, norm_pval = shapiro(resids)
    autocor_pval = (acorr_ljungbox(resids, model_df=0, return_df=False)['lb_pvalue'] < 0.05).sum() == 0
    lm, lm_pval, ff, ff_pval = het_breuschpagan(resids, sm.add_constant(resids.reset_index().index))

    records = {
        "mean_resids": np.mean(resids),
        "mean_t": mean_t,
        "mean_pval": mean_pval,
        "stationary_t": stat_t,
        "stationary_pval": 1 - stat_pval,
        'autocor_pval': autocor_pval,
        "norm_t": norm_t,
        "norm_pval": norm_pval,
        "heteroscedasticity_t": ff,
        "heteroscedasticity_pval": ff_pval,
        "check_result": (mean_pval > 0.05) & (stat_pval < 0.05) & (norm_pval > 0.05) & (ff_pval > 0.05) & autocor_pval
    }
    return records


def metrics_report(y_true, y_pred):
    mape_func = MeanAbsolutePercentageError()
    mae_func = MeanAbsoluteError()

    y_true_mean = y_true.mean() * np.ones(shape=(len(y_true),))

    mapes = mape_func.evaluate_by_index(y_true, y_pred) * 100
    maes = mae_func.evaluate_by_index(y_true, y_pred)
    mapes_zero = mape_func.evaluate_by_index(y_true, y_true_mean) * 100
    maes_zero = mae_func.evaluate_by_index(y_true, y_true_mean)

    stats_list = ['mean', 'median', lambda y: y.quantile(0.75), lambda y: y.quantile(0.95), 'std']

    res = pd.DataFrame([
        maes.agg(stats_list).reset_index(drop=True),
        maes_zero.agg(stats_list).reset_index(drop=True),
        mapes.agg(stats_list).reset_index(drop=True),
        mapes_zero.agg(stats_list).reset_index(drop=True)
    ], index=['maes', 'maes_zero', 'mapes', 'mapes_zero'])

    r2 = r2_score(y_true, y_pred)
    res.loc["r2", :] = r2
    return res, r2


def calculate_metrics_auto(y_true, y_pred):
    """
    Рассчитывает метрики отдельно для каждой компоненты и возвращает их сумму.

    Параметры:
    - y_true: numpy.ndarray или torch.Tensor с формой (batch_size, seq_len, num_components).
    - y_pred: numpy.ndarray или torch.Tensor с формой (batch_size, seq_len, num_components).

    Возвращает:
    - dict с метриками (MAE, RMSE, R2) по компонентам.
    """
    # Преобразуем в numpy, если вход в виде torch.Tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Инициализируем метрики
    mae_total, rmse_total, r2_total = 0, 0, 0

    # Перебираем компоненты
    num_components = y_true.shape[-1]
    for i in range(num_components):
        y_true_component = y_true[..., i].reshape(-1)
        y_pred_component = y_pred[..., i].reshape(-1)

        # Рассчитываем метрики для компоненты
        mae = mean_absolute_error(y_true_component, y_pred_component)
        rmse = root_mean_squared_error(y_true_component, y_pred_component)
        r2 = r2_score(y_true_component, y_pred_component)

        mae_total += mae
        rmse_total += rmse
        r2_total += r2

    return mae_total, rmse_total, r2_total
