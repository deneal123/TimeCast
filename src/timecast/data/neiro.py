from dataclasses import dataclass
from torch.utils.data import Dataset
import torch
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sklearn.model_selection import train_test_split
import pandas as pd
from timecast._internal.utils import seed_everything, dec_series
from timecast._internal.features import feature_cols
from sktime.split import SingleWindowSplitter
from sklearn.preprocessing import MinMaxScaler


# Параметры сезонной декомпозиции ряда внутри collate_fn (база сезонного цикла).
# Дефолты соответствуют retail (недельный мультипликативный паттерн). Для рядов
# с нулями/отрицательными значениями стоит передать model='additive'.
DEFAULT_DECOMPOSE_PERIOD = 7
DEFAULT_DECOMPOSE_MODEL = 'multiplicative'


def _scale_features(exogenous):
    """MinMax-нормализация ВСЕХ колонок-фич (обобщённо вместо только sell_price)."""
    for _c in exogenous.columns:
        exogenous[_c] = MinMaxScaler().fit_transform(exogenous[_c].values.reshape(-1, 1)).flatten()
    return exogenous


def collate_fn(batch,
               minmax_resid=None,
               minmax_trend=None,
               minmax_season=None,
               minmax_sellprice=None,
               minmax_series=None,
               pdata=False,
               without_test=False,
               decompose_period=DEFAULT_DECOMPOSE_PERIOD,
               decompose_model=DEFAULT_DECOMPOSE_MODEL):
    if minmax_resid and minmax_trend and minmax_season and minmax_sellprice and minmax_series:
        process = True
    else:
        process = False

    if process:
        process_batch = {
            'train': {
                'timestamp': [],
                'date_id': [],
                'series': [],
                'resid': [],
                'trend': [],
                'season': [],
                'exogenous': [],
            },
            'test': None if without_test else {
                'timestamp': [],
                'date_id': [],
                'series': [],
                'resid': [],
                'trend': [],
                'season': [],
                'exogenous': [],
            }
        }
    else:
        process_batch = {
            'train': {
                'timestamp': [],
                'date_id': [],
                'series': [],
                'exogenous': [],
            },
            'test': None if without_test else {
                'timestamp': [],
                'date_id': [],
                'series': [],
                'exogenous': [],
            }
        }

    for sample in batch:
        # Если without_test=False, объединяем train и test для декомпозиции
        if not without_test:
            combined_series = pd.concat([sample['train']['series'], sample['test']['series']])
            combined_timestamp = pd.concat([
                pd.Series(sample['train']['timestamp'], name='timestamp'),
                pd.Series(sample['test']['timestamp'], name='timestamp')
            ])
        else:
            combined_series = sample['train']['series']
            combined_timestamp = pd.Series(sample['train']['timestamp'], name='timestamp')

        if process:
            # Выполняем декомпозицию объединенных данных (или только train при without_test=True)
            resid, trend, season = dec_series(combined_series, decompose_period, decompose_model)

            # Нормализуем на объединенных данных
            resid = pd.Series(
                minmax_resid.fit_transform(resid.values.reshape(-1, 1)).flatten(),
                name='resid', index=resid.index
            )
            trend = pd.Series(
                minmax_trend.fit_transform(trend.values.reshape(-1, 1)).flatten(),
                name='trend', index=trend.index
            )
            season = pd.Series(
                minmax_season.fit_transform(season.values.reshape(-1, 1)).flatten(),
                name='season', index=season.index
            )

        # Если without_test=False, разделяем обратно на train и test
        if not without_test:
            train_len = len(sample['train']['series'])
            resid_train, resid_test = resid.iloc[:train_len], resid.iloc[train_len:]
            trend_train, trend_test = trend.iloc[:train_len], trend.iloc[train_len:]
            season_train, season_test = season.iloc[:train_len], season.iloc[train_len:]
        else:
            resid_train, trend_train, season_train = resid, trend, season

        # Train часть
        timestamp = sample['train']['timestamp']
        date_id = sample['train']['date_id']
        series = sample['train']['series']
        exogenous = sample['train']['exogenous']

        if not pdata:  # Преобразуем в тензоры torch только если pdata=False
            date_id = torch.tensor(date_id.values, dtype=torch.float32)
            series = torch.tensor(
                minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(), dtype=torch.float32
            )
            if process:
                resid_train = torch.tensor(resid_train.values, dtype=torch.float32)
                trend_train = torch.tensor(trend_train.values, dtype=torch.float32)
                season_train = torch.tensor(season_train.values, dtype=torch.float32)
            exogenous = _scale_features(exogenous)
            exogenous = torch.tensor(exogenous.values, dtype=torch.float32)
        else:
            series = pd.Series(minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(), name='series',
                               index=series.index)
            exogenous = _scale_features(exogenous)

        process_batch['train']['timestamp'].append(timestamp)
        process_batch['train']['date_id'].append(date_id if pdata else date_id)
        process_batch['train']['series'].append(series if pdata else series)
        if process:
            process_batch['train']['resid'].append(resid_train if pdata else resid_train)
            process_batch['train']['trend'].append(trend_train if pdata else trend_train)
            process_batch['train']['season'].append(season_train if pdata else season_train)
        process_batch['train']['exogenous'].append(exogenous if pdata else exogenous)

        if not without_test:
            # Test часть
            timestamp = sample['test']['timestamp']
            date_id = sample['test']['date_id']
            series = sample['test']['series']
            exogenous = sample['test']['exogenous']

            if not pdata:  # Преобразуем в тензоры torch только если pdata=False
                date_id = torch.tensor(date_id.values, dtype=torch.float32)
                series = torch.tensor(
                    minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(), dtype=torch.float32
                )
                if process:
                    resid_test = torch.tensor(resid_test.values, dtype=torch.float32)
                    trend_test = torch.tensor(trend_test.values, dtype=torch.float32)
                    season_test = torch.tensor(season_test.values, dtype=torch.float32)
                exogenous = _scale_features(exogenous)
                exogenous = torch.tensor(exogenous.values, dtype=torch.float32)
            else:
                series = pd.Series(minmax_series.fit_transform(series.values.reshape(-1, 1)).flatten(), name='series',
                                   index=series.index)
                exogenous = _scale_features(exogenous)

            process_batch['test']['timestamp'].append(timestamp)
            process_batch['test']['date_id'].append(date_id if pdata else date_id)
            process_batch['test']['series'].append(series if pdata else series)
            if process:
                process_batch['test']['resid'].append(resid_test if pdata else resid_test)
                process_batch['test']['trend'].append(trend_test if pdata else trend_test)
                process_batch['test']['season'].append(season_test if pdata else season_test)
            process_batch['test']['exogenous'].append(exogenous if pdata else exogenous)

    if not pdata:  # Преобразуем в тензоры только если pdata=False
        process_batch['train']['date_id'] = torch.stack(process_batch['train']['date_id'], dim=0)
        process_batch['train']['series'] = torch.stack(process_batch['train']['series'], dim=0)
        if process:
            process_batch['train']['resid'] = torch.stack(process_batch['train']['resid'], dim=0)
            process_batch['train']['trend'] = torch.stack(process_batch['train']['trend'], dim=0)
            process_batch['train']['season'] = torch.stack(process_batch['train']['season'], dim=0)
        process_batch['train']['exogenous'] = torch.stack(process_batch['train']['exogenous'], dim=0)

        if not without_test:
            process_batch['test']['date_id'] = torch.stack(process_batch['test']['date_id'], dim=0)
            process_batch['test']['series'] = torch.stack(process_batch['test']['series'], dim=0)
            if process:
                process_batch['test']['resid'] = torch.stack(process_batch['test']['resid'], dim=0)
                process_batch['test']['trend'] = torch.stack(process_batch['test']['trend'], dim=0)
                process_batch['test']['season'] = torch.stack(process_batch['test']['season'], dim=0)
            process_batch['test']['exogenous'] = torch.stack(process_batch['test']['exogenous'], dim=0)

    return process_batch


def get_datasets(
        dictidx: dict,
        dictmerge: dict,
        item_id: str,
        test_size: float = 0.3,
        period: int = 7,
        seq_len: int = 365,
        step_length: int = 1,
        seed: int = 17,
        future_or_estimate_or_train: str = 'train'
):
    seed_everything(seed)

    # Фильтруем данные для конкретного item_id
    item_metadata = dictmerge[f'{item_id}'].copy()
    item_metadata.sort_values(by='date_id', inplace=True)

    if future_or_estimate_or_train == 'train':

        # Разделяем данные на train и test, сохраняя порядок
        train_metadata, test_metadata = train_test_split(
            item_metadata,
            test_size=test_size,
            random_state=seed,
            shuffle=False  # Не перемешиваем данные
        )

        # SlidingWindowSplitter для кросс-валидации на train части
        train_splits = []
        splitter = SlidingWindowSplitter(
            window_length=seq_len,  # Длина окна для обучения
            fh=list(range(0, period)),  # Периоды для предсказания
            step_length=step_length  # Шаг скользящего окна
        )

        # SlidingWindowSplitter для test части
        test_splits = []
        splitter = SlidingWindowSplitter(
            window_length=seq_len,  # Длина окна для обучения
            fh=list(range(0, period)),  # Периоды для предсказания
            step_length=step_length  # Шаг скользящего окна
        )

        for train_idx, test_idx in splitter.split(train_metadata):
            train_splits.append((train_metadata.iloc[train_idx], train_metadata.iloc[test_idx]))

        for train_idx, test_idx in splitter.split(test_metadata):
            test_splits.append((test_metadata.iloc[train_idx], test_metadata.iloc[test_idx]))

        # Создаём объекты датасетов
        train_dataset = NeiroDataset(dictidx, train_splits)
        test_dataset = NeiroDataset(dictidx, test_splits)
        return train_dataset, test_dataset

    elif future_or_estimate_or_train == 'estimate':

        # Создаем список для хранения разбиений
        test_splits = []

        # Вычисляем размер данных для корректного среза
        total_length = len(item_metadata)
        start_index = total_length - seq_len - period  # Индекс начала последнего окна

        # Проверяем, что данных достаточно для такого разбиения
        if start_index < 0:
            raise ValueError("Длина данных меньше, чем seq_len + period. Увеличьте данные или уменьшите параметры.")

        item_metadata = item_metadata[start_index:]

        # Инциализация сплиттера
        splitter = SingleWindowSplitter(fh=list(range(0, period)),
                                        window_length=len(item_metadata) - period)

        # Разбиваем данные с помощью скользящего окна
        for train_idx, test_idx in splitter.split(item_metadata):
            # Сохраняем разбиения: train и test
            test_splits.append((item_metadata.iloc[train_idx], item_metadata.iloc[test_idx]))

        # Создаем тестовый датасет
        test_dataset = NeiroDataset(dictidx, test_splits)
        return test_dataset

    elif future_or_estimate_or_train == 'future':

        # Берем только последние seq_len значения
        item_metadata = item_metadata[(len(item_metadata) - seq_len):]

        # Создаем список для хранения разбиений
        test_splits = [(item_metadata, None)]

        # Создаем тестовый датасет
        test_dataset = NeiroDataset(dictidx, test_splits)

        return test_dataset


@dataclass
class NeiroDataset(Dataset):
    dictidx: dict
    metadata: list

    def __post_init__(self):
        self.datasets = []
        # Унификация: обобщённый формат задаёт feature_cols в dictidx, иначе retail-набор.
        self.feature_cols = feature_cols(self.dictidx)

        for jndex, data in enumerate(self.metadata):
            self.part_datasets = {
                "train": None,
                "test": None
            }

            # Обрабатываем train и test с помощью общего метода
            self.part_datasets['train'] = self.process_data(data[0])
            try:
                self.part_datasets['test'] = self.process_data(data[1])
            except Exception as ex:
                pass

            self.datasets.append(self.part_datasets)

    def process_data(self, data):
        """Извлекает (date_id, series, exogenous) — унифицировано для любого ряда.

        Обобщённый формат (TimeSeriesDataset): target + feature_cols + date.
        Доменный retail-формат (NeiroDataset из ClassicDataset): cnt + фикс. фичи + date_id.
        """
        date_id = data['date_id'].copy()

        if "feature_cols" in self.dictidx:
            timestamps = list(data['date'])
            series = data['target'].copy()
            exogenous = data[self.feature_cols].copy()
        else:
            timestamps = [self.dictidx['idx2date'][idx - 1] for idx in date_id]
            series = data['cnt'].copy()
            exogenous = pd.DataFrame({col: data[col].copy() for col in self.feature_cols})

        date_id.index = timestamps
        series.index = timestamps
        exogenous.index = timestamps
        return date_id, series, exogenous

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        without_test = False
        datasets = self.datasets[idx]

        timestamp_train = datasets['train'][0].index
        timestamp_train.freq = "D"
        date_id_train = datasets['train'][0]
        series_train = datasets['train'][1]
        series_train.values[series_train.values <= 0.01] = 0
        exogenous_train = datasets['train'][2]

        try:
            timestamp_test = datasets['test'][0].index
            timestamp_test.freq = "D"
            date_id_test = datasets['test'][0]
            series_test = datasets['test'][1]
            series_test.values[series_test.values <= 0.01] = 0
            exogenous_test = datasets['test'][2]
        except Exception as ex:
            without_test = True

        if not without_test:
            return {
                'train': {
                    'timestamp': timestamp_train,
                    'date_id': pd.Series(date_id_train.values, name='date_id', index=timestamp_train),
                    'series': pd.Series(series_train.values, name='series', index=timestamp_train),
                    'exogenous': pd.DataFrame(exogenous_train.values,
                                              columns=self.feature_cols,
                                              index=timestamp_train)
                },
                'test': {
                    'timestamp': timestamp_test,
                    'date_id': pd.Series(date_id_test.values, name='date_id', index=timestamp_test),
                    'series': pd.Series(series_test.values, name='series', index=timestamp_test),
                    'exogenous': pd.DataFrame(exogenous_test.values,
                                              columns=self.feature_cols,
                                              index=timestamp_test)
                }
            }
        else:
            return {
                'train': {
                    'timestamp': timestamp_train,
                    'date_id': pd.Series(date_id_train.values, name='date_id', index=timestamp_train),
                    'series': pd.Series(series_train.values, name='series', index=timestamp_train),
                    'exogenous': pd.DataFrame(exogenous_train.values,
                                              columns=self.feature_cols,
                                              index=timestamp_train)
                },
                'test': None
            }
