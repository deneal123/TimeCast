from dataclasses import dataclass
from src.library.pydantic_models import EntryClassicDataset
import pandas as pd
import duckdb as db
import matplotlib.pyplot as plt
import os
from src.utils.write_file_into_server import save_plot_into_server
import calendar as clr
from sklearn.preprocessing import MinMaxScaler
from src.utils.create_dir import create_directories_if_not_exist
from tqdm import tqdm
from copy import deepcopy
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging
log = setup_logging()
env = Env()



@dataclass
class ClassicDataset:
    entry: EntryClassicDataset

    def __post_init__(self):

        self.store_id = self.entry.StoreID
        self.shop_sales = self.entry.ShopSales
        self.shop_sales_dates = self.entry.ShopSalesDates
        self.shop_sales_prices = self.entry.ShopSalesPrices
        self.plots = self.entry.Plots
        self.save_plots = self.entry.SavePlots
        self.save_path_plots = self.entry.SavePathPlots
        
        self.table_names = self.fetch_data([self.shop_sales, self.shop_sales_dates, self.shop_sales_prices])
        self._split_merge = {}
        self.scaler = MinMaxScaler()

        if self.save_plots:
            if self.save_path_plots is None:
                self.save_path_plots = os.path.join(path_to_project(), env.__getattr__("PLOTS_PATH"))
            else:
                self.save_path_plots = os.path.join(self.save_path_plots)
            create_directories_if_not_exist([self.save_path_plots])
        
        self.month_color = 'blue'
        self.quarter_color = 'grey'

    @property
    def dictidx(self):
        return {
            "wday2idx": self.wday2idx,
            "idx2wday": self.idx2wday,
            "month2idx": self.month2idx,
            "idx2month": self.idx2month,
            "date2idx": self.date2idx,
            "idx2date": self.idx2date,
            "event_name2idx": self.event_name2idx,
            "idx2event_type": self.idx2event_type,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item
        }

    @property
    def merge(self):
        return db.sql(f"SELECT * FROM 'merge';")

    @property
    def dictmerge(self):
        return self._split_merge
    
    async def dataset(self):
        try:
            self.merge_data()
            self.split_merge()
            await self.visualiser()
        except Exception as ex:
            log.error("", exc_info=ex)

    @staticmethod
    def fetch_data(path_list: list):
        names = []
        for path in path_list:
            path = str(path).replace('/', '\\')
            name = path.split('\\')[-1].split('.')[-2]
            query = f"CREATE TABLE IF NOT EXISTS '{name}' AS FROM read_csv('{path}');"
            db.query(query)
            names.append(name)
        return names
            
    def merge_data(self):
        query = f"""
        SELECT 
            items.item_id,
            '{self.store_id}' AS store_id,
            dates.date,
            dates.wm_yr_wk,
            dates.weekday,
            dates.wday,
            dates.month,
            dates.year,
            dates.event_name_1 AS event_name,
            dates.event_type_1 AS event_type,
            dates.CASHBACK_{self.store_id} AS cashback,
            dates.date_id,
            COALESCE(s.cnt, 0) AS cnt,
            p.sell_price
        FROM 
            (SELECT DISTINCT item_id 
             FROM 'shop_sales' 
             WHERE store_id = '{self.store_id}') AS items
        CROSS JOIN
            'shop_sales_dates' AS dates
        LEFT JOIN
            'shop_sales' AS s
            ON s.store_id = '{self.store_id}' 
            AND s.item_id = items.item_id 
            AND s.date_id = dates.date_id
        LEFT JOIN
            'shop_sales_prices' AS p
            ON p.store_id = '{self.store_id}' 
            AND p.item_id = items.item_id 
            AND p.wm_yr_wk = dates.wm_yr_wk
        ORDER BY 
            items.item_id, dates.date_id;
        """
        result = db.sql(query).to_df()
        self.dict_idx(result)
        result['event_name'] = result['event_name'].map(self.event_name2idx)
        result['event_type'] = result['event_type'].map(self.event_type2idx)
        result['item_id'] = result['item_id'].map(self.item2idx)
        result['cnt'] = result['cnt'].apply(lambda x: max(x, 0.01) if isinstance(x, (int, float)) else x)
        result.drop(columns=['store_id', 'date', 'wm_yr_wk', 'weekday'], inplace=True)
        query = f"CREATE TABLE IF NOT EXISTS 'merge' AS FROM 'result';"
        db.query(query)
        query = f"SELECT * FROM 'merge';"
        merge = db.query(query).to_df()

    def dict_idx(self, data):
        unique_weekdays = data['weekday'].unique()
        unique_months = data['month'].unique()
        unique_date = data['date'].unique()
        unique_event_name = data['event_name'].unique()
        unique_event_type = data['event_type'].unique()
        unique_item = data['item_id'].unique()
        self.wday2idx = {day: idx for idx, day in enumerate(unique_weekdays)}
        self.idx2wday = {idx: day for idx, day in enumerate(unique_weekdays)}
        self.month2idx = {idx: month for idx, month in enumerate(clr.month_name) if idx > 0}
        self.idx2month = {idx: month for idx, month in enumerate(clr.month_name) if idx > 0}
        self.date2idx = {date: idx for idx, date in enumerate(unique_date)}
        self.idx2date = {idx: date for idx, date in enumerate(unique_date)}
        self.event_name2idx = {event_name: idx for idx, event_name in enumerate(unique_event_name)}
        self.idx2event_name = {idx: event_name for idx, event_name in enumerate(unique_event_name)}
        self.event_type2idx = {event_type: idx for idx, event_type in enumerate(unique_event_type)}
        self.idx2event_type = {idx: event_type for idx, event_type in enumerate(unique_event_type)}
        self.item2idx = {item: idx for idx, item in enumerate(unique_item)}
        self.idx2item = {idx: item for idx, item in enumerate(unique_item)}

    def split_merge(self):
        # Получаем объединенную таблицу merge
        query = f"SELECT * FROM 'merge';"
        merge = db.query(query).to_df()        
        # Проходим по каждому уникальному item_id в словаре item2idx
        for item, idx in self.item2idx.items():
            # Фильтруем строки таблицы для текущего item_id
            item_table = merge[merge['item_id'] == idx]
            # Добавляем таблицу в словарь под ключом item
            self._split_merge[item] = item_table

    async def visualiser(self):
        # Создаем сетку для графиков
        fig, ax = plt.subplots(15, 1, figsize=(12, 70))
        ax = ax.flatten()  # Преобразуем для удобной индексации
    
        for i, (item, df) in enumerate(self._split_merge.items()):
    
            # Нормализация данных
            normalized_cnt = self.scaler.fit_transform(df[['cnt']])
            normalized_price = self.scaler.fit_transform(df[['sell_price']])
    
            # Построение графиков cnt и sell_price
            ax[i].plot(df['date_id'], normalized_cnt, label='cnt', color='orange')
            ax[i].plot(df['date_id'], normalized_price, label='price', color='purple')
    
            # Вертикальные линии для месяцев
            for month in df['date_id'].unique()[::30]:  # Каждую 30-ю дату (условно месяц)
                ax[i].axvline(x=month, color=self.month_color, linestyle='--',
                              alpha=0.3, label='Month' if month == df['date_id'].unique()[0] else '')
    
            # Вертикальные линии для кварталов
            for quarter in df['date_id'].unique()[::90]:  # Каждую 90-ю дату (условно квартал)
                ax[i].axvline(x=quarter, color=self.quarter_color, linestyle='--',
                              alpha=0.3, label='Quarter' if quarter == df['date_id'].unique()[0] else '')
    
            # Настройка графика
            ax[i].set_title(f'{item}', fontsize=10)  # Заголовок с названием item
            ax[i].set_xlabel('date_id')
            ax[i].set_ylabel('cnt')
            ax[i].legend(loc='upper right', fontsize=8)  # Легенда в правом верхнем углу
            ax[i].grid(True)
    
        # Удаляем лишние пустые графики
        for j in range(i + 1, len(ax)):
            fig.delaxes(ax[j])
    
        # Добавляем общий заголовок
        fig.suptitle('Time Series Plots for Items', fontsize=16)
        fig.tight_layout()
        if self.plots:
            plt.show()
        if self.save_plots:
            path = os.path.join(self.save_path_plots, "classic_dataset.png")
            await save_plot_into_server(fig, path)
        plt.close(fig)
