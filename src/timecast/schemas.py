from functools import wraps

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from timecast._internal.logging import setup_logging
from timecast.exceptions import ValidationFailedError

log = setup_logging()



def validate_with_pydantic(model_cls):
    """
    Декоратор для валидации данных с использованием Pydantic-модели.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Проверяем данные в аргументах функции
            try:
                data = kwargs.get("entry", args[0] if args else None)
                if not data:
                    raise ValidationFailedError("No data provided for validation.")
                # Валидация данных
                if isinstance(data, BaseModel):
                    data = data.model_dump(by_alias=True)
                validated_data = model_cls(**data)
                # Передаем валидированные данные дальше
                kwargs["entry"] = validated_data
                return func(*args, **kwargs)
            except ValidationError as ve:
                log.exception("Validation failed", exc_info=ve)
                raise ValidationFailedError("Invalid data for Pydantic model.") from ve

        return wrapper

    return decorator


def auto_generate_docstring(cls: type[BaseModel]) -> type[BaseModel]:
    """
    Декоратор для автоматического добавления docstring в классы Pydantic.
    """

    def generate_docstring(model: type[BaseModel]) -> str:
        """
        Генерация строки документации из описания полей модели Pydantic.
        """
        docstring = []
        for field_name, field_info in model.model_fields.items():
            field_details = f"Field '{field_name}':\n"
            if field_info.description:  # Получение описания
                field_details += f"  Description: {field_info.description}\n"
            if field_info.examples:  # Получение примеров
                field_details += f"  Examples: {field_info.examples}\n"
            docstring.append(field_details)
        return "\n".join(docstring)

    # Добавляем описание к существующему docstring
    cls.__doc__ = (cls.__doc__ or "") + "\n\n" + generate_docstring(cls)
    return cls


@auto_generate_docstring
class EntryClassicDataset(BaseModel):
    """
    Класс для валидации входных данных ClassicDataset
    """
    StoreID: StrictStr = Field(...,
                               alias="store_id",
                               examples=["STORE_1"],
                               description="Строковое название магазина в наборе данных")
    ShopSales: StrictStr = Field(...,
                                 alias="shop_sales",
                                 examples=["./data/shop_sales.csv"],
                                 description="Путь к набору данных shop_sales")
    ShopSalesDates: StrictStr = Field(...,
                                      alias="shop_sales_dates",
                                      examples=["./data/shop_sales_dates.csv"],
                                      description="Путь к набору данных shop_sales_dates")
    ShopSalesPrices: StrictStr = Field(...,
                                       alias="shop_sales_prices",
                                       examples=["./data/shop_sales_prices.csv"],
                                       description="Путь к набору данных shop_sales_prices")
    Plots: StrictBool | None = Field(False,
                                        alias="plots",
                                        examples=[True],
                                        description="Строить графики временных рядов? Да/Нет")
    SavePlots: StrictBool | None = Field(True,
                                            alias="save_plots",
                                            examples=[True],
                                            description="Сохранять графики временных рядов? Да/Нет")
    SavePathPlots: StrictStr | None = Field(None,
                                               alias="save_path_plots",
                                               examples=["./plots"],
                                               description="Если сохраняем графики, то куда? Дефолтный путь ./plots")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        shop_sales = values.get("shop_sales", None)
        shop_sales_dates = values.get("shop_sales_dates", None)
        shop_sales_prices = values.get("shop_sales_prices", None)

        if exclude_fields and "shop_sales" in exclude_fields and exclude_fields["shop_sales"]:
            values["shop_sales"] = "ok"

        if exclude_fields and "shop_sales_dates" in exclude_fields and exclude_fields["shop_sales_dates"]:
            values["shop_sales_dates"] = "ok"

        if exclude_fields and "shop_sales_prices" in exclude_fields and exclude_fields["shop_sales_prices"]:
            values["shop_sales_prices"] = "ok"

        if shop_sales is None and not exclude_fields:
            raise ValidationFailedError("Field 'shop_sales' must be defined when 'exclude_fields'"
                                       " is not provided or shop_sales is not excluded.")

        if shop_sales_dates is None and not exclude_fields:
            raise ValidationFailedError("Field 'shop_sales_dates' must be defined when 'exclude_fields'"
                                       " is not provided or shop_sales_dates is not excluded.")

        if shop_sales_prices is None and not exclude_fields:
            raise ValidationFailedError("Field 'shop_sales_prices' must be defined when 'exclude_fields'"
                                       " is not provided or shop_sales_prices is not excluded.")

        return values


@auto_generate_docstring
class EntryTimeSeriesDataset(BaseModel):
    """
    Валидация обобщённого временного ряда: один tidy-CSV (long).
    Колонки: time (дата), target (значение), опц. series_id, произвольные feature_*.
    Не привязан к домену (магазины/товары — частный случай).
    """
    Source: StrictStr = Field(...,
                              alias="source",
                              examples=["./data/series.csv"],
                              description="Путь к tidy-CSV (long): колонки time, target, [series_id], feature_*")
    TimeCol: StrictStr | None = Field("time",
                                         alias="time_col",
                                         examples=["time"],
                                         description="Имя колонки времени (дата/таймстамп)")
    TargetCol: StrictStr | None = Field("target",
                                           alias="target_col",
                                           examples=["target"],
                                           description="Имя колонки целевого значения")
    IdCol: StrictStr | None = Field(None,
                                       alias="series_id_col",
                                       examples=["series_id"],
                                       description="Имя колонки идентификатора ряда (None → один ряд)")
    FeatureCols: list[StrictStr] | None = Field(None,
                                                   alias="feature_cols",
                                                   examples=[["price", "promo"]],
                                                   description="Колонки-фичи (None → все, кроме time/target/series_id)")
    Plots: StrictBool | None = Field(False,
                                        alias="plots",
                                        examples=[True],
                                        description="Строить графики? Да/Нет")
    SavePlots: StrictBool | None = Field(True,
                                            alias="save_plots",
                                            examples=[True],
                                            description="Сохранять графики? Да/Нет")
    SavePathPlots: StrictStr | None = Field(None,
                                               alias="save_path_plots",
                                               examples=["./plots"],
                                               description="Куда сохранять графики (дефолт ./plots)")


@auto_generate_docstring
class EntryClassicProcess(BaseModel):
    """
    Класс для валидации входных данных ClassicProcess
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DictMerge: dict[str, pd.DataFrame] | StrictStr = Field(...,
                                                                 alias="dictmerge",
                                                                 examples=["Dict[str, pd.DataFrame]"],
                                                                 description="Словарь полученный методом ClassicDataset.dictmerge")
    DictDecompose: dict[str, StrictInt] = Field(...,
                                                alias="dictdecompose",
                                                examples=[{
                                                    "week": 7,
                                                    "month": 30,
                                                    "quarter": 90
                                                }],
                                                description="Словарь, по какому периоду будет осуществляться декомпозиция")
    RemoveBound: dict[str, StrictInt] | None = Field(...,
                                                        alias="remove_bound",
                                                        examples=[{
                                                            "lower_bound_factor": 5,
                                                            "upper_bound_factor": 5
                                                        }],
                                                        description="Границы выравнивания выбросов по медиане Q1 -+ bound_factor * IQR")
    Plots: StrictBool | None = Field(False,
                                        alias="plots",
                                        examples=[True],
                                        description="Строить графики предобработанных временных рядов? Да/Нет")
    SavePlots: StrictBool | None = Field(True,
                                            alias="save_plots",
                                            examples=[True],
                                            description="Сохранять графики временных рядов? Да/Нет")
    SavePathPlots: StrictStr | None = Field(None,
                                               alias="save_path_plots",
                                               examples=["./plots"],
                                               description="Если сохраняем графики, то куда? Дефолтный путь ./plots")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        dict_merge = values.get("dictmerge", None)

        if exclude_fields and "dictmerge" in exclude_fields and exclude_fields["dictmerge"]:
            values["dictmerge"] = "ok"

        # Если exclude_fields не передано или оно не включает dictmerge, то проверяем, что DictMerge передано
        if dict_merge is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictmerge' must be defined when 'exclude_fields'"
                                       " is not provided or dictmerge is not excluded.")

        return values


@auto_generate_docstring
class EntryClassicGraduate(BaseModel):
    """
    Класс для валидации входных данных ClassicGraduate
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DictIdx: dict | StrictStr = Field(...,
                                            alias="dictidx",
                                            examples=["Dict"],
                                            description="Словарь полученный методом ClassicProcess.dictidx")
    DictMerge: dict[str, pd.DataFrame] | StrictStr = Field(...,
                                                                 alias="dictmerge",
                                                                 examples=["Dict[str, pd.DataFrame]"],
                                                                 description="Словарь полученный методом ClassicDataset.dictmerge")
    DictSeasonal: dict[str, StrictInt] = Field(...,
                                               alias="dictseasonal",
                                               examples=[{
                                                   "week": 7,
                                                   "month": 30,
                                                   "quater": 90
                                               }],
                                               description="Словарь диапазонов предсказаний, на какую дистанцию предсказывать?")
    ModelsParams: dict[str, tuple] = Field(...,
                                           alias="models_params",
                                           examples=[{
                                               "AUTOARIMA": (3, 3, 0, 0, 1, 1, 'week'),
                                               "AUTOREG": (7, 'week'),
                                               "AUTOETS": ('week',),
                                               "PROPHET": (25,),
                                               "TBATS": (None,)
                                           }],
                                           description="Словарь с моделями и параметрами, из которых будет выбираться лучшая модель")
    SavePathWeights: StrictStr | None = Field(None,
                                                 alias="save_path_weights",
                                                 examples=["./weights_classic"],
                                                 description="Если сохраняем веса, то куда? Дефолтный путь ./weights_classic")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        dict_idx = values.get("dictidx", None)
        dict_merge = values.get("dictmerge", None)

        # Если в exclude_fields указано, что поле dictmerge должно быть исключено, удаляем его
        if exclude_fields and "dictidx" in exclude_fields and exclude_fields["dictidx"]:
            values["dictidx"] = "ok"

        if exclude_fields and "dictmerge" in exclude_fields and exclude_fields["dictmerge"]:
            values["dictmerge"] = "ok"

        # Если exclude_fields не передано или оно не включает dictidx, то проверяем, что DictIdx передано
        if dict_idx is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictidx' must be defined when 'exclude_fields'"
                                       " is not provided or dictidx is not excluded.")

        # Если exclude_fields не передано или оно не включает dictmerge, то проверяем, что DictMerge передано
        if dict_merge is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictmerge' must be defined when 'exclude_fields'"
                                       " is not provided or dictmerge is not excluded.")

        return values


@auto_generate_docstring
class EntryClassicInference(BaseModel):
    """
    Класс для валидации входных данных ClassicInference
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DictIdx: dict | StrictStr = Field(...,
                                            alias="dictidx",
                                            examples=["Dict"],
                                            description="Словарь полученный методом ClassicProcess.dictidx")
    DictMerge: dict[str, pd.DataFrame] | StrictStr = Field(...,
                                                                 alias="dictmerge",
                                                                 examples=["Dict[str, pd.DataFrame]"],
                                                                 description="Словарь полученный методом ClassicDataset.dictmerge")
    DictSeasonal: dict[str, StrictInt] = Field(...,
                                               alias="dictseasonal",
                                               examples=[{
                                                   "week": 7,
                                                   "month": 30,
                                                   "quater": 90
                                               }],
                                               description="Словарь диапазонов предсказаний, на какую дистанцию предсказывать?")
    FutureOrEstimate: StrictStr = Field('estimate',
                                        alias="future_or_estimate",
                                        examples=['estimate'],
                                        description=("Режим инференса, когда estimate оценивает последний кусок данных,"
                                                     "когда future делает предсказание в будущее"))
    SavePathWeights: StrictStr | None = Field(None,
                                                 alias="save_path_weights",
                                                 examples=["./weights_classic"],
                                                 description="Если сохраняем веса, то куда? Дефолтный путь ./weights_classic")
    Plots: StrictBool | None = Field(False,
                                        alias="plots",
                                        examples=[True],
                                        description="Строить графики предобработанных временных рядов? Да/Нет")
    SavePlots: StrictBool | None = Field(True,
                                            alias="save_plots",
                                            examples=[True],
                                            description="Сохранять графики временных рядов? Да/Нет")
    SavePathPlots: StrictStr | None = Field(None,
                                               alias="save_path_plots",
                                               examples=["./plots"],
                                               description="Если сохраняем графики, то куда? Дефолтный путь ./plots")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        dict_idx = values.get("dictidx", None)
        dict_merge = values.get("dictmerge", None)

        # Если в exclude_fields указано, что поле dictmerge должно быть исключено, удаляем его
        if exclude_fields and "dictidx" in exclude_fields and exclude_fields["dictidx"]:
            values["dictidx"] = "ok"

        if exclude_fields and "dictmerge" in exclude_fields and exclude_fields["dictmerge"]:
            values["dictmerge"] = "ok"

        # Если exclude_fields не передано или оно не включает dictidx, то проверяем, что DictIdx передано
        if dict_idx is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictidx' must be defined when 'exclude_fields'"
                                       " is not provided or dictidx is not excluded.")

        # Если exclude_fields не передано или оно не включает dictmerge, то проверяем, что DictMerge передано
        if dict_merge is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictmerge' must be defined when 'exclude_fields'"
                                       " is not provided or dictmerge is not excluded.")

        return values


@auto_generate_docstring
class EntryNeiroInference(BaseModel):
    """
    Класс для валидации входных данных NeiroInference
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DictIdx: dict | StrictStr = Field(...,
                                            alias="dictidx",
                                            examples=["Dict"],
                                            description="Словарь полученный методом ClassicProcess.dictidx")
    DictMerge: dict[str, pd.DataFrame] | StrictStr = Field(...,
                                                                 alias="dictmerge",
                                                                 examples=["Dict[str, pd.DataFrame]"],
                                                                 description="Словарь полученный методом ClassicDataset.dictmerge")
    DictSeasonal: dict[str, StrictInt] = Field(...,
                                               alias="dictseasonal",
                                               examples=[{
                                                   "week": 7,
                                                   "month": 30,
                                                   "quater": 90
                                               }],
                                               description="Словарь диапазонов предсказаний, на какую дистанцию предсказывать?")
    DictModels: dict[str, dict] = Field(...,
                                        alias="dictmodels",
                                        examples=[{
                                            "IF": {
                                                "num_variates": 7,
                                                "num_tokens_per_variate": 1,
                                                "dim": 256,
                                                "depth": 6,
                                                "heads": 8,
                                                "dim_head": 64,
                                                "use_reversible_instance_norm": True
                                            },
                                            "IFFT": {
                                                "num_variates": 7,
                                                "num_tokens_per_variate": 1,
                                                "dim": 256,
                                                "depth": 6,
                                                "heads": 8,
                                                "dim_head": 64,
                                                "use_reversible_instance_norm": True
                                            }
                                        }],
                                        description="Словарь с параметрами моделей")
    FutureOrEstimate: StrictStr = Field('estimate',
                                        alias="future_or_estimate",
                                        examples=['estimate'],
                                        description=("Режим инференса, когда estimate оценивает последний кусок данных,"
                                                     "когда future делает предсказание в будущее"))
    SeqLen: StrictInt | None = Field(365,
                                        alias="seq_len",
                                        examples=[365],
                                        description="Длина последовательности (lookback)")
    PathWeights: StrictStr | None = Field(None,
                                             alias="path_to_weights",
                                             examples=["./weights_neiro"],
                                             description="Если сохраняем веса, то куда? Дефолтный путь ./weights_neiro")
    Plots: StrictBool | None = Field(False,
                                        alias="plots",
                                        examples=[True],
                                        description="Строить графики временных рядов? Да/Нет")
    SavePlots: StrictBool | None = Field(True,
                                            alias="save_plots",
                                            examples=[True],
                                            description="Сохранять графики временных рядов? Да/Нет")
    SavePathPlots: StrictStr | None = Field(None,
                                               alias="save_path_plots",
                                               examples=["./plots"],
                                               description="Если сохраняем график, то куда? Дефолтный путь ./plots")
    UseDevice: StrictStr | None = Field('cuda',
                                           alias="use_device",
                                           examples=["cuda"],
                                           description="Какое устройство использовать? cpu/cuda")
    NumWorkers: StrictInt | None = Field(0,
                                            alias="num_workers",
                                            examples=[0],
                                            description="Кол. используемых потоков при подгрузке данных DataLoader (0 это 1)")
    PinMemory: StrictBool | None = Field(False,
                                            alias="pin_memory",
                                            examples=[False],
                                            description="Если True ускоряет загрузку данных на видеокарте, для cpu всегда False")
    DecomposePeriod: StrictInt | None = Field(7,
                                                 alias="decompose_period",
                                                 examples=[7],
                                                 description="Период сезонной декомпозиции ряда (база цикла), по умолчанию 7")
    DecomposeModel: StrictStr | None = Field("multiplicative",
                                                alias="decompose_model",
                                                examples=["multiplicative", "additive"],
                                                description="Модель декомпозиции; для рядов с нулями/отрицательными — additive")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        dict_idx = values.get("dictidx", None)
        dict_merge = values.get("dictmerge", None)

        # Если в exclude_fields указано, что поле dictmerge должно быть исключено, удаляем его
        if exclude_fields and "dictidx" in exclude_fields and exclude_fields["dictidx"]:
            values["dictidx"] = "ok"

        if exclude_fields and "dictmerge" in exclude_fields and exclude_fields["dictmerge"]:
            values["dictmerge"] = "ok"

        # Если exclude_fields не передано или оно не включает dictidx, то проверяем, что DictIdx передано
        if dict_idx is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictidx' must be defined when 'exclude_fields'"
                                       " is not provided or dictidx is not excluded.")

        # Если exclude_fields не передано или оно не включает dictmerge, то проверяем, что DictMerge передано
        if dict_merge is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictmerge' must be defined when 'exclude_fields'"
                                       " is not provided or dictmerge is not excluded.")

        return values


@auto_generate_docstring
class EntryNeiroGraduate(BaseModel):
    """
    Класс для валидации входных данных NeiroGraduate
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    DictIdx: dict | StrictStr = Field(...,
                                            alias="dictidx",
                                            examples=["Dict"],
                                            description="Словарь полученный методом ClassicProcess.dictidx")
    DictMerge: dict[str, pd.DataFrame] | StrictStr = Field(...,
                                                                 alias="dictmerge",
                                                                 examples=["Dict[str, pd.DataFrame]"],
                                                                 description="Словарь полученный методом ClassicDataset.dictmerge")
    DictSeasonal: dict[str, StrictInt] = Field(...,
                                               alias="dictseasonal",
                                               examples=[{
                                                   "week": 7,
                                                   "month": 30,
                                                   "quater": 90
                                               }],
                                               description="Словарь диапазонов предсказаний, на какую дистанцию предсказывать?")
    DictModels: dict[str, dict] = Field(...,
                                        alias="dictmodels",
                                        examples=[{
                                            "IF": {
                                                "num_variates": 7,
                                                "num_tokens_per_variate": 1,
                                                "dim": 512,
                                                "depth": 6,
                                                "heads": 8,
                                                "dim_head": 64,
                                                "use_reversible_instance_norm": True
                                            },
                                            "IFFT": {
                                                "num_variates": 7,
                                                "num_tokens_per_variate": 1,
                                                "dim": 512,
                                                "depth": 6,
                                                "heads": 8,
                                                "dim_head": 64,
                                                "use_reversible_instance_norm": True
                                            }
                                        }],
                                        description="Словарь с параметрами моделей")
    SeqLen: StrictInt | None = Field(365,
                                        alias="seq_len",
                                        examples=[365],
                                        description="Длина последовательности (lookback)")
    TestSize: float | None = Field(0.3,
                                    ge=0.1, le=0.5,
                                    alias="test_size",
                                    examples=[0.3],
                                    description="Доля тестовой выборки, которую не будет видеть модель")
    StepLen: StrictInt | None = Field(1,
                                         alias="step_length",
                                         examples=[1],
                                         description="Кол. шагов, через которые будут браться срезы данных")
    PathWeights: StrictStr | None = Field(None,
                                             alias="path_to_weights",
                                             examples=["./weights_neiro"],
                                             description="Если сохраняем веса, то куда? Дефолтный путь ./weights_neiro")
    UseDevice: StrictStr | None = Field('cuda',
                                           alias="use_device",
                                           examples=["cuda"],
                                           description="Какое устройство использовать? cpu/cuda")
    StartLerningRate: float | None = Field(0.0001,
                                             ge=1e-08, le=0.01,
                                             alias="start_learning_rate",
                                             examples=[0.0001],
                                             description="Начальная величина шага градиентного спуска")
    BatchSize: StrictInt | None = Field(10,
                                           alias="batch_size",
                                           examples=[10],
                                           description="Размер пакета при обучении")
    NumWorkers: StrictInt | None = Field(0,
                                            alias="num_workers",
                                            examples=[0],
                                            description="Кол. используемых потоков при подгрузке данных DataLoader (0 это 1)")
    PinMemory: StrictBool | None = Field(False,
                                            alias="pin_memory",
                                            examples=[False],
                                            description="Если True ускоряет загрузку данных на видеокарте, для cpu всегда False")
    NumEpochs: StrictInt | None = Field(20,
                                           alias="num_epochs",
                                           examples=[20],
                                           description="Количество эпох обучения для каждой модели")
    NameOptimizer: StrictStr | None = Field("Adam",
                                               alias="name_optimizer",
                                               examples=["Adam"],
                                               description="Название оптимизатора из доступных в torch.nn.optim")
    Seed: StrictInt | None = Field(17,
                                      alias="seed",
                                      examples=[17],
                                      description="Сажает зерно")
    DecomposePeriod: StrictInt | None = Field(7,
                                                 alias="decompose_period",
                                                 examples=[7],
                                                 description="Период сезонной декомпозиции ряда (база цикла), по умолчанию 7")
    DecomposeModel: StrictStr | None = Field("multiplicative",
                                                alias="decompose_model",
                                                examples=["multiplicative", "additive"],
                                                description="Модель декомпозиции; для рядов с нулями/отрицательными — additive")

    exclude_fields: dict[str, bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def handle_excluded_fields(cls, values):
        exclude_fields = values.get("exclude_fields", {})
        dict_idx = values.get("dictidx", None)
        dict_merge = values.get("dictmerge", None)

        # Если в exclude_fields указано, что поле dictmerge должно быть исключено, удаляем его
        if exclude_fields and "dictidx" in exclude_fields and exclude_fields["dictidx"]:
            values["dictidx"] = "ok"

        if exclude_fields and "dictmerge" in exclude_fields and exclude_fields["dictmerge"]:
            values["dictmerge"] = "ok"

        # Если exclude_fields не передано или оно не включает dictidx, то проверяем, что DictIdx передано
        if dict_idx is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictidx' must be defined when 'exclude_fields'"
                                       " is not provided or dictidx is not excluded.")

        # Если exclude_fields не передано или оно не включает dictmerge, то проверяем, что DictMerge передано
        if dict_merge is None and not exclude_fields:
            raise ValidationFailedError("Field 'dictmerge' must be defined when 'exclude_fields'"
                                       " is not provided or dictmerge is not excluded.")

        return values


@auto_generate_docstring
class EntrySeasonAnalyticPipeline(BaseModel):
    """
    Валидация для данных аналитики.
    Обрабатывает только входные данные для EntryClassicDataset.
    Поля для EntryClassicProcess проверяются после выполнения `ClassicDataset`.
    """
    Dataset: EntryClassicDataset = Field(...,
                                         alias="dataset",
                                         description="Данные для валидации EntryClassicDataset")
    Process: EntryClassicProcess = Field(...,
                                           alias="proccess",
                                           description="Данные для валидации EntryClassicProcess")

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data):

        if "dataset" in data:
            dataset_data = data["dataset"]
            dataset_data["exclude_fields"] = {"shop_sales": True, "shop_sales_dates": True, "shop_sales_prices": True}

        if "proccess" in data:
            process_data = data["proccess"]
            process_data["exclude_fields"] = {"dictmerge": True}
            data["proccess"] = EntryClassicProcess(**process_data)

        super().__init__(**data)


@auto_generate_docstring
class EntryClassicGraduatePipeline(BaseModel):
    """
    Валидация для данных аналитики.
    Обрабатывает только входные данные для EntryClassicDataset.
    Поля для EntryClassicGraduate проверяются после выполнения `ClassicDataset`.
    """
    Dataset: EntryClassicDataset = Field(...,
                                         alias="dataset",
                                         description="Данные для валидации EntryClassicDataset")
    Graduate: EntryClassicGraduate = Field(...,
                                           alias="graduate",
                                           description="Данные для валидации EntryClassicGraduate")

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data):

        if "dataset" in data:
            dataset_data = data["dataset"]
            dataset_data["exclude_fields"] = {"shop_sales": True, "shop_sales_dates": True, "shop_sales_prices": True}

        if "graduate" in data:
            graduate_data = data["graduate"]
            graduate_data["exclude_fields"] = {"dictidx": True, "dictmerge": True}
            data["graduate"] = EntryClassicGraduate(**graduate_data)

        super().__init__(**data)


@auto_generate_docstring
class EntryClassicInferencePipeline(BaseModel):
    """
    Валидация для данных аналитики.
    Обрабатывает только входные данные для EntryClassicDataset.
    Поля для EntryClassicInference проверяются после выполнения `ClassicDataset`.
    """
    Dataset: EntryClassicDataset = Field(...,
                                         alias="dataset",
                                         description="Данные для валидации EntryClassicDataset")
    Inference: EntryClassicInference = Field(...,
                                             alias="inference",
                                             description="Данные для валидации EntryClassicInference")

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data):

        if "dataset" in data:
            dataset_data = data["dataset"]
            dataset_data["exclude_fields"] = {"shop_sales": True, "shop_sales_dates": True, "shop_sales_prices": True}

        if "inference" in data:
            graduate_data = data["inference"]
            graduate_data["exclude_fields"] = {"dictidx": True, "dictmerge": True}
            data["inference"] = EntryClassicInference(**graduate_data)

        super().__init__(**data)


@auto_generate_docstring
class EntryNeiroGraduatePipeline(BaseModel):
    """
    Валидация для данных аналитики.
    Обрабатывает только входные данные для EntryClassicDataset.
    Поля для EntryNeiroGraduate проверяются после выполнения `ClassicDataset`.
    """
    Dataset: EntryClassicDataset = Field(...,
                                         alias="dataset",
                                         description="Данные для валидации EntryClassicDataset")
    Graduate: EntryNeiroGraduate = Field(...,
                                         alias="graduate",
                                         description="Данные для валидации EntryNeiroGraduate")

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data):

        if "dataset" in data:
            dataset_data = data["dataset"]
            dataset_data["exclude_fields"] = {"shop_sales": True, "shop_sales_dates": True, "shop_sales_prices": True}

        if "graduate" in data:
            graduate_data = data["graduate"]
            graduate_data["exclude_fields"] = {"dictidx": True, "dictmerge": True}
            data["graduate"] = EntryNeiroGraduate(**graduate_data)

        super().__init__(**data)


@auto_generate_docstring
class EntryNeiroInferencePipeline(BaseModel):
    """
    Валидация для данных аналитики.
    Обрабатывает только входные данные для EntryClassicDataset.
    Поля для EntryNeiroInference проверяются после выполнения `ClassicDataset`.
    """
    Dataset: EntryClassicDataset = Field(...,
                                         alias="dataset",
                                         description="Данные для валидации EntryClassicDataset")
    Inference: EntryNeiroInference = Field(...,
                                           alias="inference",
                                           description="Данные для валидации EntryNeiroInference")

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **data):

        if "dataset" in data:
            dataset_data = data["dataset"]
            dataset_data["exclude_fields"] = {"shop_sales": True, "shop_sales_dates": True, "shop_sales_prices": True}

        if "inference" in data:
            inference_data = data["inference"]
            inference_data["exclude_fields"] = {"dictidx": True, "dictmerge": True}
            data["inference"] = EntryNeiroInference(**inference_data)

        super().__init__(**data)
