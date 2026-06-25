import asyncio
import logging
import os
import warnings

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from timecast import (
    EntryClassicGraduatePipeline,
    EntryClassicInferencePipeline,
    EntryNeiroGraduatePipeline,
    EntryNeiroInferencePipeline,
    EntrySeasonAnalyticPipeline,
    TimeCastError,
    configure_paths,
)

from src import path_to_project
from src.services.analytic_services import season_analytic_pipeline
from src.services.classic_services import classic_graduate_pipeline, classic_inference_pipeline
from src.services.file_services import get_zip_from_server, upload_csv_to_server
from src.services.neiro_services import neiro_graduate_pipeline, neiro_inference_pipeline
from src.services.storage_service import storage
from src.services.task_store import TaskStatus, task_store
from src.services.timeseries_services import (
    timeseries_graduate_pipeline,
    timeseries_inference_pipeline,
    timeseries_neiro_graduate_pipeline,
    timeseries_neiro_inference_pipeline,
)
from src.utils.custom_logging import setup_logging

warnings.simplefilter("ignore", category=FutureWarning)

# Загружаем .env один раз при старте (python-dotenv вместо кастомного env.py).
load_dotenv(os.path.join(path_to_project(), ".env"))
log = setup_logging()


def _abs_path(value):
    """Делает путь абсолютным относительно корня проекта; None — оставить дефолт библиотеки."""
    return os.path.join(path_to_project(), value) if value else None


# Подставляем в библиотеку пути из .env (как раньше — абсолютные, от корня проекта).
# Если переменная не задана, остаётся относительный дефолт из timecast/config.py.
configure_paths(
    data_dir=_abs_path(os.environ.get("DATA_PATH")),
    plots_dir=_abs_path(os.environ.get("PLOTS_PATH")),
    weights_classic_dir=_abs_path(os.environ.get("WEIGHTS_CLASSIC_PATH")),
    weights_neiro_dir=_abs_path(os.environ.get("WEIGHTS_NEIRO_PATH")),
)

app_server = FastAPI(title="TimeCast API", version="1.3.2",
                     description="This API server is intended for the TimeCast project. For rights, contact the service owner.")

app = FastAPI()

app.mount("/server", app_server)

# Список разрешённых origin берётся из переменной окружения CORS_ORIGINS
# (значения через запятую). По умолчанию — локальный фронтенд. В проде НЕ "*",
# так как allow_origins=["*"] несовместимо с allow_credentials=True.
cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_server.mount("/public", StaticFiles(directory=os.path.join(path_to_project(), "public")), name="public")


# Единый перевод доменных ошибок библиотеки в HTTP-ответы.
# Пакет timecast не зависит от FastAPI и поднимает TimeCastError-наследников;
# здесь, на краю веб-слоя, они превращаются в корректный HTTP-код.
@app_server.exception_handler(TimeCastError)
async def timecast_error_handler(request: Request, exc: TimeCastError):
    log.exception("Domain error", exc_info=exc)
    return JSONResponse(status_code=getattr(exc, "http_status", 500),
                        content={"detail": str(exc) or exc.__class__.__name__})


# Определяем теги
ServerStreamTag = OpenApiTag(name="Stream", description="Operations stream")
ServerFileTag = OpenApiTag(name="File", description="Operations file")
ServerAnalyticTag = OpenApiTag(name="Analytic", description="Operations analytic")
ServerGraduateTag = OpenApiTag(name="Graduate", description="Operations graduate")
ServerInferenceTag = OpenApiTag(name="Inference", description="Operations inference")
ServerTaskTag = OpenApiTag(name="Tasks", description="Async training queue")

# Настройка документации с тегами
app_server.openapi_tags = [
    ServerStreamTag.model_dump(),
    ServerFileTag.model_dump(),
    ServerAnalyticTag.model_dump(),
    ServerGraduateTag.model_dump(),
    ServerInferenceTag.model_dump(),
    ServerTaskTag.model_dump(),
]


# Очередь для логов
log_queue = asyncio.Queue()


class LogStreamHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        log_entry = self.format(record)
        # Добавляем лог в очередь с проверкой, есть ли активный цикл событий
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print(f"Adding log entry to queue: {log_entry}")
            # Если цикл событий уже работает, используем asyncio.create_task
            asyncio.create_task(log_queue.put(log_entry))
        else:
            # Если цикл не работает, добавляем задачу вручную через run_until_complete
            loop.run_until_complete(log_queue.put(log_entry))


log_stream_handler = LogStreamHandler()
log.addHandler(log_stream_handler)


async def log_generator():
    while True:
        log_entry = await log_queue.get()  # Ждем, пока появится новый лог
        yield f"data: {log_entry}\n\n"  # Форматируем как SSE-сообщение
        log_queue.task_done()


@app_server.get("/stream-logs", tags=["Stream"])
async def stream_logs():
    return StreamingResponse(log_generator(), media_type="text/event-stream")


# Тестирование логов (например, для проверки)
@app_server.get("/generate-log", tags=["Stream"])
async def generate_log():
    log.info("This is a new log message!")  # Генерация тестового лога
    return {"message": "Log generated"}


# Health-check для Docker/nginx (доступен как /server/health).
@app_server.get("/health", tags=["Stream"])
async def health():
    return {"status": "ok"}



@app_server.post("/upload_csv/", response_model=dict, tags=["File"])
async def upload_csv(files: list[UploadFile] = File(...)):
    """
    Route for upload csv files.

    :param files: CSV Files. [UploadFile]

    :return: response model None.
    """
    try:
        return await upload_csv_to_server(files)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex



@app_server.get("/get_zip/", response_model=dict, tags=["File"])
async def get_zip():
    """
    Route for get zip file.

    :return: response model URL.
    """
    try:
        return get_zip_from_server()
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex



@app_server.post("/season_analytic/", response_model=dict, tags=["Analytic"])
async def season_analytic(entry: EntrySeasonAnalyticPipeline):
    """
    Route for season analytic.

    :param entry: Validate pydantic model. [EntryClassicGraduatePipeline]

    :return: response model None.
    """
    try:
        return await season_analytic_pipeline(entry)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/classic_graduate/", response_model=dict, tags=["Graduate"])
async def classic_graduate(entry: EntryClassicGraduatePipeline):
    """
    Route for graduate of classical models.

    :param entry: Validate pydantic model. [EntryClassicGraduatePipeline]

    :return: response model None.
    """
    try:
        return await classic_graduate_pipeline(entry)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


class TimeSeriesGraduateRequest(BaseModel):
    """Обобщённый запрос: любой временной ряд (tidy CSV) + параметры обучения."""
    dataset: dict  # поля EntryTimeSeriesDataset: source, [time_col, target_col, series_id_col, feature_cols]
    graduate: dict  # dictseasonal, models_params, [save_path_weights]


class TimeSeriesInferenceRequest(BaseModel):
    """Обобщённый запрос: любой временной ряд (tidy CSV) + параметры инференса."""
    dataset: dict  # поля EntryTimeSeriesDataset
    inference: dict  # dictseasonal, [future_or_estimate, save_path_weights, plots, ...]


@app_server.post("/timeseries_graduate/", response_model=dict, tags=["Graduate"])
async def timeseries_graduate(entry: TimeSeriesGraduateRequest):
    """Обучение classic-моделей на ПРОИЗВОЛЬНОМ временном ряду (без привязки к домену)."""
    try:
        return await timeseries_graduate_pipeline(entry.dataset, entry.graduate)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/timeseries_inference/", response_model=dict, tags=["Inference"])
async def timeseries_inference(entry: TimeSeriesInferenceRequest):
    """Инференс classic-моделей на ПРОИЗВОЛЬНОМ временном ряду (без привязки к домену)."""
    try:
        return await timeseries_inference_pipeline(entry.dataset, entry.inference)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/timeseries_neiro_graduate/", response_model=dict, tags=["Graduate"])
async def timeseries_neiro_graduate(entry: TimeSeriesGraduateRequest):
    """Обучение нейросети на ПРОИЗВОЛЬНОМ временном ряду (num_variates авто = 3 + число фич)."""
    try:
        return await timeseries_neiro_graduate_pipeline(entry.dataset, entry.graduate)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/timeseries_neiro_inference/", response_model=dict, tags=["Inference"])
async def timeseries_neiro_inference(entry: TimeSeriesInferenceRequest):
    """Инференс нейросети на ПРОИЗВОЛЬНОМ временном ряду (без привязки к домену)."""
    try:
        return await timeseries_neiro_inference_pipeline(entry.dataset, entry.inference)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/neiro_graduate/", response_model=dict, tags=["Graduate"])
async def neiro_graduate(entry: EntryNeiroGraduatePipeline):
    """
    Route for graduate of neiro models.

    :param entry: Validate pydantic model. [EntryNeiroGraduatePipeline]

    :return: response model None.
    """
    try:
        return await neiro_graduate_pipeline(entry)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/classic_inference/", response_model=dict, tags=["Inference"])
async def classic_inference(entry: EntryClassicInferencePipeline):
    """
    Route for inference of classical models.

    :param entry: Validate pydantic model. [EntryClassicInference]

    :return: response model None.
    """
    try:
        return await classic_inference_pipeline(entry)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex


@app_server.post("/neiro_inference/", response_model=dict, tags=["Inference"])
async def neiro_inference(entry: EntryNeiroInferencePipeline):
    """
    Route for inference of neiro models.

    :param entry: Validate pydantic model. [EntryNeiroInferencePipeline]

    :return: response model None.
    """
    try:
        return await neiro_inference_pipeline(entry)
    except HTTPException as ex:
        log.exception("Error", exc_info=ex)
        raise ex



# ---------------------------------------------------------------------------
# Async task queue — обучение без блокировки HTTP-соединения
# ---------------------------------------------------------------------------

def _task_to_dict(t) -> dict:
    return {
        "task_id": t.task_id,
        "operation": t.operation,
        "status": t.status,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
        "result": t.result,
        "error": t.error,
    }


@app_server.get("/tasks/", response_model=list, tags=["Tasks"])
async def list_tasks():
    """Список всех задач (по убыванию времени создания)."""
    return [_task_to_dict(t) for t in task_store.list_all()]


@app_server.get("/tasks/{task_id}", response_model=dict, tags=["Tasks"])
async def get_task(task_id: str):
    """Статус конкретной задачи."""
    record = task_store.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return _task_to_dict(record)


async def _upload_artifacts(task_id: str) -> list[dict]:
    """Загружает веса и графики в S3 после завершения обучения (если S3 настроен)."""
    if storage is None:
        return []
    from timecast import get_paths

    paths = get_paths()
    results: list[dict] = []
    for label, dir_path in [
        ("weights/classic", paths.weights_classic_dir),
        ("weights/neiro", paths.weights_neiro_dir),
        ("plots", paths.plots_dir),
    ]:
        if dir_path:
            uploaded = await asyncio.to_thread(
                storage.upload_dir, dir_path, f"{task_id}/{label}"
            )
            results.extend(uploaded)
    if results:
        log.info("Task %s: загружено %d артефактов в S3", task_id, len(results))
    return results


async def _run_task(task_id: str, coro):
    """Запускает корутину-пайплайн как фоновую задачу, обновляя task_store."""
    task_store.update(task_id, status=TaskStatus.RUNNING)
    try:
        result = await coro
        artifacts = await _upload_artifacts(task_id)
        if artifacts:
            result = {**result, "artifacts": artifacts}
        task_store.update(task_id, status=TaskStatus.DONE, result=result)
    except Exception as exc:
        log.exception("Background task failed", exc_info=exc)
        task_store.update(task_id, status=TaskStatus.FAILED, error=str(exc))


@app_server.post("/classic_graduate/queue/", response_model=dict, tags=["Tasks"])
async def queue_classic_graduate(entry: EntryClassicGraduatePipeline,
                                 background_tasks: BackgroundTasks):
    """Постановка classic-обучения в очередь. Возвращает task_id немедленно."""
    task_id = task_store.create("classic_graduate")
    background_tasks.add_task(_run_task, task_id, classic_graduate_pipeline(entry))
    return {"task_id": task_id, "status": "pending"}


@app_server.post("/neiro_graduate/queue/", response_model=dict, tags=["Tasks"])
async def queue_neiro_graduate(entry: EntryNeiroGraduatePipeline,
                               background_tasks: BackgroundTasks):
    """Постановка нейро-обучения (retail) в очередь."""
    task_id = task_store.create("neiro_graduate")
    background_tasks.add_task(_run_task, task_id, neiro_graduate_pipeline(entry))
    return {"task_id": task_id, "status": "pending"}


@app_server.post("/timeseries_graduate/queue/", response_model=dict, tags=["Tasks"])
async def queue_timeseries_graduate(entry: TimeSeriesGraduateRequest,
                                    background_tasks: BackgroundTasks):
    """Постановка classic-обучения (generic tidy CSV) в очередь."""
    task_id = task_store.create("timeseries_graduate")
    background_tasks.add_task(
        _run_task, task_id, timeseries_graduate_pipeline(entry.dataset, entry.graduate)
    )
    return {"task_id": task_id, "status": "pending"}


@app_server.post("/timeseries_neiro_graduate/queue/", response_model=dict, tags=["Tasks"])
async def queue_timeseries_neiro_graduate(entry: TimeSeriesGraduateRequest,
                                          background_tasks: BackgroundTasks):
    """Постановка нейро-обучения (generic tidy CSV) в очередь."""
    task_id = task_store.create("timeseries_neiro_graduate")
    background_tasks.add_task(
        _run_task, task_id, timeseries_neiro_graduate_pipeline(entry.dataset, entry.graduate)
    )
    return {"task_id": task_id, "status": "pending"}


def run_server():
    import logging

    import uvicorn
    import yaml

    from src import path_to_logging
    uvicorn_log_config = path_to_logging()
    with open(uvicorn_log_config) as f:
        uvicorn_config = yaml.safe_load(f.read())
        logging.config.dictConfig(uvicorn_config)
    reload = os.getenv("DEBUG", "FALSE").upper() == "TRUE"
    uvicorn.run("src.app.server:app",
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("SERVER_PORT", "8000")),
                log_config=uvicorn_log_config, reload=reload)


if __name__ == "__main__":
    log.info("Start run server")
    run_server()
