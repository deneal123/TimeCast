import os
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, status, Form
from fastapi.staticfiles import StaticFiles
from typing import Dict
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from src.utils.custom_logging import setup_logging
from env import Env
from src import path_to_project
from src.library.pydantic_models import (EntrySeasonAnalyticPipeline, EntryClassicGraduatePipeline,
                                         EntryClassicInferencePipeline, EntryNeiroGraduatePipeline,
                                         EntryNeiroInferencePipeline)
from src.services.analytic_services import season_analytic_pipeline
from src.services.classic_services import classic_graduate_pipeline, classic_inference_pipeline
from src.services.neiro_services import neiro_graduate_pipeline, neiro_inference_pipeline
from src.services.file_services import upload_csv_to_server, get_zip_from_server

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

env = Env()
log = setup_logging()

app_server = FastAPI(title="TimeCast API", version="1.3.2",
                     description="This API server is intended for the TimeCast project. For rights, contact the service owner.")

app = FastAPI()

app.mount("/server", app_server)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_server.mount("/public", StaticFiles(directory=os.path.join(path_to_project(), "public")), name="public")


# Определяем теги
ServerFileTag = OpenApiTag(name="File", description="Operations file")
ServerAnalyticTag = OpenApiTag(name="Analytic", description="Operations analytic")
ServerGraduateTag = OpenApiTag(name="Graduate", description="Operations graduate")
ServerInferenceTag = OpenApiTag(name="Inference", description="Operations inference")

# Настройка документации с тегами
app_server.openapi_tags = [
    ServerFileTag.model_dump(),
    ServerAnalyticTag.model_dump(),
    ServerGraduateTag.model_dump(),
    ServerInferenceTag.model_dump()
]



@app_server.post("/upload_csv/", response_model=Dict, tags=["File"])
async def upload_csv(files: list[UploadFile] = File(...)):
    """
    Route for upload csv files.

    :param files: CSV Files. [UploadFile]

    :return: response model None.
    """
    try:
        return await upload_csv_to_server(files)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex



@app_server.get("/get_zip/", response_model=Dict, tags=["File"])
async def get_zip():
    """
    Route for get zip file.

    :return: response model URL.
    """
    try:
        return get_zip_from_server()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex



@app_server.post("/season_analytic/", response_model=Dict, tags=["Analytic"])
async def season_analytic(entry: EntrySeasonAnalyticPipeline):
    """
    Route for season analytic.

    :param entry: Validate pydantic model. [EntryClassicGraduatePipeline]

    :return: response model None.
    """
    try:
        return await season_analytic_pipeline(entry)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/classic_graduate/", response_model=Dict, tags=["Graduate"])
async def classic_graduate(entry: EntryClassicGraduatePipeline):
    """
    Route for graduate of classical models.

    :param entry: Validate pydantic model. [EntryClassicGraduatePipeline]

    :return: response model None.
    """
    try:
        return await classic_graduate_pipeline(entry)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/neiro_graduate/", response_model=Dict, tags=["Graduate"])
async def neiro_graduate(entry: EntryNeiroGraduatePipeline):
    """
    Route for graduate of neiro models.

    :param entry: Validate pydantic model. [EntryNeiroGraduatePipeline]

    :return: response model None.
    """
    try:
        return await neiro_graduate_pipeline(entry)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/classic_inference/", response_model=Dict, tags=["Inference"])
async def classic_inference(entry: EntryClassicInferencePipeline):
    """
    Route for inference of classical models.

    :param entry: Validate pydantic model. [EntryClassicInference]

    :return: response model None.
    """
    try:
        return await classic_inference_pipeline(entry)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/neiro_inference/", response_model=Dict, tags=["Inference"])
async def neiro_inference(entry: EntryNeiroInferencePipeline):
    """
    Route for inference of neiro models.

    :param entry: Validate pydantic model. [EntryNeiroInferencePipeline]

    :return: response model None.
    """
    try:
        return await neiro_inference_pipeline(entry)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex



def run_server():
    import logging
    import uvicorn
    import yaml
    from src import path_to_logging
    uvicorn_log_config = path_to_logging()
    with open(uvicorn_log_config, 'r') as f:
        uvicorn_config = yaml.safe_load(f.read())
        logging.config.dictConfig(uvicorn_config)
    if env.__getattr__("DEBUG") == "TRUE":
        reload = True
    elif env.__getattr__("DEBUG") == "FALSE":
        reload = False
    else:
        raise Exception("Not init debug mode in env file")
    uvicorn.run("server:app", host=env.__getattr__("HOST"), port=int(env.__getattr__("SERVER_PORT")),
                log_config=uvicorn_log_config, reload=reload)


if __name__ == "__main__":
    if env.__getattr__("OFF_DATABASE") == "FALSE":
        # Создание датабазы и таблиц, если они не существуют
        log.info("Start create/update database")
        from create_sql import CreateSQL

        create_sql = CreateSQL()
        create_sql.read_sql()

    # Запуск сервера и бота
    log.info("Start run server")
    run_server()
