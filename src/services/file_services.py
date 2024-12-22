import os
from typing import Dict
from fastapi import UploadFile, HTTPException, status
from src.utils.create_dir import create_directories_if_not_exist
import pandas as pd
from src.utils.write_file_into_server import write_file_into_server
from src import path_to_project
from env import Env
from src.utils.custom_logging import setup_logging
log = setup_logging()
env = Env()



async def upload_csv_to_server(
        files: list[UploadFile]
) -> Dict:

    try:
        # Инициализируем путь до папки
        for file in files:
            # Записываем каждый файл на сервер
            filename = await write_file_into_server("data", file)
        return {"message": "success"}
    except Exception as ex:
        log.error(ex)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Files not uploaded")
