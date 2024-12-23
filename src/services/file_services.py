import os
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException, status
from src.utils.create_dir import create_directories_if_not_exist
import pandas as pd
from src.utils.write_file_into_server import write_file_into_server
from src import path_to_project
from datetime import datetime
from env import Env
import zipfile
import uuid
from src.utils.return_url_object import return_url_object
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
        log.info("CSV was successfully uploaded")
        return {"message": "CSV was successfully uploaded"}
    except Exception as ex:
        log.error(ex)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Files not uploaded")


def get_zip_from_server(
) -> None:

    try:
        path_to_plots = os.path.join(path_to_project(), env.__getattr__("PLOTS_PATH"))
        path_to_zip = os.path.join(path_to_project(), env.__getattr__("ZIP_PATH"))
        zip_filename = create_zip_with_unique_name(path_to_plots, path_to_zip)
        log.info("Url was successfully got")
        return {"url": f"{return_url_object(zip_filename, 'zip')}"}
    except Exception as ex:
        log.error(ex)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Zip not send")



def create_zip_with_unique_name(
        source_dir: str,  # Исходная директория, откуда будут собраны файлы
        destination_dir: str,  # Директория, куда будет сохранен zip-архив
        zip_prefix: Optional[str] = "archive"  # Префикс для имени архива
) -> str:
    """
    Собирает все файлы из исходной директории и сохраняет их в другой директории в формате .zip с уникальным именем.

    :param source_dir: Директория, откуда будут собраны файлы.
    :param destination_dir: Директория, куда будет сохранен zip-архив.
    :param zip_prefix: Префикс для имени архива.
    :return: Путь к созданному zip-файлу.
    """
    try:
        # Проверяем существование исходной и создаем директорию назначения, если нужно
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Генерируем уникальное имя для архива
        unique_name = f"{zip_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(destination_dir, unique_name)

        # Создаем архив
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Добавляем файл в архив с сохранением относительного пути
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)

        log.info(f"ZIP архив успешно создан: {zip_path}")
        return unique_name
    except Exception as ex:
        log.error(f"Ошибка при создании ZIP архива: {ex}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании ZIP архива."
        )
