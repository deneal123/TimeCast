import os
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException, status
from src.utils.write_file_into_server import write_file_into_server
from src import path_to_project
from datetime import datetime
import zipfile
from src.utils.return_url_object import return_url_object
from src.utils.custom_logging import setup_logging
log = setup_logging()



ALLOWED_UPLOAD_EXTENSIONS = {".csv"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 МБ на файл


def _validate_upload(file: UploadFile) -> str:
    """Проверяет имя/расширение файла и возвращает безопасное basename.

    Защита от path traversal: разрешаем только basename без разделителей пути.
    """
    raw_name = file.filename or ""
    safe_name = os.path.basename(raw_name)
    if not safe_name or safe_name != raw_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Недопустимое имя файла: {raw_name!r}")
    if os.path.splitext(safe_name)[1].lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Разрешены только файлы {sorted(ALLOWED_UPLOAD_EXTENSIONS)}")
    return safe_name


async def upload_csv_to_server(
        files: list[UploadFile]
) -> Dict:

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файлы не переданы")

    saved = []
    for file in files:
        safe_name = _validate_upload(file)
        # Проверка размера: читаем в память (CSV небольшие), затем перематываем поток.
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Пустой файл: {safe_name}")
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"Файл {safe_name} превышает {MAX_UPLOAD_SIZE // (1024 * 1024)} МБ")
        await file.seek(0)
        try:
            await write_file_into_server("data", file)
            saved.append(safe_name)
        except HTTPException:
            raise
        except Exception as ex:
            log.error(ex)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Files not uploaded")

    log.info(f"CSV uploaded: {saved}")
    return {"message": "CSV was successfully uploaded", "files": saved}


def get_zip_from_server(
) -> None:

    try:
        path_to_plots = os.path.join(path_to_project(), os.getenv("PLOTS_PATH", "public/plots"))
        path_to_zip = os.path.join(path_to_project(), os.getenv("ZIP_PATH", "public/zip"))
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
