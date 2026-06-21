import os
from aiofiles import open as aio_open
from src import path_to_project
from src.utils.custom_logging import setup_logging
from env import Env
log = setup_logging()
env = Env()


async def write_file_into_server(name_object: str, file) -> None:
    # Путь загрузки файлов (UploadFile — реально асинхронный поток запроса, остаётся async).
    # basename — защита от path traversal на случай вызова в обход валидации сервиса.
    file_name = os.path.basename(file.filename or "")
    file_location = os.path.join(path_to_project(), env.__getattr__("UPLOAD_DIR"), f"{name_object}")
    os.makedirs(file_location, exist_ok=True)
    async with aio_open(os.path.join(file_location, file_name), "wb") as buffer:
        await buffer.write(await file.read())
    return file_name
