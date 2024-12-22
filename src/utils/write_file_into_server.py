import os
from aiofiles import open as aio_open
from src import path_to_project
from env import Env
env = Env()


async def write_file_into_server(name_object: str, file) -> None:
    # Получаем мя файла
    file_name = file.filename
    # Записываем путь к файлу
    file_location = os.path.join(path_to_project(), env.__getattr__("UPLOAD_DIR"), f"{name_object}")
    # Проверяем существует ли папка, в которой храняться файлы
    os.makedirs(file_location, exist_ok=True)
    # Открывааем файл и записываем данные изображения
    async with aio_open(os.path.join(file_location, file_name), "wb") as buffer:
        await buffer.write(await file.read())
    return file_name
