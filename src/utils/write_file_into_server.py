import os
from aiofiles import open as aio_open
from io import BytesIO
import requests
import aiohttp
from src import path_to_project
from src.utils.custom_logging import setup_logging
from env import Env
log = setup_logging()
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


async def save_plot_into_server(fig, file_path):
    # Используем BytesIO для сохранения изображения в памяти
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    # Асинхронная запись в файл
    async with aio_open(file_path, 'wb') as buffer:
        await buffer.write(buf.read())
    buf.close()


async def download_all_files_rep_hugging_face(model_name: str, save_dir: str, token: str = None):
    """
    Скачивает все файлы с расширениями .pt, .json и .zip из указанного репозитория на Hugging Face.

    :param model_name: Название репозитория на Hugging Face (например, "GrafTrahula/STORE_NEIRO").
    :param save_dir: Локальная директория, куда будут сохранены файлы.
    :param token: (Необязательно) Токен доступа для приватных репозиториев.
    """
    # URL для получения информации о содержимом репозитория
    api_url = f"https://huggingface.co/api/models/{model_name}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            # Запрашиваем список файлов из репозитория
            async with session.get(api_url) as response:
                response.raise_for_status()
                data = await response.json()

            # Фильтруем файлы с расширениями .pt, .json, .zip
            pt_files = [file["rfilename"] for file in data.get("siblings", []) if file["rfilename"].endswith(".pt")]
            json_files = [file["rfilename"] for file in data.get("siblings", []) if file["rfilename"].endswith(".json")]
            zip_files = [file["rfilename"] for file in data.get("siblings", []) if file["rfilename"].endswith(".zip")]

            # Убедимся, что папка для сохранения существует
            os.makedirs(save_dir, exist_ok=True)

            # Если .pt файлы найдены, скачиваем их
            if pt_files:
                for file_name in pt_files:
                    await download_file(session, model_name, file_name, save_dir)
            else:
                # Если .pt файлы не найдены, скачиваем .json и .zip файлы
                if json_files or zip_files:
                    for file_name in json_files + zip_files:
                        await download_file(session, model_name, file_name, save_dir)
                else:
                    log.warning("Нет доступных файлов для скачивания.")

        except aiohttp.ClientError as e:
            log.error(f"Ошибка при работе с Hugging Face API: {e}")


async def download_file(session: aiohttp.ClientSession, model_name: str, file_name: str, save_dir: str):
    """
    Скачивает файл из репозитория Hugging Face и сохраняет его в указанную директорию.

    :param session: Сессия aiohttp для выполнения запросов.
    :param model_name: Название репозитория.
    :param file_name: Имя файла для скачивания.
    :param save_dir: Локальная директория для сохранения.
    """
    file_url = f"https://huggingface.co/{model_name}/resolve/main/{file_name}"
    save_path = os.path.join(save_dir, file_name)

    try:
        async with session.get(file_url) as response:
            response.raise_for_status()
            async with aio_open(save_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
        log.info(f"Файл {file_name} успешно скачан.")
    except aiohttp.ClientError as e:
        log.error(f"Ошибка при скачивании файла {file_name}: {e}")
