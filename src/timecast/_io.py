"""Файловый I/O библиотеки: сохранение графиков и загрузка весов с Hugging Face.

Синхронные функции — вызываются из CPU-bound обучения/инференса. Веб-слой при
необходимости выполняет их в отдельном потоке (`asyncio.to_thread`).
"""
import os
from io import BytesIO
import requests
from timecast._logging import setup_logging

log = setup_logging()


def save_plot_into_server(fig, file_path):
    """Сохраняет matplotlib-фигуру в PNG-файл."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    with open(file_path, "wb") as buffer:
        buffer.write(buf.read())
    buf.close()


def download_all_files_rep_hugging_face(model_name: str, save_dir: str, token: str = None):
    """
    Скачивает файлы (.pt / .json / .zip) из репозитория модели на Hugging Face.

    :param model_name: Репозиторий (например, "GrafTrahula/STORE_NEIRO").
    :param save_dir: Директория для сохранения.
    :param token: (Необязательно) Токен доступа для приватных репозиториев.
    """
    api_url = f"https://huggingface.co/api/models/{model_name}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        pt_files = [f["rfilename"] for f in data.get("siblings", []) if f["rfilename"].endswith(".pt")]
        json_files = [f["rfilename"] for f in data.get("siblings", []) if f["rfilename"].endswith(".json")]
        zip_files = [f["rfilename"] for f in data.get("siblings", []) if f["rfilename"].endswith(".zip")]

        os.makedirs(save_dir, exist_ok=True)

        if pt_files:
            for file_name in pt_files:
                download_file(model_name, file_name, save_dir, headers)
        elif json_files or zip_files:
            for file_name in json_files + zip_files:
                download_file(model_name, file_name, save_dir, headers)
        else:
            log.warning("Нет доступных файлов для скачивания.")

    except requests.RequestException as e:
        log.error(f"Ошибка при работе с Hugging Face API: {e}")


def download_file(model_name: str, file_name: str, save_dir: str, headers: dict = None):
    """Скачивает один файл из репозитория Hugging Face."""
    file_url = f"https://huggingface.co/{model_name}/resolve/main/{file_name}"
    save_path = os.path.join(save_dir, file_name)

    try:
        with requests.get(file_url, headers=headers, stream=True) as response:
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        log.info(f"Файл {file_name} успешно скачан.")
    except requests.RequestException as e:
        log.error(f"Ошибка при скачивании файла {file_name}: {e}")
