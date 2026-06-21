"""Доменные исключения библиотеки TimeCast.

Библиотека не должна зависеть от веб-слоя (FastAPI). Вместо `HTTPException`
бизнес-логика поднимает эти исключения, а перевод в HTTP-ответы выполняется
единым обработчиком на уровне FastAPI (см. `src/pipeline/server.py`).

Поле `http_status` — подсказка для веб-слоя; сама библиотека его не использует.
"""


class TimeCastError(Exception):
    """Базовое исключение библиотеки TimeCast."""

    #: Рекомендуемый HTTP-код для веб-слоя (по умолчанию 500).
    http_status: int = 500


class DataNotFoundError(TimeCastError):
    """Входные данные (например, CSV-файлы) не найдены."""

    http_status = 404


class ValidationFailedError(TimeCastError):
    """Невалидные входные данные/параметры операции."""

    http_status = 400
