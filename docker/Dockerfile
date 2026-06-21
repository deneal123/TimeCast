# syntax=docker/dockerfile:1
# Backend TimeCast (prod). Контекст сборки = КОРЕНЬ репозитория
# (чтобы были доступны и backend/, и timecast/).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# curl — для healthcheck; uv — для быстрой установки зависимостей.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

WORKDIR /app

# CPU-колесо torch заранее, чтобы образ не тянул CUDA-сборку (GPU — отдельный образ/индекс).
RUN uv pip install --system 'torch>=2.0.0' --index-url https://download.pytorch.org/whl/cpu

# Библиотека timecast (тянет ML-стек: sktime/statsmodels/duckdb/iTransformer/…).
COPY timecast /opt/timecast
RUN uv pip install --system /opt/timecast

# Web-зависимости backend (зеркалят backend/pyproject.toml).
RUN uv pip install --system \
    fastapi uvicorn python-multipart aiofiles python-dotenv pyyaml rich pydantic

# Исходники приложения.
COPY backend/src /app/src
COPY backend/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["server"]
