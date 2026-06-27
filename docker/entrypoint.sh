#!/usr/bin/env sh
# Entrypoint backend-образа TimeCast.
#   server         -> uvicorn (reload если DEBUG=TRUE; workers из UVICORN_WORKERS)
#   celery <args>  -> celery (требует celery-app; по умолчанию за профилем celery)
#   <иное>         -> выполнить как есть
#
# NOTE: UVICORN_WORKERS > 1 требует Redis-backed TaskStore (см. src/services/task_store.py),
# иначе каждый worker видит только свои задачи. По умолчанию 1.
set -e

case "$1" in
  server)
    WORKERS="${UVICORN_WORKERS:-1}"
    if [ "${DEBUG}" = "TRUE" ]; then
      # reload несовместим с multi-worker
      exec uvicorn src.app.server:app \
        --host 0.0.0.0 --port "${SERVER_PORT:-8000}" \
        --reload
    else
      exec uvicorn src.app.server:app \
        --host 0.0.0.0 --port "${SERVER_PORT:-8000}" \
        --workers "${WORKERS}"
    fi
    ;;
  celery)
    shift
    exec celery "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
