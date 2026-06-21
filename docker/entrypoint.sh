#!/usr/bin/env sh
# Entrypoint backend-образа TimeCast.
#   server         -> uvicorn (reload, если DEBUG=TRUE)
#   celery <args>  -> celery (требует celery-app; по умолчанию за профилем celery)
#   <иное>         -> выполнить как есть
set -e

case "$1" in
  server)
    RELOAD=""
    if [ "${DEBUG}" = "TRUE" ]; then RELOAD="--reload"; fi
    exec uvicorn src.pipeline.server:app --host 0.0.0.0 --port "${SERVER_PORT:-8000}" ${RELOAD}
    ;;
  celery)
    shift
    exec celery "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
