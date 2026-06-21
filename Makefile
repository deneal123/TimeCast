# Makefile for TimeCast

.PHONY: build test lint run stop frontend-analyze

DOCKERHUB_REPO ?= deneal123
TAG ?= latest
IMAGE := $(DOCKERHUB_REPO):$(TAG)
NO_CACHE ?= false
DOCKER_BUILD_ARGS ?=
BUILD_CONTEXT_BACKEND ?= .
MODE ?= dev

ifeq ($(NO_CACHE),true)
	NO_CACHE_FLAG = --no-cache
else
	NO_CACHE_FLAG =
endif

build:
	@echo "Building images (mode=$(MODE)) via docker/build.sh..."
	@if [ -x docker/build.sh ]; then \
	  BUILD_OPTS=""; \
	  if [ "$(NO_CACHE)" = "true" ]; then BUILD_OPTS="$$BUILD_OPTS --no-cache"; fi; \
	  docker/build.sh --$(MODE) $$BUILD_OPTS; \
	else \
	  docker build $(NO_CACHE_FLAG) $(DOCKER_BUILD_ARGS) -t $(IMAGE) -f backend/docker/Dockerfile $(BUILD_CONTEXT_BACKEND); \
	fi

test:
	@echo "Running library (timecast) tests..."
	cd timecast && uv run pytest -q
	@echo "Running frontend tests..."
	@if [ -d frontend ] && command -v npm >/dev/null 2>&1; then \
		cd frontend && npm run test:ci; \
	else \
		echo "Skipping frontend tests: npm not available"; \
	fi

lint:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files || true
	@if [ -d frontend ] && command -v npm >/dev/null 2>&1; then \
	  echo "Frontend lint + format..."; \
	  npm --prefix frontend run lint || true; \
	  npm --prefix frontend run format; \
	fi

run:
	@echo "Running services via docker/run.sh (mode=$(MODE))"
	@if [ -x docker/run.sh ]; then \
	  docker/run.sh --$(MODE); \
	else \
	  echo "run.sh not found or not executable"; exit 1; \
	fi

stop:
	@echo "Stopping services via docker/run.sh --stop"
	@if [ -x docker/run.sh ]; then \
	  docker/run.sh --stop; \
	else \
	  echo "run.sh not found or not executable"; exit 1; \
	fi

frontend-analyze:
	@echo "Frontend bundle analysis (source-map-explorer)..."
	cd frontend && npm run build:analyze
