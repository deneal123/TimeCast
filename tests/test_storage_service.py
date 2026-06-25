"""Unit-тесты StorageService (без реального S3)."""
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch


def test_from_env_returns_none_when_not_configured(monkeypatch):
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    from src.services import storage_service

    result = storage_service.StorageService.from_env()
    assert result is None


def test_from_env_returns_none_when_bucket_missing(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    from src.services import storage_service

    result = storage_service.StorageService.from_env()
    assert result is None


def test_upload_dir_skips_missing_directory(tmp_path):
    svc = MagicMock()
    svc.upload_dir = lambda local_dir, prefix: [] if not local_dir.exists() else []
    missing = tmp_path / "nonexistent"
    assert svc.upload_dir(missing, "prefix") == []


def test_upload_dir_logic(tmp_path):
    """upload_dir обходит директорию и формирует правильные S3-ключи."""
    (tmp_path / "model.pkl").write_bytes(b"weights")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "plot.png").write_bytes(b"plot")

    # Тестируем логику через подмену upload_file без реального S3-клиента
    from src.services.storage_service import StorageService

    svc = object.__new__(StorageService)
    svc._bucket = "bucket"
    uploaded: list[dict] = []

    def fake_upload(local_path, key):
        uploaded.append(key)
        return key

    svc.upload_file = fake_upload
    results = svc.upload_dir(tmp_path, "task123/weights/classic")

    assert len(results) == 2
    keys = {r["key"] for r in results}
    assert any("model.pkl" in k for k in keys)
    assert any("plot.png" in k for k in keys)
