"""Smoke tests for upload/file API routes."""
from fastapi.testclient import TestClient

from src.app.server import app

client = TestClient(app)


def test_upload_csv_bad_extension():
    r = client.post(
        "/server/upload_csv/",
        files=[("files", ("data.txt", b"a,b\n1,2\n", "text/plain"))],
    )
    assert r.status_code == 400
    assert "Разрешены" in r.json()["detail"]


def test_upload_csv_path_traversal():
    r = client.post(
        "/server/upload_csv/",
        files=[("files", ("../evil.csv", b"a,b\n1,2\n", "text/csv"))],
    )
    assert r.status_code == 400


def test_upload_csv_empty_file():
    r = client.post(
        "/server/upload_csv/",
        files=[("files", ("empty.csv", b"", "text/csv"))],
    )
    assert r.status_code == 400


