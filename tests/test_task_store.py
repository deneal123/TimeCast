"""Unit-тесты TaskStore + smoke для /tasks/ эндпоинтов."""
from fastapi.testclient import TestClient

from src.app.server import app
from src.services.task_store import TaskStatus, TaskStore


# ---------------------------------------------------------------------------
# TaskStore unit tests
# ---------------------------------------------------------------------------

def test_create_and_get():
    store = TaskStore()
    task_id = store.create("classic_graduate")
    record = store.get(task_id)
    assert record is not None
    assert record.task_id == task_id
    assert record.operation == "classic_graduate"
    assert record.status == TaskStatus.PENDING


def test_update_status():
    store = TaskStore()
    task_id = store.create("timeseries_graduate")
    store.update(task_id, status=TaskStatus.RUNNING)
    assert store.get(task_id).status == TaskStatus.RUNNING
    store.update(task_id, status=TaskStatus.DONE, result={"results": {}})
    r = store.get(task_id)
    assert r.status == TaskStatus.DONE
    assert r.result == {"results": {}}


def test_update_error():
    store = TaskStore()
    task_id = store.create("neiro_graduate")
    store.update(task_id, status=TaskStatus.FAILED, error="boom")
    r = store.get(task_id)
    assert r.status == TaskStatus.FAILED
    assert r.error == "boom"


def test_get_missing_returns_none():
    store = TaskStore()
    assert store.get("nonexistent") is None


def test_list_all_contains_all():
    store = TaskStore()
    ids = {store.create("op") for _ in range(3)}
    listed = {t.task_id for t in store.list_all()}
    assert ids.issubset(listed)


def test_eviction_removes_oldest_terminal():
    store = TaskStore(max_tasks=3)
    ids = [store.create("op") for _ in range(3)]
    for tid in ids:
        store.update(tid, status=TaskStatus.DONE)
    new_id = store.create("op")
    assert len(store.list_all()) == 3
    assert store.get(new_id) is not None
    assert store.get(ids[0]) is None  # oldest evicted


def test_eviction_skips_active_tasks():
    store = TaskStore(max_tasks=2)
    active = store.create("op")
    store.update(active, status=TaskStatus.RUNNING)
    done = store.create("op")
    store.update(done, status=TaskStatus.DONE)
    new_id = store.create("op")
    assert store.get(active) is not None  # running is never evicted
    assert store.get(new_id) is not None
    assert store.get(done) is None


# ---------------------------------------------------------------------------
# Smoke: /server/tasks/ endpoints
# ---------------------------------------------------------------------------

client = TestClient(app)


def test_tasks_list_empty():
    resp = client.get("/server/tasks/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_task_not_found():
    resp = client.get("/server/tasks/doesnotexist")
    assert resp.status_code == 404


def test_health():
    resp = client.get("/server/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
