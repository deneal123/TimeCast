"""In-memory store для фоновых задач обучения.

Достаточно для однопроцессного сервера; при масштабировании
замените `task_store` на Redis-backed реализацию с тем же интерфейсом.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskRecord:
    task_id: str
    operation: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    result: Any = None
    error: str | None = None


_MAX_TASKS = 200


class TaskStore:
    def __init__(self, max_tasks: int = _MAX_TASKS) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = Lock()
        self._max = max_tasks

    def _evict_oldest_terminal(self) -> None:
        """Remove the oldest done/failed task to stay within capacity. Called under lock."""
        terminal = sorted(
            (t for t in self._tasks.values() if t.status in (TaskStatus.DONE, TaskStatus.FAILED)),
            key=lambda t: t.created_at,
        )
        to_drop = len(self._tasks) - self._max
        for record in terminal[:to_drop]:
            del self._tasks[record.task_id]

    def create(self, operation: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        record = TaskRecord(task_id=task_id, operation=operation, created_at=now, updated_at=now)
        with self._lock:
            self._tasks[task_id] = record
            if len(self._tasks) > self._max:
                self._evict_oldest_terminal()
        return task_id

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_all(self) -> list[TaskRecord]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def update(self, task_id: str, **kwargs: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            record = self._tasks.get(task_id)
            if record:
                for k, v in kwargs.items():
                    setattr(record, k, v)
                record.updated_at = now


task_store = TaskStore()
