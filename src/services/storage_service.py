"""Загрузка артефактов (веса, графики) в S3-совместимое хранилище.

Настройка через переменные окружения:
    S3_ENDPOINT    — URL хранилища (http://minio:9000 / Selectel / AWS)
    S3_BUCKET      — имя бакета (создаётся автоматически, если отсутствует)
    S3_ACCESS_KEY  — access key
    S3_SECRET_KEY  — secret key
    S3_REGION      — регион (по умолчанию us-east-1)

Если S3_ENDPOINT или S3_BUCKET не заданы — сервис отключён, методы возвращают [].
"""
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


def _get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


class StorageService:
    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str,
                 region: str = "us-east-1") -> None:
        if not _BOTO3_AVAILABLE:
            raise RuntimeError("boto3 не установлен. Выполните: pip install boto3")
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self._bucket)
                log.info("Создан S3-бакет: %s", self._bucket)
            else:
                raise

    def upload_file(self, local_path: Path, s3_key: str) -> str:
        """Загружает файл в S3, возвращает s3_key."""
        self._client.upload_file(str(local_path), self._bucket, s3_key)
        log.debug("Загружен артефакт: s3://%s/%s", self._bucket, s3_key)
        return s3_key

    def upload_dir(self, local_dir: Path, prefix: str) -> list[dict]:
        """Рекурсивно загружает директорию. Возвращает [{key, size}] для новых файлов."""
        if not local_dir.exists():
            return []
        results = []
        for fp in local_dir.rglob("*"):
            if not fp.is_file():
                continue
            relative = fp.relative_to(local_dir)
            key = f"{prefix}/{relative}".replace("\\", "/")
            try:
                key = self.upload_file(fp, key)
                results.append({"key": key, "size": fp.stat().st_size})
            except Exception:
                log.exception("Ошибка загрузки %s → s3://%s/%s", fp, self._bucket, key)
        return results

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Генерирует временную ссылку для скачивания артефакта."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires,
        )

    @classmethod
    def from_env(cls) -> "StorageService | None":
        endpoint = _get_env("S3_ENDPOINT")
        bucket = _get_env("S3_BUCKET")
        if not endpoint or not bucket:
            return None
        access_key = _get_env("S3_ACCESS_KEY", "")
        secret_key = _get_env("S3_SECRET_KEY", "")
        region = _get_env("S3_REGION", "us-east-1")
        try:
            return cls(endpoint, bucket, access_key or "", secret_key or "", region or "us-east-1")
        except Exception:
            log.exception("Не удалось подключиться к S3 (%s). Загрузка артефактов отключена.", endpoint)
            return None


# Синглтон — инициализируется один раз при старте сервера (или None, если не настроен).
storage: StorageService | None = StorageService.from_env()
