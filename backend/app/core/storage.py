"""Object storage behind one S3-compatible interface: R2 in prod, local files in dev and tests."""

from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import config
from app.core.enums import StorageBackend

BOTO_ERRORS = (ClientError, BotoCoreError)
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
S3_MAX_ATTEMPTS = 3


class StorageError(Exception):
    """Raised when an object storage operation fails."""

    def __init__(self, operation: str, key: str):
        self.operation = operation
        self.key = key
        super().__init__(f"Storage {operation} failed for '{key}'")


class ObjectStore(Protocol):
    """The object operations the pipeline needs, whichever backend serves them."""

    def put(self, key: str, content: bytes) -> None:
        """Write an object, replacing any object already at the key."""
        ...

    def get(self, key: str) -> bytes:
        """Read an object's bytes, raising StorageError if it is not there."""
        ...

    def exists(self, key: str) -> bool:
        """Whether an object is stored at the key."""
        ...


class LocalObjectStore:
    """Objects as files under a root directory, so dev and tests need no network."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        """The file a key names, refusing keys that would escape the root."""
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise StorageError("resolve", key)
        return path

    def put(self, key: str, content: bytes) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except OSError as exc:
            raise StorageError("put", key) from exc

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except OSError as exc:
            raise StorageError("get", key) from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


class S3ObjectStore:
    """Objects in an S3-compatible bucket; R2 is the one this project points it at."""

    def __init__(self, client: Any, bucket: str):
        self.client = client
        self.bucket = bucket

    def put(self, key: str, content: bytes) -> None:
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        except BOTO_ERRORS as exc:
            raise StorageError("put", key) from exc

    def get(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except BOTO_ERRORS as exc:
            raise StorageError("get", key) from exc

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in NOT_FOUND_CODES:
                return False
            raise StorageError("head", key) from exc
        except BotoCoreError as exc:
            raise StorageError("head", key) from exc
        return True


def r2_client() -> Any:
    """An S3 client pointed at this account's R2 endpoint, retrying transient failures."""
    return boto3.client(
        "s3",
        endpoint_url=config.R2_ENDPOINT_URL,
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=BotoConfig(retries={"mode": "standard", "max_attempts": S3_MAX_ATTEMPTS}),
    )


def get_object_store() -> ObjectStore:
    """The store this environment is configured for."""
    if config.STORAGE_BACKEND is StorageBackend.R2:
        return S3ObjectStore(r2_client(), config.R2_BUCKET)
    return LocalObjectStore(config.RAW_DATA_DIR)
