"""Object storage behind one S3-compatible interface: R2 in prod, local files in dev and tests."""

from pathlib import Path
from typing import Protocol

from fastapi import status

from app.core.config import config
from app.core.enums import StorageBackend
from app.core.exceptions import DomainError


class StorageError(DomainError):
    """An object storage operation failed, named by the operation and what refused it."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, operation: str, key: str, reason: object | None = None):
        detail = f": {reason}" if reason is not None else ""
        super().__init__(f"Storage {operation} failed for '{key}'{detail}")


def validate_key(key: str) -> None:
    """Refuse anything but a plain relative path, so both backends accept the same keys.

    Split on the raw string, not a path type: S3 keeps '.' and '' segments literally,
    so a key the local backend would normalise is a different object there.
    """
    if not key or {"", ".", ".."} & set(key.split("/")):
        raise StorageError("access", key, "key is not a plain relative path")


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
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        """The file a key names, once the key is known to stay inside the root."""
        validate_key(key)
        return self.root / key

    def put(self, key: str, content: bytes) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        except OSError as exc:
            raise StorageError("put", key, exc) from exc

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except OSError as exc:
            raise StorageError("get", key, exc) from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


def get_object_store() -> ObjectStore:
    """The store this environment is configured for; R2 credentials are read only if selected."""
    match config.STORAGE_BACKEND:
        case StorageBackend.LOCAL:
            return LocalObjectStore(config.RAW_DATA_DIR)
        case StorageBackend.R2:
            from app.core.s3 import r2_object_store

            return r2_object_store()
