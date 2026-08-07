"""Object storage: the local backend's behaviour, the S3 backend's calls, backend selection."""

from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core import storage
from app.core.enums import StorageBackend
from app.core.storage import (
    LocalObjectStore,
    S3ObjectStore,
    StorageError,
    get_object_store,
)

KEY = "32023R1805/32023R1805/abc.html"
HTML = b"<html>act</html>"


def local(tmp_path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "raw")


def test_put_then_get_round_trips(tmp_path):
    store = local(tmp_path)
    store.put(KEY, HTML)
    assert store.get(KEY) == HTML


def test_put_creates_the_directories_the_key_names(tmp_path):
    store = local(tmp_path)
    store.put(KEY, HTML)
    assert (tmp_path / "raw" / KEY).read_bytes() == HTML


def test_exists_is_false_before_put_and_true_after(tmp_path):
    store = local(tmp_path)
    assert not store.exists(KEY)
    store.put(KEY, HTML)
    assert store.exists(KEY)


def test_get_of_a_missing_key_raises_storage_error(tmp_path):
    with pytest.raises(StorageError, match="get failed"):
        local(tmp_path).get(KEY)


def test_put_replaces_the_object_already_at_the_key(tmp_path):
    store = local(tmp_path)
    store.put(KEY, HTML)
    store.put(KEY, b"<html>newer</html>")
    assert store.get(KEY) == b"<html>newer</html>"


def test_a_key_that_would_escape_the_root_is_refused(tmp_path):
    with pytest.raises(StorageError, match="resolve failed"):
        local(tmp_path).get("../../etc/passwd")


class FakeS3:
    """Records calls and answers from a dict, raising whatever a key is scripted to raise."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def _answer(self, operation: str, key: str) -> bytes:
        self.calls.append((operation, key))
        if self.error is not None:
            raise self.error
        if key not in self.objects:
            raise not_found()
        return self.objects[key]

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.calls.append(("put_object", Key))
        if self.error is not None:
            raise self.error
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body = self._answer("get_object", Key)
        return {"Body": _Body(body)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._answer("head_object", Key)
        return {}


class _Body:
    def __init__(self, content: bytes):
        self.content = content

    def read(self) -> bytes:
        return self.content


def not_found() -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


def denied() -> ClientError:
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "HeadObject")


def test_s3_put_then_get_round_trips():
    store = S3ObjectStore(FakeS3(), "regrag-raw")
    store.put(KEY, HTML)
    assert store.get(KEY) == HTML


def test_s3_put_targets_the_configured_bucket():
    client = FakeS3()
    S3ObjectStore(client, "regrag-raw").put(KEY, HTML)
    assert client.calls == [("put_object", KEY)]


def test_s3_exists_reads_a_missing_object_as_absent_not_an_error():
    assert S3ObjectStore(FakeS3(), "regrag-raw").exists(KEY) is False


def test_s3_exists_is_true_for_a_stored_object():
    assert S3ObjectStore(FakeS3({KEY: HTML}), "regrag-raw").exists(KEY) is True


def test_s3_exists_surfaces_a_non_404_client_error():
    client = FakeS3()
    client.error = denied()
    with pytest.raises(StorageError, match="head failed"):
        S3ObjectStore(client, "regrag-raw").exists(KEY)


def test_s3_exists_surfaces_a_transport_failure():
    client = FakeS3()
    client.error = EndpointConnectionError(endpoint_url="https://r2.example")
    with pytest.raises(StorageError, match="head failed"):
        S3ObjectStore(client, "regrag-raw").exists(KEY)


def test_s3_get_of_a_missing_object_raises_storage_error():
    with pytest.raises(StorageError, match="get failed"):
        S3ObjectStore(FakeS3(), "regrag-raw").get(KEY)


def test_s3_put_failure_raises_storage_error():
    client = FakeS3()
    client.error = EndpointConnectionError(endpoint_url="https://r2.example")
    with pytest.raises(StorageError, match="put failed"):
        S3ObjectStore(client, "regrag-raw").put(KEY, HTML)


def test_the_default_backend_is_local_and_rooted_at_the_raw_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", StorageBackend.LOCAL)
    monkeypatch.setattr(storage.config, "RAW_DATA_DIR", tmp_path)
    store = get_object_store()
    assert isinstance(store, LocalObjectStore)
    assert store.root == tmp_path


def test_the_r2_backend_is_an_s3_store_on_the_configured_bucket(monkeypatch):
    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", StorageBackend.R2)
    monkeypatch.setattr(storage.config, "R2_BUCKET", "regrag-raw")
    monkeypatch.setattr(storage, "r2_client", lambda: FakeS3())
    store = get_object_store()
    assert isinstance(store, S3ObjectStore)
    assert store.bucket == "regrag-raw"
