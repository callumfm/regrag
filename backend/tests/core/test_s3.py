"""The S3 backend: the calls it makes, and how it reads botocore's failures."""

from io import BytesIO
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core.s3 import S3ObjectStore, r2_object_store
from app.core.storage import StorageError
from tests.conftest import r2_config

KEY = "32023R1805/32023R1805/abc.html"
HTML = b"<html>act</html>"


class FakeS3:
    """Records calls and answers from a dict, raising whatever the client is scripted to raise."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def _record(self, operation: str, key: str) -> None:
        self.calls.append((operation, key))
        if self.error is not None:
            raise self.error

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self._record("put_object", Key)
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._record("get_object", Key)
        if Key not in self.objects:
            raise not_found()
        return {"Body": BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._record("head_object", Key)
        if Key not in self.objects:
            raise not_found()
        return {}


def not_found() -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


def denied() -> ClientError:
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "HeadObject")


def store(objects: dict[str, bytes] | None = None) -> S3ObjectStore:
    return S3ObjectStore(FakeS3(objects), "regrag-raw")


def test_put_then_get_round_trips():
    s3 = store()
    s3.put(KEY, HTML)
    assert s3.get(KEY) == HTML


def test_put_targets_the_configured_bucket():
    client = FakeS3()
    S3ObjectStore(client, "regrag-raw").put(KEY, HTML)
    assert client.calls == [("put_object", KEY)]


def test_exists_reads_a_missing_object_as_absent_not_an_error():
    assert store().exists(KEY) is False


def test_exists_is_true_for_a_stored_object():
    assert store({KEY: HTML}).exists(KEY) is True


def test_exists_surfaces_a_non_404_client_error():
    client = FakeS3()
    client.error = denied()
    with pytest.raises(StorageError, match="head failed"):
        S3ObjectStore(client, "regrag-raw").exists(KEY)


def test_exists_surfaces_a_transport_failure():
    client = FakeS3()
    client.error = EndpointConnectionError(endpoint_url="https://r2.example")
    with pytest.raises(StorageError, match="head failed"):
        S3ObjectStore(client, "regrag-raw").exists(KEY)


def test_get_of_a_missing_object_raises_storage_error():
    with pytest.raises(StorageError, match="get failed"):
        store().get(KEY)


def test_put_failure_raises_storage_error():
    client = FakeS3()
    client.error = EndpointConnectionError(endpoint_url="https://r2.example")
    with pytest.raises(StorageError, match="put failed"):
        S3ObjectStore(client, "regrag-raw").put(KEY, HTML)


def test_a_failure_names_the_botocore_cause():
    """An operator reading the run row has only this message to tell AccessDenied from a 404."""
    client = FakeS3()
    client.error = denied()
    with pytest.raises(StorageError, match="AccessDenied"):
        S3ObjectStore(client, "regrag-raw").exists(KEY)


def test_a_key_that_is_not_a_plain_relative_path_is_refused():
    """The local backend refuses these, so the S3 backend must not silently accept them."""
    with pytest.raises(StorageError, match="access failed"):
        store().put("../../etc/passwd", HTML)


def test_an_unusable_endpoint_fails_as_a_storage_error(monkeypatch):
    """The CLI contracts on StorageError, so boto3's ValueError must not escape as-is."""
    r2_config(monkeypatch, R2_ACCOUNT_ID="not a host")

    with pytest.raises(StorageError, match="connect failed"):
        r2_object_store()
