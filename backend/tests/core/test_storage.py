"""Object storage: the seam's key rules, the local backend's behaviour, backend selection."""

import pytest
from pydantic import ValidationError

from app.core import storage
from app.core.enums import StorageBackend
from app.core.exceptions import DomainError
from app.core.storage import LocalObjectStore, StorageError, get_object_store
from tests.conftest import R2_ENV

KEY = "32023R1805/32023R1805/abc.html"
HTML = b"<html>act</html>"


def test_put_then_get_round_trips(local_store: LocalObjectStore):
    local_store.put(KEY, HTML)
    assert local_store.get(KEY) == HTML


def test_put_creates_the_directories_the_key_names(local_store: LocalObjectStore):
    local_store.put(KEY, HTML)
    assert (local_store.root / KEY).read_bytes() == HTML


def test_exists_is_false_before_put_and_true_after(local_store: LocalObjectStore):
    assert not local_store.exists(KEY)
    local_store.put(KEY, HTML)
    assert local_store.exists(KEY)


def test_get_of_a_missing_key_raises_storage_error(local_store: LocalObjectStore):
    with pytest.raises(StorageError, match="get failed"):
        local_store.get(KEY)


def test_a_failed_operation_names_what_refused_it(local_store: LocalObjectStore):
    """The run row records only this message, so the cause has to be in it."""
    with pytest.raises(StorageError, match="No such file or directory"):
        local_store.get(KEY)


def test_a_storage_failure_is_a_domain_error(local_store: LocalObjectStore):
    """Serving a stored document must fail in the one shape every handler returns."""
    with pytest.raises(DomainError):
        local_store.get(KEY)


def test_put_replaces_the_object_already_at_the_key(local_store: LocalObjectStore):
    local_store.put(KEY, HTML)
    local_store.put(KEY, b"<html>newer</html>")
    assert local_store.get(KEY) == b"<html>newer</html>"


@pytest.mark.parametrize("key", ["../../etc/passwd", "/etc/passwd", "a/./b.html", ""])
def test_a_key_that_is_not_a_plain_relative_path_is_refused(local_store: LocalObjectStore, key):
    with pytest.raises(StorageError, match="access failed"):
        local_store.get(key)


def test_exists_refuses_such_a_key_rather_than_calling_it_absent(local_store: LocalObjectStore):
    """Both backends have to agree here, so exists must not quietly answer for a bad key."""
    with pytest.raises(StorageError, match="access failed"):
        local_store.exists("../../etc/passwd")


def test_the_default_backend_is_local_and_rooted_at_the_raw_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", StorageBackend.LOCAL)
    monkeypatch.setattr(storage.config, "RAW_DATA_DIR", tmp_path)
    store = get_object_store()
    assert isinstance(store, LocalObjectStore)
    assert store.root == tmp_path.resolve()


def test_the_r2_backend_is_an_s3_store_on_the_configured_bucket(monkeypatch):
    from app.core.s3 import S3ObjectStore

    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", StorageBackend.R2)
    monkeypatch.setattr("app.core.s3.r2_client", lambda r2: object())
    for name, value in R2_ENV.items():
        monkeypatch.setenv(name, value)

    store = get_object_store()

    assert isinstance(store, S3ObjectStore)
    assert store.bucket == "regrag-raw"


def test_selecting_r2_without_its_credentials_fails_before_any_work(monkeypatch):
    """The store is built before the run opens, so a misconfigured bucket stops it there."""
    monkeypatch.setattr(storage.config, "STORAGE_BACKEND", StorageBackend.R2)
    for name in R2_ENV:
        monkeypatch.setenv(name, "")

    with pytest.raises(ValidationError, match="R2_BUCKET"):
        get_object_store()
