"""The S3-compatible backend for object storage; R2 is the one this project points it at."""

from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import R2Config
from app.core.retry import MAX_ATTEMPTS
from app.core.storage import StorageError, validate_key

BOTO_ERRORS = (ClientError, BotoCoreError)
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3ObjectStore:
    """Objects in an S3-compatible bucket, reached through a boto3 client."""

    def __init__(self, client: Any, bucket: str):
        self.client = client
        self.bucket = bucket

    def put(self, key: str, content: bytes) -> None:
        validate_key(key)
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content)
        except BOTO_ERRORS as exc:
            raise StorageError("put", key, exc) from exc

    def get(self, key: str) -> bytes:
        validate_key(key)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except BOTO_ERRORS as exc:
            raise StorageError("get", key, exc) from exc

    def exists(self, key: str) -> bool:
        validate_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in NOT_FOUND_CODES:
                return False
            raise StorageError("head", key, exc) from exc
        except BotoCoreError as exc:
            raise StorageError("head", key, exc) from exc
        return True


def r2_client(r2: R2Config) -> Any:
    """An S3 client pointed at this account's R2 endpoint, retrying transient failures."""
    return boto3.client(
        "s3",
        endpoint_url=r2.R2_ENDPOINT_URL,
        aws_access_key_id=r2.R2_ACCESS_KEY_ID,
        aws_secret_access_key=r2.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=BotoConfig(retries={"mode": "standard", "max_attempts": MAX_ATTEMPTS}),
    )


def r2_object_store() -> S3ObjectStore:
    """The configured R2 bucket as an object store, reading credentials as it is built."""
    r2 = R2Config()
    try:
        client = r2_client(r2)
    except ValueError as exc:
        raise StorageError("connect", r2.R2_BUCKET, exc) from exc
    return S3ObjectStore(client, r2.R2_BUCKET)
