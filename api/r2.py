"""
Cloudflare R2 helpers via boto3's S3-compatible interface.

All functions are synchronous. Call them with asyncio.to_thread() from async
FastAPI handlers, or directly from the sync Modal GPU worker.
"""
from __future__ import annotations

import os


def make_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_bytes(client, key: str, data: bytes) -> None:
    client.put_object(Bucket=os.environ["R2_BUCKET"], Key=key, Body=data)


def download_bytes(client, key: str) -> bytes:
    resp = client.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
    return resp["Body"].read()


def presign_get(client, key: str, expires: int = 3600) -> str:
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["R2_BUCKET"], "Key": key},
        ExpiresIn=expires,
    )
