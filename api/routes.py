"""
FastAPI application: POST /api/reconstructions and GET /api/reconstructions/{job_id}.

This module is imported inside the Modal web_app function body, so it runs only
in the API container. Heavy Modal/GPU imports are never touched here.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import db, r2

# ---------------------------------------------------------------------------
# Tunable limits
# ---------------------------------------------------------------------------

MAX_IMAGES = 30
MAX_FILE_BYTES = 20 * 1024 * 1024    # 20 MB per file
MAX_TOTAL_BYTES = 200 * 1024 * 1024  # 200 MB total
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MIME_TO_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await db.create_pool()
    app.state.r2 = r2.make_client()
    yield
    await app.state.pool.close()


fastapi_app = FastAPI(lifespan=lifespan)

cors_origin = os.environ.get("CORS_ORIGIN", "http://localhost:5173")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[cors_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@fastapi_app.post("/api/reconstructions", status_code=202)
async def create_reconstruction(
    request: Request,
    images: list[UploadFile] = File(...),
):
    if not images:
        raise HTTPException(400, "At least one image is required")
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"Too many images (max {MAX_IMAGES})")

    # Read and validate each upload
    file_data: list[tuple[str, bytes]] = []
    total_bytes = 0
    for i, upload in enumerate(images):
        if upload.content_type not in ALLOWED_MIME:
            raise HTTPException(
                400,
                f"File {upload.filename!r}: unsupported type {upload.content_type!r}. "
                f"Allowed: {sorted(ALLOWED_MIME)}",
            )
        data = await upload.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(
                400,
                f"File {upload.filename!r} is {len(data) // 1024 // 1024} MB "
                f"(max {MAX_FILE_BYTES // 1024 // 1024} MB)",
            )
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                400,
                f"Total upload size exceeds {MAX_TOTAL_BYTES // 1024 // 1024} MB",
            )
        ext = MIME_TO_EXT[upload.content_type]
        file_data.append((f"{i:04d}{ext}", data))

    job_id = uuid.uuid4()
    r2_client = request.app.state.r2

    # Upload images to R2 concurrently
    async def _upload(name: str, data: bytes) -> str:
        key = f"inputs/{job_id}/{name}"
        await asyncio.to_thread(r2.upload_bytes, r2_client, key, data)
        return key

    input_keys = list(await asyncio.gather(*(_upload(n, d) for n, d in file_data)))

    # Insert job row
    pool = request.app.state.pool
    await db.insert_job(pool, job_id, input_keys)

    # Spawn GPU worker fire-and-forget (import lazily to avoid circular imports
    # at module load time)
    from api.main import worker
    await worker.spawn.aio(str(job_id))

    return {"job_id": str(job_id), "status": "queued"}


@fastapi_app.get("/api/reconstructions/{job_id}")
async def get_reconstruction(job_id: str, request: Request):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(404, "Job not found")

    pool = request.app.state.pool
    row = await db.get_job(pool, jid)
    if row is None:
        raise HTTPException(404, "Job not found")

    result: dict[str, Any] | None = None
    if row["status"] == "succeeded" and row["result_key"]:
        r2_client = request.app.state.r2
        ply_url = await asyncio.to_thread(r2.presign_get, r2_client, row["result_key"])
        result = {
            "ply_url": ply_url,
            "point_count": row["point_count"],
            "view_count": row["view_count"],
            "duration_ms": row["duration_ms"],
        }

    return {
        "job_id": str(row["id"]),
        "status": row["status"],
        "created_at": row["created_at"].isoformat(),
        "result": result,
        "error": row["error"],
    }
