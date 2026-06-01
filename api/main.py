"""
Modal wiring for the async job layer.

Imports the existing Modal app from inference/app.py (same App object) and
registers two new functions on it:
  - worker:   GPU function that runs inference and writes results to Postgres + R2
  - web_app:  lightweight ASGI function that serves the FastAPI routes

Dev:    modal serve api/main.py
Deploy: modal deploy api/main.py

The validate entrypoint in inference/app.py continues to work unchanged:
    modal run inference/app.py::validate
"""
from __future__ import annotations

import modal

# Import the existing app so worker and web_app join the same Modal app.
# inference/app.py only imports modal, time, and pathlib at module level —
# no torch/CUDA — so this is safe in the lightweight API container.
from inference.app import app, da3_image, weights_volume, WEIGHTS_DIR

secrets = [modal.Secret.from_name("da3-parallax-secrets")]

# GPU worker image: extends da3_image (which already has torch, DA3, open3d,
# and the inference package) with DB and R2 deps + the api package.
worker_image = (
    da3_image
    .pip_install("psycopg2-binary", "boto3")
    .add_local_python_source("api")
)

# API image: lightweight, no GPU deps.
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "asyncpg",
        "boto3",
        "python-multipart",
    )
    .add_local_python_source("api")
    .add_local_python_source("inference")  # needed so `from inference.app import app` works
)

# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------


@app.function(
    image=worker_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=secrets,
    timeout=600,
)
def worker(job_id: str) -> None:
    """
    Download images from R2, run DA3 inference, upload PLY, update Postgres.
    Writes nothing back to the API — Postgres is the shared source of truth.
    """
    import tempfile
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    from depth_anything_3.api import DepthAnything3

    from api.db import sync_get_job, sync_update_job
    from api.r2 import download_bytes, make_client, upload_bytes
    from inference.reconstruction import build_point_cloud, ply_to_bytes

    r2_client = make_client()
    t0 = time.perf_counter()

    try:
        sync_update_job(job_id, status="running")

        row = sync_get_job(job_id)
        input_keys: list[str] = row["input_keys"]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths: list[str] = []
            for key in sorted(input_keys):
                name = Path(key).name
                local = Path(tmpdir) / name
                local.write_bytes(download_bytes(r2_client, key))
                image_paths.append(str(local))

            print(f"[worker:{job_id}] {len(image_paths)} images downloaded")

            model = DepthAnything3.from_pretrained(
                "depth-anything/DA3-LARGE-1.1",
                cache_dir=WEIGHTS_DIR,
            )
            model = model.to(device="cuda")

            pred = model.inference(image_paths)

            depths = np.asarray(pred.depth)
            confs = np.asarray(pred.conf)
            intrinsics = np.asarray(pred.intrinsics)
            extrinsics = np.asarray(pred.extrinsics)
            images_np = np.asarray(pred.processed_images, dtype=np.uint8)

        # Reconstruction happens outside the tmpdir context (arrays already in memory)
        result = build_point_cloud(
            depths=depths,
            confs=confs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=images_np,
        )

        ply_data = ply_to_bytes(result)
        result_key = f"results/{job_id}.ply"
        upload_bytes(r2_client, result_key, ply_data)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        sync_update_job(
            job_id,
            status="succeeded",
            result_key=result_key,
            point_count=result.voxel_count,
            view_count=len(image_paths),
            duration_ms=duration_ms,
        )
        print(
            f"[worker:{job_id}] succeeded — "
            f"{result.voxel_count:,} points, {len(image_paths)} views, {duration_ms}ms"
        )

    except Exception:
        error_msg = traceback.format_exc()[-500:]
        sync_update_job(job_id, status="failed", error=error_msg)
        print(f"[worker:{job_id}] failed:\n{error_msg}")


# ---------------------------------------------------------------------------
# ASGI app (FastAPI)
# ---------------------------------------------------------------------------


@app.function(
    image=api_image,
    secrets=secrets,
    min_containers=1,  # keep one warm so first request isn't slow
)
@modal.asgi_app()
def web_app():
    from api.routes import fastapi_app
    return fastapi_app
