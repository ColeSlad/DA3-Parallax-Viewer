"""
Modal wiring for the async job layer.

Imports the existing Modal app from inference/app.py (same App object) and
registers three functions on it:
  - worker:            GPU function — DA3 inference + point cloud (no gsplat)
  - insertion_worker:  GPU function — TRELLIS asset generation + placement + merge
  - web_app:           lightweight ASGI function that serves the FastAPI routes

Dev:    modal serve api/main.py
Deploy: modal deploy api/main.py

The validate entrypoint in inference/app.py continues to work unchanged:
    modal run inference/app.py::validate
"""
from __future__ import annotations

import modal

# Import the existing app so all functions join the same Modal app.
# inference/app.py only imports modal, time, and pathlib at module level —
# no torch/CUDA — so this is safe in the lightweight API container.
from inference.app import app, da3_image, gsplat_image, weights_volume, WEIGHTS_DIR

secrets = [modal.Secret.from_name("da3-parallax-secrets")]

# Reconstruction worker only needs DA3 — no gsplat CUDA extensions.
reconstruction_image = da3_image.add_local_python_source("api")

# Insertion worker needs gsplat (refit_asset_gaussians) — keep the heavier image.
insertion_image = gsplat_image.add_local_python_source("api")

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
# GPU worker — reconstruction: DA3 inference + point cloud
# ---------------------------------------------------------------------------


@app.function(
    image=reconstruction_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=secrets,
    timeout=600,
)
def worker(job_id: str) -> None:
    """
    Download images from R2, run DA3 inference, upload point cloud PLY, update Postgres.
    The gsplat scene fit has been removed — insertion uses point cloud gaussians directly,
    so there is no need to fit a scene splat on the critical path.
    """
    import json
    import os
    import tempfile
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    from api.db import sync_get_job, sync_update_job
    from api.r2 import download_bytes, make_client, upload_bytes
    from inference.reconstruction import build_point_cloud, ply_to_bytes
    from inference.splat_fit import _infer_world_up

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
            intrinsics = np.asarray(pred.intrinsics, dtype=np.float32)
            extrinsics = np.asarray(pred.extrinsics, dtype=np.float32)
            images_np = np.asarray(pred.processed_images, dtype=np.uint8)

        # All numpy arrays are in memory; tmpdir (image files) is now cleaned up.
        del model
        torch.cuda.empty_cache()
        print(f"[worker:{job_id}] DA3 done, GPU freed")

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
        print(f"[worker:{job_id}] point cloud uploaded ({result.voxel_count:,} pts)")

        # Delete input images — no longer needed once the PLY is uploaded
        for key in input_keys:
            r2_client.delete_object(Bucket=os.environ["R2_BUCKET"], Key=key)
        print(f"[worker:{job_id}] deleted {len(input_keys)} input images from R2")

        world_up = _infer_world_up(extrinsics)
        meta_val = json.dumps({"world_up": world_up.tolist()})

        duration_ms = int((time.perf_counter() - t0) * 1000)
        sync_update_job(
            job_id,
            status="succeeded",
            result_key=result_key,
            point_count=result.voxel_count,
            view_count=len(image_paths),
            duration_ms=duration_ms,
            meta=meta_val,
        )
        print(
            f"[worker:{job_id}] succeeded — "
            f"{result.voxel_count:,} pts, {len(image_paths)} views, {duration_ms}ms"
        )

    except Exception:
        error_msg = traceback.format_exc()[-500:]
        sync_update_job(job_id, status="failed", error=error_msg)
        print(f"[worker:{job_id}] failed:\n{error_msg}")


# ---------------------------------------------------------------------------
# GPU worker — insertion: TRELLIS asset generation + placement + merge
# ---------------------------------------------------------------------------


@app.function(
    image=insertion_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=secrets,
    timeout=1800,  # TRELLIS ~7 min (parallel with point cloud) + refit 500 iter ~1 min
)
def insertion_worker(job_id: str) -> None:
    """
    Generate a 3D asset from a text prompt (via TRELLIS on A100), place it
    into the parent scene's world frame, merge with the scene splat, and
    upload the combined splat to R2.  Postgres is the source of truth.
    """
    import json
    import time
    import traceback
    from pathlib import Path

    import modal
    import numpy as np

    from api.db import sync_get_job, sync_update_job
    from api.r2 import download_bytes, make_client, upload_bytes
    from inference.reconstruction import parse_point_cloud_ply
    from inference.splat_fit import (
        pointcloud_to_gaussians,
        refit_asset_gaussians,
        write_splat_ply,
    )
    from inference.splat_insert import merge_gaussians, place_asset
    from inference.splat_trellis import dict_to_gaussianscene

    r2_client = make_client()
    t0 = time.perf_counter()

    try:
        sync_update_job(job_id, status="running")

        # 1. Load insertion job
        row = sync_get_job(job_id)
        params = row["params"]
        if isinstance(params, str):
            params = json.loads(params)
        parent_id = str(row["parent_id"])

        prompt = params["prompt"]
        position = tuple(float(v) for v in params["position"])
        size_m = float(params["size_m"])

        # 2. Load parent scene metadata
        parent_row = sync_get_job(parent_id)
        if parent_row["status"] != "succeeded":
            raise RuntimeError(f"Parent job {parent_id} is not succeeded")
        pointcloud_key = parent_row["result_key"]
        if not pointcloud_key:
            raise RuntimeError(f"Parent job {parent_id} has no result_key")
        meta = parent_row["meta"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        world_up = np.array(meta["world_up"], dtype=np.float32)

        # 3. Kick off TRELLIS generation on A100 immediately (non-blocking),
        # then do local point-cloud work while it runs on the remote GPU.
        generate_asset_fn = modal.Function.from_name("da3-parallax", "generate_asset")
        print(f"[insertion_worker:{job_id}] spawning generate_asset prompt={prompt!r}")
        asset_call = generate_asset_fn.spawn(prompt=prompt, seed=42)

        pc_data = download_bytes(r2_client, pointcloud_key)
        pc_xyz, pc_rgb = parse_point_cloud_ply(pc_data)
        scene = pointcloud_to_gaussians(pc_xyz, pc_rgb)
        print(f"[insertion_worker:{job_id}] scene loaded: {scene.n:,} point-cloud gaussians")

        # Wait for TRELLIS (may already be done if point-cloud prep took long enough)
        asset_data = asset_call.get()
        print(f"[insertion_worker:{job_id}] asset generated: {asset_data['n_gaussians']:,} gaussians")

        # 4. Refit asset gaussians (reduced iterations — TRELLIS output is already high quality)
        asset_gs = dict_to_gaussianscene(asset_data)
        asset_gs = refit_asset_gaussians(
            asset_gs,
            mesh_xyz=asset_data.get("mesh_xyz"),
            mesh_rgb=asset_data.get("mesh_rgb"),
            n_views=8,
            image_hw=(512, 512),
            n_iters=500,
            init_scale=5e-3,
        )

        # 6. Place asset in scene world frame
        placed = place_asset(
            asset=asset_gs,
            target_center=position,
            target_size=size_m,
            world_up=world_up,
            snap_xyz=None,  # explicit world coordinate; no surface snap
        )
        print(
            f"[insertion_worker:{job_id}] placed {placed.n:,} gaussians "
            f"at {position}  size={size_m}m"
        )

        # 7. Merge and upload
        combined = merge_gaussians(scene, placed)
        combined_path = Path("/tmp/combined.ply")
        write_splat_ply(combined, combined_path)
        combined_data = combined_path.read_bytes()
        combined_key = f"results/{job_id}.ply"
        upload_bytes(r2_client, combined_key, combined_data)
        print(f"[insertion_worker:{job_id}] combined splat uploaded ({combined.n:,} gaussians)")

        duration_ms = int((time.perf_counter() - t0) * 1000)
        sync_update_job(
            job_id,
            status="succeeded",
            result_key=combined_key,
            duration_ms=duration_ms,
            meta=json.dumps({
                "prompt": prompt,
                "position": list(position),
                "size_m": size_m,
                "n_scene_gaussians": scene.n,
                "n_asset_gaussians": placed.n,
                "n_combined_gaussians": combined.n,
            }),
        )
        print(f"[insertion_worker:{job_id}] succeeded in {duration_ms}ms")

    except Exception:
        error_msg = traceback.format_exc()[-500:]
        sync_update_job(job_id, status="failed", error=error_msg)
        print(f"[insertion_worker:{job_id}] failed:\n{error_msg}")


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
