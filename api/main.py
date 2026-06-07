"""
Modal wiring for the async job layer.

Imports the existing Modal app from inference/app.py (same App object) and
registers three functions on it:
  - worker:            GPU function — DA3 inference + gsplat scene fit
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
from inference.app import app, gsplat_image, weights_volume, WEIGHTS_DIR

secrets = [modal.Secret.from_name("da3-parallax-secrets")]

# GPU worker image: gsplat_image has DA3 + gsplat + psycopg2 + boto3;
# add the api package last (add_local_* must come after all build steps).
worker_image = gsplat_image.add_local_python_source("api")

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
# GPU worker — reconstruction: DA3 inference + point cloud + gsplat scene fit
# ---------------------------------------------------------------------------


@app.function(
    image=worker_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=secrets,
    timeout=3600,  # gsplat fit adds ~20-30 min on top of ~5 min DA3
)
def worker(job_id: str) -> None:
    """
    Download images from R2, run DA3 inference, fit a gsplat scene, upload
    both the point cloud PLY and the splat PLY, then update Postgres.
    Postgres is the single source of truth; nothing is returned to the API.
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
    from inference.splat_fit import _infer_world_up, fit_gaussians, write_splat_ply

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

        # --- gsplat scene fit ---
        scene = fit_gaussians(
            xyz=result.xyz,
            rgb=result.rgb,
            images=images_np,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            n_iters=2000,
            init_scale=0.02,
        )
        splat_path = Path("/tmp/scene.ply")
        write_splat_ply(scene, splat_path)
        splat_data = splat_path.read_bytes()
        splat_key = f"results/{job_id}_splat.ply"
        upload_bytes(r2_client, splat_key, splat_data)
        print(f"[worker:{job_id}] splat uploaded ({scene.n:,} gaussians)")

        world_up = _infer_world_up(extrinsics)
        meta_val = json.dumps({
            "world_up": world_up.tolist(),
            "n_scene_gaussians": scene.n,
        })

        duration_ms = int((time.perf_counter() - t0) * 1000)
        sync_update_job(
            job_id,
            status="succeeded",
            result_key=result_key,
            splat_key=splat_key,
            point_count=result.voxel_count,
            view_count=len(image_paths),
            duration_ms=duration_ms,
            meta=meta_val,
        )
        print(
            f"[worker:{job_id}] succeeded — "
            f"{result.voxel_count:,} pts, {scene.n:,} gaussians, "
            f"{len(image_paths)} views, {duration_ms}ms"
        )

    except Exception:
        error_msg = traceback.format_exc()[-500:]
        sync_update_job(job_id, status="failed", error=error_msg)
        print(f"[worker:{job_id}] failed:\n{error_msg}")


# ---------------------------------------------------------------------------
# GPU worker — insertion: TRELLIS asset generation + placement + merge
# ---------------------------------------------------------------------------


@app.function(
    image=worker_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=secrets,
    timeout=3600,  # generate_asset (A100) ~10 min + refit ~10 min
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
    from inference.splat_fit import (
        read_splat_ply,
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
        splat_key = parent_row["splat_key"]
        if not splat_key:
            raise RuntimeError(f"Parent job {parent_id} has no splat_key")
        meta = parent_row["meta"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        world_up = np.array(meta["world_up"], dtype=np.float32)

        # 3. Download and deserialize scene splat
        splat_data = download_bytes(r2_client, splat_key)
        scene = read_splat_ply(splat_data)
        print(f"[insertion_worker:{job_id}] scene loaded: {scene.n:,} gaussians")

        # 4. Generate asset Gaussians via TRELLIS (runs on A100 in its own container)
        generate_asset = modal.Function.from_name("da3-parallax", "generate_asset")
        print(f"[insertion_worker:{job_id}] calling generate_asset prompt={prompt!r}")
        asset_data = generate_asset.remote(prompt=prompt, seed=42)
        print(f"[insertion_worker:{job_id}] asset generated: {asset_data['n_gaussians']:,} gaussians")

        # 5. Deserialize + refit asset Gaussians against mesh training views
        asset_gs = dict_to_gaussianscene(asset_data)
        asset_gs = refit_asset_gaussians(
            asset_gs,
            mesh_xyz=asset_data.get("mesh_xyz"),
            mesh_rgb=asset_data.get("mesh_rgb"),
            n_views=20,
            image_hw=(512, 512),
            n_iters=2000,
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
