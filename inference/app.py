"""
Modal app: DA3 multi-view inference + point-cloud reconstruction.

Usage:
    modal run inference/app.py::validate
    modal run inference/app.py::validate --conf-percentile 40 --voxel-size 0.01

The `reconstruct` function is the GPU entry point; `validate` is the local driver
that pulls the SOH example images, calls it, and writes output/validation.ply.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("da3-parallax")

# Persistent volume so model weights survive cold starts (~5 GB for DA3-LARGE-1.1)
weights_volume = modal.Volume.from_name("da3-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# ---------------------------------------------------------------------------
# Container image
#
# Installation order matters:
#   1. torch + torchvision + xformers together so pip resolves the correct
#      CUDA-matched xformers wheel in a single pass.
#   2. Clone DA3 and pip install -e it.
#   3. Remaining pure-Python deps.
#
# We use cu121 wheels (CUDA 12.1) which Modal's L4 runtime supports.
# ---------------------------------------------------------------------------

da3_image = (
    modal.Image.debian_slim(python_version="3.11")
    # git for cloning DA3; libgl1 + libglib2.0-0 satisfy opencv's runtime
    # shared-library deps on a headless Debian image
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2",
        "torchvision",
        "xformers",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands(
        "git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /opt/da3",
        "pip install -e /opt/da3",
        # DA3 pulls opencv-python which links against libGL; swap it for the
        # headless wheel so no display server is needed at runtime
        "pip install opencv-python-headless --upgrade",
    )
    .pip_install(
        "numpy",
        "open3d",
        "Pillow",
        "huggingface_hub",
    )
    .add_local_python_source("inference")
)

# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------


@app.function(
    image=da3_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=600,  # 10 min; first run downloads ~5 GB weights
    # Keep one container warm to avoid cold starts during iterative tuning
    # (comment out for production scale-to-zero behaviour)
    # keep_warm=1,
)
def reconstruct(
    image_paths_on_container: list[str],
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
) -> bytes:
    """
    Run DA3 inference on a list of image paths (container-side) and return
    a PLY file as raw bytes.

    Args:
        image_paths_on_container: Absolute paths to images inside the container.
        conf_percentile: Confidence filter percentile (0–100). Higher keeps fewer,
            cleaner points.
        voxel_size: Open3D voxel leaf size in world units. Larger = fewer points.

    Returns:
        Binary PLY bytes.
    """
    import torch
    from depth_anything_3.api import DepthAnything3
    from inference.reconstruction import ReconstructionResult, build_point_cloud, ply_to_bytes

    t0 = time.perf_counter()
    print(f"[DA3] Loading model from {WEIGHTS_DIR} ...")
    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3-LARGE-1.1",
        cache_dir=WEIGHTS_DIR,
    )
    model = model.to(device="cuda")
    print(f"[DA3] Model loaded in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print(f"[DA3] Running inference on {len(image_paths_on_container)} images ...")
    pred = model.inference(image_paths_on_container)
    print(f"[DA3] Inference done in {time.perf_counter() - t1:.1f}s")

    import numpy as np

    depths = np.asarray(pred.depth)
    confs = np.asarray(pred.conf)
    intrinsics = np.asarray(pred.intrinsics)
    extrinsics = np.asarray(pred.extrinsics)
    images_np = np.asarray(pred.processed_images, dtype=np.uint8)

    t2 = time.perf_counter()
    result = build_point_cloud(
        depths=depths,
        confs=confs,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=images_np,
        conf_percentile=conf_percentile,
        voxel_size=voxel_size,
    )
    print(
        f"[DA3] Reconstruction: {result.raw_count:,} raw → "
        f"{result.filtered_count:,} conf-filtered (threshold={result.conf_threshold:.4f}) → "
        f"{result.voxel_count:,} after voxel ds  [{time.perf_counter() - t2:.1f}s]"
    )

    return ply_to_bytes(result)


# ---------------------------------------------------------------------------
# Helper that stages the SOH example images inside the container
# ---------------------------------------------------------------------------


@app.function(
    image=da3_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=600,
)
def reconstruct_from_soh(
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
) -> tuple[bytes, dict]:
    """
    Clone DA3 example images (SOH scene) from the installed repo and run
    reconstruction.  Returns (ply_bytes, metrics_dict).
    """
    import glob
    import time

    from inference.reconstruction import build_point_cloud, ply_to_bytes

    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    # The DA3 repo was cloned to /opt/da3; example images live there
    soh_dir = "/opt/da3/assets/examples/SOH"
    # Collect all jpg/png in the scene directory, sorted for determinism
    image_paths = sorted(
        glob.glob(f"{soh_dir}/*.jpg") + glob.glob(f"{soh_dir}/*.png")
    )
    if not image_paths:
        raise RuntimeError(
            f"No images found in {soh_dir}. "
            "Verify the DA3 repo clone includes the assets directory."
        )
    print(f"[DA3] Found {len(image_paths)} images in SOH scene: {[Path(p).name for p in image_paths]}")

    t0 = time.perf_counter()
    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3-LARGE-1.1",
        cache_dir=WEIGHTS_DIR,
    )
    model = model.to(device="cuda")
    print(f"[DA3] Model loaded in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    pred = model.inference(image_paths)
    infer_s = time.perf_counter() - t1
    print(f"[DA3] Inference done in {infer_s:.1f}s")

    # pred fields may be numpy arrays or torch tensors depending on DA3 version
    depths = np.asarray(pred.depth)
    confs = np.asarray(pred.conf)
    intrinsics = np.asarray(pred.intrinsics)
    extrinsics = np.asarray(pred.extrinsics)
    images_np = np.asarray(pred.processed_images, dtype=np.uint8)

    t2 = time.perf_counter()
    result = build_point_cloud(
        depths=depths,
        confs=confs,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=images_np,
        conf_percentile=conf_percentile,
        voxel_size=voxel_size,
    )
    recon_s = time.perf_counter() - t2
    total_s = time.perf_counter() - t0

    metrics = {
        "view_count": len(image_paths),
        "raw_count": result.raw_count,
        "filtered_count": result.filtered_count,
        "voxel_count": result.voxel_count,
        "conf_threshold": result.conf_threshold,
        "infer_s": round(infer_s, 2),
        "recon_s": round(recon_s, 2),
        "total_s": round(total_s, 2),
    }

    return ply_to_bytes(result), metrics


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def validate(
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
    output: str = "output/validation.ply",
):
    """
    Drive reconstruction against the SOH example scene and write a .ply locally.

    Usage:
        modal run inference/app.py::validate
        modal run inference/app.py::validate --conf-percentile 40 --voxel-size 0.01
    """
    print(
        f"\n=== DA3-Parallax validation ===\n"
        f"  conf_percentile : {conf_percentile}\n"
        f"  voxel_size      : {voxel_size}\n"
    )

    t0 = time.perf_counter()
    ply_bytes, metrics = reconstruct_from_soh.remote(
        conf_percentile=conf_percentile,
        voxel_size=voxel_size,
    )
    wall_s = time.perf_counter() - t0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(ply_bytes)

    print(
        f"\n=== Results ===\n"
        f"  Views              : {metrics['view_count']}\n"
        f"  Raw points         : {metrics['raw_count']:,}\n"
        f"  After conf filter  : {metrics['filtered_count']:,}  "
        f"(conf > p{conf_percentile:.0f} = {metrics['conf_threshold']:.4f})\n"
        f"  After voxel ds     : {metrics['voxel_count']:,}\n"
        f"  Inference time     : {metrics['infer_s']} s\n"
        f"  Reconstruction time: {metrics['recon_s']} s\n"
        f"  Wall-clock total   : {wall_s:.1f} s\n"
        f"\n  Output: {out_path.resolve()}\n"
    )
