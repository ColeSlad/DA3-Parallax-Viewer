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

_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"

# Base image for DA3-only functions: debian slim + torch.
_da3_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url=_TORCH_INDEX,
    )
    .run_commands(
        "git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /opt/da3",
        "pip install -e /opt/da3",
        "pip install opencv-python-headless --upgrade",
    )
    .pip_install(
        "numpy",
        "open3d",
        "Pillow",
        "huggingface_hub",
        "psycopg2-binary",
        "boto3",
    )
)

# DA3 inference image (used by reconstruct / reconstruct_from_bytes / reconstruct_from_soh)
da3_image = _da3_base.add_local_python_source("inference")

# gsplat image: built on pytorch/pytorch devel so nvcc is available at image-build
# time, which lets gsplat compile its CUDA extensions during pip install.
# Can't inherit _da3_base (debian slim has no nvcc), so we reinstall DA3 deps here.
# gsplat_image uses the NVIDIA CUDA devel image (has nvcc, pure pip — no conda).
# The pytorch/pytorch images use conda; pip install -e /opt/da3 upgrades torch
# inside conda and breaks torchvision's C++ ABI. Pure-pip avoids that entirely.
gsplat_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url=_TORCH_INDEX,
    )
    .run_commands(
        "git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /opt/da3",
        "pip install -e /opt/da3",
        "pip install opencv-python-headless --upgrade",
        # DA3 has a loose torch>=2.0 dep; pip may have upgraded it. Re-pin.
        f"pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url {_TORCH_INDEX}",
    )
    .pip_install(
        "numpy",
        "open3d",
        "Pillow",
        "huggingface_hub",
        "psycopg2-binary",
        "boto3",
        "imageio[pillow]",
    )
    .pip_install("gsplat")  # nvcc is in PATH; CUDA extensions compile correctly
    .add_local_python_source("inference")
)

# ---------------------------------------------------------------------------
# GPU inference function
# ---------------------------------------------------------------------------


@app.function(
    image=da3_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
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
    secrets=[modal.Secret.from_name("huggingface-token")],
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
# General-purpose function: accepts raw image bytes from the local machine
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@app.function(
    image=da3_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=600,
)
def reconstruct_from_bytes(
    image_data: list[tuple[str, bytes]],  # [(filename, bytes), ...]
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
) -> tuple[bytes, dict]:
    """
    Accept image bytes from the local machine, write them to a temp dir in
    the container, and run reconstruction.  Returns (ply_bytes, metrics_dict).
    """
    import tempfile
    import time

    import numpy as np
    from depth_anything_3.api import DepthAnything3

    from inference.reconstruction import build_point_cloud, ply_to_bytes

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for name, data in image_data:
            p = Path(tmpdir) / name
            p.write_bytes(data)
            image_paths.append(str(p))
        image_paths.sort()

        print(f"[DA3] Received {len(image_paths)} images: {[Path(p).name for p in image_paths]}")

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
    image_dir: str = "",
):
    """
    Drive reconstruction and write a .ply locally.

    Usage:
        # Use the bundled DA3 SOH example images
        modal run inference/app.py::validate

        # Use your own photos
        modal run inference/app.py::validate --image-dir /path/to/photos

        # Tune quality
        modal run inference/app.py::validate --image-dir ./photos --conf-percentile 40 --voxel-size 0.01
    """
    print(
        f"\n=== DA3-Parallax validation ===\n"
        f"  conf_percentile : {conf_percentile}\n"
        f"  voxel_size      : {voxel_size}\n"
        f"  image_dir       : {image_dir or '(SOH example scene)'}\n"
    )

    t0 = time.perf_counter()

    if image_dir:
        src = Path(image_dir)
        if not src.is_dir():
            raise SystemExit(f"--image-dir {image_dir!r} is not a directory")
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not files:
            raise SystemExit(f"No images ({', '.join(_IMAGE_EXTS)}) found in {image_dir}")
        print(f"  Loading {len(files)} images from {src.resolve()} ...")
        image_data = [(p.name, p.read_bytes()) for p in files]
        ply_bytes, metrics = reconstruct_from_bytes.remote(
            image_data=image_data,
            conf_percentile=conf_percentile,
            voxel_size=voxel_size,
        )
    else:
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


# ---------------------------------------------------------------------------
# Splat pipeline: Stage 1 (fit) + Stage 2 (insert placeholder)
# ---------------------------------------------------------------------------


@app.function(
    image=gsplat_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=3600,  # fit can take 20–30 min for large scenes
)
def run_splat_pipeline(
    image_data: list[tuple[str, bytes]] | None = None,  # None → SOH example
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
    n_iters: int = 2000,
    asset_center: tuple[float, float, float] | None = None,  # None → scene centroid
    asset_size: float = 0.3,
    asset_color: tuple[float, float, float] = (1.0, 0.1, 0.1),
    n_orbit_frames: int = 12,
    snap: bool = False,  # snap asset base to surface below it
) -> dict:
    """
    Full pipeline on GPU:
      1. DA3 inference
      2. Point-cloud reconstruction
      3. Gaussian scene fit (Stage 1)
      4. Placeholder asset insertion (Stage 2)

    Returns dict with:
      scene_ply        bytes   3DGS PLY of the fitted scene
      combined_ply     bytes   3DGS PLY of scene + asset
      scene_pngs       list[bytes]  orbit PNG frames (scene only)
      combined_pngs    list[bytes]  orbit PNG frames (scene + asset)
      metrics          dict
    """
    import glob
    import io
    import tempfile

    import numpy as np
    from PIL import Image

    from depth_anything_3.api import DepthAnything3
    from inference.reconstruction import build_point_cloud
    from inference.splat_fit import fit_gaussians, render_orbit, write_splat_ply
    from inference.splat_insert import make_placeholder_cube, merge_gaussians
    from inference.splat_insert import snap_to_surface as snap_fn

    # --- 1. DA3 inference ---
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1", cache_dir=WEIGHTS_DIR)
    model = model.to("cuda")

    if image_data is None:
        soh_dir = "/opt/da3/assets/examples/SOH"
        image_paths = sorted(glob.glob(f"{soh_dir}/*.jpg") + glob.glob(f"{soh_dir}/*.png"))
        if not image_paths:
            raise RuntimeError(f"No images in {soh_dir}")
        pred = model.inference(image_paths)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            image_paths = []
            for name, data in image_data:
                p = Path(tmp) / name
                p.write_bytes(data)
                image_paths.append(str(p))
            image_paths.sort()
            pred = model.inference(image_paths)

    depths = np.asarray(pred.depth)
    confs = np.asarray(pred.conf)
    intrinsics = np.asarray(pred.intrinsics, dtype=np.float32)
    extrinsics = np.asarray(pred.extrinsics, dtype=np.float32)
    images_np = np.asarray(pred.processed_images, dtype=np.uint8)
    n_views = depths.shape[0]
    H, W = depths.shape[1], depths.shape[2]

    # --- 2. Point-cloud reconstruction ---
    result = build_point_cloud(
        depths=depths, confs=confs, intrinsics=intrinsics,
        extrinsics=extrinsics, processed_images=images_np,
        conf_percentile=conf_percentile, voxel_size=voxel_size,
    )
    print(f"[splat] {result.voxel_count:,} points after voxel ds")

    # --- 3. Stage 1: fit ---
    scene = fit_gaussians(
        xyz=result.xyz, rgb=result.rgb,
        images=images_np, intrinsics=intrinsics, extrinsics=extrinsics,
        n_iters=n_iters, init_scale=voxel_size,
    )

    ref_K = intrinsics[0]  # use first camera's intrinsic for orbit
    scene_center = result.xyz.mean(axis=0)

    scene_frames = render_orbit(
        scene, ref_K, (H, W),
        extrinsics=extrinsics,
        scene_center=scene_center,
        n_frames=n_orbit_frames,
    )

    scene_ply_path = Path("/tmp/scene.ply")
    write_splat_ply(scene, scene_ply_path)
    scene_ply_bytes = scene_ply_path.read_bytes()

    # --- 4. Stage 2: asset insertion ---
    center = tuple(float(v) for v in scene_center) if asset_center is None else asset_center

    if snap:
        center = snap_fn(center, asset_size, result.xyz)
        print(f"[splat] snapped asset center to {center}")

    print(f"[splat] placing {asset_size:.3f}-unit cube at {center}, color={asset_color}")
    asset = make_placeholder_cube(center=center, size=asset_size, color=asset_color)
    combined = merge_gaussians(scene, asset)

    combined_frames = render_orbit(
        combined, ref_K, (H, W),
        extrinsics=extrinsics,
        scene_center=scene_center,
        n_frames=n_orbit_frames,
    )

    combined_ply_path = Path("/tmp/combined.ply")
    write_splat_ply(combined, combined_ply_path)
    combined_ply_bytes = combined_ply_path.read_bytes()

    # Encode frames to PNG bytes
    def to_png(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    return {
        "scene_ply": scene_ply_bytes,
        "combined_ply": combined_ply_bytes,
        "scene_pngs": [to_png(f) for f in scene_frames],
        "combined_pngs": [to_png(f) for f in combined_frames],
        "metrics": {
            "n_views": n_views,
            "n_points": result.voxel_count,
            "n_scene_gaussians": scene.n,
            "n_combined_gaussians": combined.n,
            "asset_center": center,
            "asset_size": asset_size,
        },
    }


@app.local_entrypoint()
def splat_validate(
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
    n_iters: int = 2000,
    asset_x: float = 0.0,
    asset_y: float = 0.0,
    asset_z: float = 0.0,
    asset_size: float = 0.3,
    snap: bool = False,
    image_dir: str = "",
    output_dir: str = "output/splat",
):
    """
    Run the full splat pipeline and save outputs locally.

    Usage:
        # SOH example scene, asset at scene centroid
        modal run inference/app.py::splat_validate

        # Custom images, explicit asset position
        modal run inference/app.py::splat_validate \\
            --image-dir ./photos \\
            --asset-x 0.1 --asset-y -0.2 --asset-z 0.0 \\
            --asset-size 0.3

        # Snap asset base to surface
        modal run inference/app.py::splat_validate --snap
    """
    asset_center = (asset_x, asset_y, asset_z) if (asset_x or asset_y or asset_z) else None

    image_data = None
    if image_dir:
        src = Path(image_dir)
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not files:
            raise SystemExit(f"No images found in {image_dir}")
        image_data = [(p.name, p.read_bytes()) for p in files]
        print(f"Loaded {len(files)} images from {src.resolve()}")

    print(f"\n=== Splat pipeline ===")
    print(f"  iters={n_iters}  voxel={voxel_size}  asset_size={asset_size}")
    if asset_center:
        print(f"  asset_center={asset_center}")
    else:
        print(f"  asset_center=scene_centroid (auto)")

    result = run_splat_pipeline.remote(
        image_data=image_data,
        conf_percentile=conf_percentile,
        voxel_size=voxel_size,
        n_iters=n_iters,
        asset_center=asset_center,
        asset_size=asset_size,
        snap=snap,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "scene.ply").write_bytes(result["scene_ply"])
    (out / "combined.ply").write_bytes(result["combined_ply"])

    scene_dir = out / "scene_orbit"
    scene_dir.mkdir(exist_ok=True)
    for i, png in enumerate(result["scene_pngs"]):
        (scene_dir / f"frame_{i:02d}.png").write_bytes(png)

    combined_dir = out / "combined_orbit"
    combined_dir.mkdir(exist_ok=True)
    for i, png in enumerate(result["combined_pngs"]):
        (combined_dir / f"frame_{i:02d}.png").write_bytes(png)

    m = result["metrics"]
    print(
        f"\n=== Results ===\n"
        f"  Views           : {m['n_views']}\n"
        f"  Point cloud     : {m['n_points']:,}\n"
        f"  Scene gaussians : {m['n_scene_gaussians']:,}\n"
        f"  Combined total  : {m['n_combined_gaussians']:,}\n"
        f"  Asset center    : {m['asset_center']}\n"
        f"  Asset size      : {m['asset_size']}\n"
        f"\n  scene.ply       → {(out / 'scene.ply').resolve()}\n"
        f"  combined.ply    → {(out / 'combined.ply').resolve()}\n"
        f"  scene_orbit/    → {scene_dir.resolve()} ({len(result['scene_pngs'])} frames)\n"
        f"  combined_orbit/ → {combined_dir.resolve()} ({len(result['combined_pngs'])} frames)\n"
    )
