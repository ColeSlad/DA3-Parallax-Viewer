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

# Persistent volume for TRELLIS + SDXL-Turbo weights (~20 GB total on first run)
trellis_weights_volume = modal.Volume.from_name("trellis-weights", create_if_missing=True)
TRELLIS_WEIGHTS_DIR = "/trellis-weights"

# Persistent volume for pipeline outputs — survives local client disconnects.
# Use `modal run inference/app.py::download_outputs` to retrieve after a detached run.
outputs_volume = modal.Volume.from_name("da3-outputs", create_if_missing=True)
OUTPUTS_DIR = "/da3-outputs"

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

# TRELLIS image: TRELLIS text/image-to-3D Gaussian generation.
#
# Key deps:
#   spconv-cu124 : sparse 3D convolutions for TRELLIS's structured latent model
#   diffusers    : SDXL-Turbo text-to-image (text path) + TRELLIS internal pipelines
#   transformers : image encoder (DINO/SigLIP) used by TRELLIS
#   rembg        : background removal before TRELLIS (preprocess_image=True)
#
# TRELLIS is installed --no-deps so we control the dep graph and avoid
# torch version clobbering. The explicit dep list above covers what we need
# for Gaussian output; mesh/nvdiffrast deps are intentionally omitted.
#
# GPU: L4 (24 GB) fits TRELLIS-image-large (~16 GB peak) + SDXL-Turbo (~4 GB)
# sequentially. If OOM, bump to gpu="A100".
trellis_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url=_TORCH_INDEX,
    )
    # spconv must be installed before TRELLIS so TRELLIS can import it
    .pip_install("spconv-cu124")
    .pip_install(
        # Pin to torch-2.4.0-era versions.
        # transformers >=4.46 requires torch.float8_e8m0fnu (added in torch 2.5).
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        # TRELLIS --basic deps (from setup.sh)
        "imageio[pillow]",
        "imageio-ffmpeg",
        "tqdm",
        "easydict",
        "opencv-python-headless",
        "scipy",
        "ninja",
        "rembg[cpu]",   # plain rembg has no onnxruntime backend; [cpu] adds onnxruntime
        "onnxruntime",  # explicit in case rembg[cpu] doesn't pull it
        "trimesh",
        "open3d",
        "xatlas",
        "pyvista",
        "pymeshfix",
        "igraph",
        "einops",
        "plyfile",
        "numpy",
        "Pillow",
        "sentencepiece",  # required by FLUX.1-schnell's T5 tokenizer
        "huggingface_hub",
    )
    .run_commands(
        # utils3d: pinned commit from TRELLIS's own setup.sh
        "pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        # xformers: needed for sparse attention (trellis/modules/sparse/attention).
        # That module only accepts 'xformers' or 'flash_attn'; 'sdpa' is not an option.
        # cu121 wheel works on CUDA 12.4 (CUDA runtime is forward-compatible).
        "pip install xformers==0.0.27.post2 --extra-index-url https://download.pytorch.org/whl/cu121",
        # kaolin: needed because flexicubes (a git submodule of TRELLIS) does
        #   `from kaolin.utils.testing import check_tensor` at module level.
        # trellis/representations/__init__.py imports MeshExtractResult eagerly,
        # so this import fires even when formats=["gaussian"] only.
        # NVIDIA only ships torch-2.4.0_cu121 wheel; it works on CUDA 12.4.
        "pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
        # Clone TRELLIS with submodules: flexicubes lives at
        #   trellis/representations/mesh/flexicubes (git submodule).
        # Without --recurse-submodules the flexicubes directory is empty and
        # the import fails. --shallow-submodules keeps the clone fast.
        "git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/microsoft/TRELLIS /opt/trellis",
        # Re-pin torch in case kaolin or any other step upgraded it
        f"pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url {_TORCH_INDEX}",
    )
    .env({
        "PYTHONPATH": "/opt/trellis",
        # Regular attention (trellis/modules/attention): sdpa uses torch's built-in
        # scaled_dot_product_attention, no extra package needed.
        "ATTN_BACKEND": "sdpa",
        # Sparse attention (trellis/modules/sparse/attention): only accepts
        # 'xformers' or 'flash_attn' — sdpa is not supported there.
        # xformers is installed above; this env var selects it.
        "SPARSE_ATTN_BACKEND": "xformers",
        # Point HF cache at the persistent volume mount. Setting this in the image
        # env (rather than os.environ at runtime) ensures it is resolved before any
        # HF library code runs — even if a TRELLIS module imports transformers/diffusers
        # at module level before our function body executes.
        "HF_HOME": TRELLIS_WEIGHTS_DIR,
    })
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


# ---------------------------------------------------------------------------
# TRELLIS asset generation: its own Modal GPU function
# ---------------------------------------------------------------------------


@app.function(
    image=trellis_image,
    gpu="A100",  # FLUX.1-schnell is ~22 GB bf16; L4 (22 GB) is too tight alongside TRELLIS (16 GB)
    volumes={TRELLIS_WEIGHTS_DIR: trellis_weights_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1800,  # 30 min: first run downloads TRELLIS (~15 GB) + FLUX weights
)
def generate_asset(
    prompt: str,
    image_bytes: bytes | None = None,
    seed: int = 42,
) -> dict:
    """
    Generate 3D Gaussians from a text prompt (or image) using TRELLIS.

    Returns a dict of numpy arrays (GaussianScene) in TRELLIS canonical frame.
    Call run_trellis_pipeline() to place them in the scene world frame.
    """
    from inference.splat_trellis import generate_asset_gaussians, gaussianscene_to_dict

    gs, input_image_png, mesh_xyz, mesh_rgb = generate_asset_gaussians(
        prompt=prompt,
        image_bytes=image_bytes,
        seed=seed,
        weights_dir=TRELLIS_WEIGHTS_DIR,
    )
    print(f"[generate_asset] {gs.n:,} gaussians — returning to caller")
    return {
        **gaussianscene_to_dict(gs),
        "n_gaussians": gs.n,
        "prompt": prompt,
        "input_image_png": input_image_png,
        "mesh_xyz": mesh_xyz,
        "mesh_rgb": mesh_rgb,
    }


# ---------------------------------------------------------------------------
# TRELLIS pipeline: DA3 + gsplat fit + place generated asset + render
# ---------------------------------------------------------------------------


@app.function(
    image=gsplat_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights_volume, OUTPUTS_DIR: outputs_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=3600,
)
def run_trellis_pipeline(
    asset_data: dict,                                    # from generate_asset()
    image_data: list[tuple[str, bytes]] | None = None,  # None → SOH example
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
    n_iters: int = 2000,
    asset_center: tuple[float, float, float] | None = None,  # None → scene centroid
    asset_size: float = 0.3,
    snap: bool = True,
) -> dict:
    """
    Full pipeline on GPU:
      1. DA3 inference + point-cloud reconstruction
      2. Gaussian scene fit (Stage 1) via gsplat
      3. Place generated TRELLIS asset into scene world frame (Stage 2)
      4. Merge + render orbit views

    asset_data is the dict returned by generate_asset() — numpy arrays
    representing the asset in TRELLIS canonical frame (Y-up, unit scale).
    place_asset() handles axis alignment to the scene world frame.

    Returns the same dict format as run_splat_pipeline():
      scene_ply, combined_ply, scene_pngs, combined_pngs, metrics
    """
    import glob
    import io
    import tempfile

    import numpy as np
    from PIL import Image

    from depth_anything_3.api import DepthAnything3
    from inference.reconstruction import build_point_cloud
    from inference.splat_fit import fit_gaussians, render_from_cameras, render_around_asset, refit_asset_gaussians, write_splat_ply, _infer_world_up
    from inference.splat_insert import place_asset, merge_gaussians
    from inference.splat_trellis import dict_to_gaussianscene

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
    print(f"[trellis_pipeline] {result.voxel_count:,} points after voxel ds")

    # --- 3. Stage 1: Gaussian scene fit ---
    scene = fit_gaussians(
        xyz=result.xyz, rgb=result.rgb,
        images=images_np, intrinsics=intrinsics, extrinsics=extrinsics,
        n_iters=n_iters, init_scale=voxel_size,
    )

    scene_center = result.xyz.mean(axis=0)

    scene_frames = render_from_cameras(scene, intrinsics, extrinsics, (H, W))

    scene_ply_path = Path("/tmp/scene.ply")
    write_splat_ply(scene, scene_ply_path)
    scene_ply_bytes = scene_ply_path.read_bytes()

    # --- 4. Stage 2: place TRELLIS asset in scene world frame ---
    # Axis alignment: TRELLIS Y-up → scene world_up (inferred from camera poses)
    world_up = _infer_world_up(extrinsics)
    print(f"[trellis_pipeline] world_up={world_up.tolist()}")

    center = tuple(float(v) for v in scene_center) if asset_center is None else asset_center
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

    placed = place_asset(
        asset=asset_gs,
        target_center=center,
        target_size=asset_size,
        world_up=world_up,
        snap_xyz=result.xyz if snap else None,
    )
    print(
        f"[trellis_pipeline] placed {placed.n:,} asset gaussians at {center}"
        f"  size={asset_size:.3f}m  prompt={asset_data.get('prompt', '')!r}"
    )

    combined = merge_gaussians(scene, placed)

    combined_frames = render_from_cameras(combined, intrinsics, extrinsics, (H, W))

    # Tight orbit around the placed asset. Use the actual Gaussian centroid rather
    # than the pre-snap `center` tuple — snap shifts z inside place_asset but the
    # caller's variable isn't updated, so center may have the wrong z.
    placed_means_np = placed.means.numpy()
    # Use bounding-box center, not mean: asymmetric objects (tall hydrant, wider base)
    # bias the mean away from geometric center, cutting off the top in the orbit render.
    actual_asset_center = (
        (placed_means_np.max(axis=0) + placed_means_np.min(axis=0)) / 2.0
    ).astype(np.float32)
    print(f"[trellis_pipeline] actual asset bbox center: {actual_asset_center.tolist()}")
    # Render asset ALONE for clean quality diagnostic (no scene bleed).
    asset_close_frames = render_around_asset(
        placed,
        asset_center=actual_asset_center,
        asset_size=asset_size,
        ref_intrinsic=intrinsics[0],
        image_hw=(H, W),
        world_up=world_up,
        n_frames=8,
    )

    combined_ply_path = Path("/tmp/combined.ply")
    write_splat_ply(combined, combined_ply_path)
    combined_ply_bytes = combined_ply_path.read_bytes()

    def to_png(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    scene_pngs        = [to_png(f) for f in scene_frames]
    combined_pngs     = [to_png(f) for f in combined_frames]
    asset_close_pngs  = [to_png(f) for f in asset_close_frames]
    metrics = {
        "n_views": n_views,
        "n_points": result.voxel_count,
        "n_scene_gaussians": scene.n,
        "n_asset_gaussians": placed.n,
        "n_combined_gaussians": combined.n,
        "asset_center": center,
        "asset_size": asset_size,
        "world_up": world_up.tolist(),
        "prompt": asset_data.get("prompt", ""),
    }

    # Persist all outputs to the outputs volume so they survive a local disconnect.
    # Retrieve with:  modal run inference/app.py::download_outputs
    import json
    vol_out = Path(OUTPUTS_DIR) / "latest"
    vol_out.mkdir(parents=True, exist_ok=True)
    (vol_out / "scene.ply").write_bytes(scene_ply_bytes)
    (vol_out / "combined.ply").write_bytes(combined_ply_bytes)
    if asset_data.get("input_image_png"):
        (vol_out / "trellis_input.png").write_bytes(asset_data["input_image_png"])
    for subdir, pngs in [("scene_orbit", scene_pngs), ("combined_orbit", combined_pngs), ("asset_close", asset_close_pngs)]:
        d = vol_out / subdir
        d.mkdir(exist_ok=True)
        for i, png in enumerate(pngs):
            (d / f"frame_{i:02d}.png").write_bytes(png)
    (vol_out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    outputs_volume.commit()
    print(f"[trellis_pipeline] outputs saved to volume → {vol_out}")

    return {
        "scene_ply": scene_ply_bytes,
        "combined_ply": combined_ply_bytes,
        "scene_pngs": scene_pngs,
        "combined_pngs": combined_pngs,
        "asset_close_pngs": asset_close_pngs,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Local entrypoint: text prompt → placed asset → orbit renders + PLY
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def trellis_insert(
    prompt: str = "a small potted plant",
    image: str = "",          # path to image file; overrides prompt if set
    asset_x: float = 0.0,
    asset_y: float = 0.0,
    asset_z: float = 0.0,
    asset_size: float = 0.3,  # target real-world bounding box in meters
    snap: bool = True,        # snap asset base to scene surface
    seed: int = 42,
    n_iters: int = 2000,
    image_dir: str = "",      # scene images; empty → SOH example
    output_dir: str = "output/trellis",
):
    """
    Generate a text-prompted 3D asset, place it in the example scene, and save
    orbit renders and combined PLY locally.

    Two Modal GPU functions run in sequence:
      1. generate_asset  (trellis_image, L4): text → SDXL-Turbo → TRELLIS → Gaussians
      2. run_trellis_pipeline (gsplat_image, L4): DA3 + fit + place + merge + render

    Usage:
        # Default: "a small potted plant" placed at scene centroid
        modal run inference/app.py::trellis_insert

        # Custom prompt and explicit placement
        modal run inference/app.py::trellis_insert \\
            --prompt "a red fire hydrant" \\
            --asset-x 0.1 --asset-y -0.2 --asset-size 0.4

        # Use an input image instead of text-to-image
        modal run inference/app.py::trellis_insert \\
            --image ./my_object.jpg

        # Custom scene images
        modal run inference/app.py::trellis_insert \\
            --image-dir ./photos --prompt "a ceramic mug"
    """
    asset_center = (asset_x, asset_y, asset_z) if (asset_x or asset_y or asset_z) else None
    image_bytes: bytes | None = None
    if image:
        p = Path(image)
        if not p.is_file():
            raise SystemExit(f"--image {image!r} not found")
        image_bytes = p.read_bytes()
        print(f"Using image input: {p.name} ({len(image_bytes)//1024} KB)")
    else:
        print(f"Using text prompt: {prompt!r}")

    scene_image_data = None
    if image_dir:
        src = Path(image_dir)
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not files:
            raise SystemExit(f"No images found in {image_dir}")
        scene_image_data = [(p.name, p.read_bytes()) for p in files]
        print(f"Loaded {len(files)} scene images from {src.resolve()}")

    print(f"\n=== TRELLIS insert pipeline ===")
    print(f"  prompt    : {prompt!r}")
    print(f"  asset_size: {asset_size} m")
    print(f"  asset_center: {asset_center or 'scene centroid (auto)'}")
    print(f"  snap      : {snap}")
    print(f"  iters     : {n_iters}")

    # Step 1: generate asset Gaussians (TRELLIS container)
    print("\n[1/2] Generating 3D asset ...")
    t0 = time.perf_counter()
    asset_data = generate_asset.remote(
        prompt=prompt,
        image_bytes=image_bytes,
        seed=seed,
    )
    gen_s = time.perf_counter() - t0
    print(f"      → {asset_data['n_gaussians']:,} gaussians in {gen_s:.1f}s")

    # Step 2: scene fit + place + render (gsplat container)
    print("\n[2/2] Fitting scene and placing asset ...")
    t1 = time.perf_counter()
    result = run_trellis_pipeline.remote(
        asset_data=asset_data,
        image_data=scene_image_data,
        n_iters=n_iters,
        asset_center=asset_center,
        asset_size=asset_size,
        snap=snap,
    )
    pipeline_s = time.perf_counter() - t1

    # Save outputs
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "scene.ply").write_bytes(result["scene_ply"])
    (out / "combined.ply").write_bytes(result["combined_ply"])

    if asset_data.get("input_image_png"):
        img_path = out / "trellis_input.png"
        img_path.write_bytes(asset_data["input_image_png"])
        print(f"  TRELLIS input    → {img_path.resolve()}")

    def _write_frames(directory: Path, pngs: list[bytes]) -> None:
        """Clear directory and write fresh frames — prevents stale files from old runs."""
        import shutil
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        for i, png in enumerate(pngs):
            (directory / f"frame_{i:02d}.png").write_bytes(png)

    scene_dir = out / "scene_orbit"
    combined_dir = out / "combined_orbit"
    asset_dir = out / "asset_close"

    _write_frames(scene_dir, result["scene_pngs"])
    _write_frames(combined_dir, result["combined_pngs"])
    _write_frames(asset_dir, result["asset_close_pngs"])

    m = result["metrics"]
    print(
        f"\n=== Results ===\n"
        f"  Prompt           : {m['prompt']!r}\n"
        f"  Asset gaussians  : {m['n_asset_gaussians']:,}\n"
        f"  Scene gaussians  : {m['n_scene_gaussians']:,}\n"
        f"  Combined total   : {m['n_combined_gaussians']:,}\n"
        f"  Asset center     : {m['asset_center']}\n"
        f"  Asset size       : {m['asset_size']} m\n"
        f"  World up         : {m['world_up']}\n"
        f"  Generation time  : {gen_s:.1f} s\n"
        f"  Pipeline time    : {pipeline_s:.1f} s\n"
        f"\n  scene.ply        → {(out / 'scene.ply').resolve()}\n"
        f"  combined.ply     → {(out / 'combined.ply').resolve()}\n"
        f"  scene_orbit/     → {scene_dir.resolve()} ({len(result['scene_pngs'])} frames)\n"
        f"  combined_orbit/  → {combined_dir.resolve()} ({len(result['combined_pngs'])} frames)\n"
        f"  asset_close/     → {asset_dir.resolve()} ({len(result['asset_close_pngs'])} frames)\n"
    )


@app.function(
    image=gsplat_image,
    volumes={OUTPUTS_DIR: outputs_volume},
)
def _read_outputs_from_volume() -> dict:
    """Read the latest trellis_insert outputs from the outputs volume."""
    import json as _json

    vol_out = Path(OUTPUTS_DIR) / "latest"
    if not vol_out.exists():
        raise RuntimeError("No outputs found in volume — has trellis_insert completed yet?")

    result: dict = {}
    result["scene_ply"]    = (vol_out / "scene.ply").read_bytes()
    result["combined_ply"] = (vol_out / "combined.ply").read_bytes()
    result["metrics"]      = _json.loads((vol_out / "metrics.json").read_text())
    img_path = vol_out / "trellis_input.png"
    result["input_image_png"] = img_path.read_bytes() if img_path.exists() else None

    for key, subdir in [
        ("scene_pngs",       "scene_orbit"),
        ("combined_pngs",    "combined_orbit"),
        ("asset_close_pngs", "asset_close"),
    ]:
        d = vol_out / subdir
        result[key] = [f.read_bytes() for f in sorted(d.iterdir())] if d.exists() else []

    return result


@app.local_entrypoint()
def download_outputs(output_dir: str = "output/trellis"):
    """
    Download the latest trellis_insert outputs from the Modal outputs volume.
    Use this after a detached run or if the connection dropped mid-transfer:

        modal run --detach inference/app.py::trellis_insert --prompt "..." ...
        modal run inference/app.py::download_outputs
    """
    import shutil

    print("Downloading outputs from Modal volume ...")
    result = _read_outputs_from_volume.remote()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "scene.ply").write_bytes(result["scene_ply"])
    (out / "combined.ply").write_bytes(result["combined_ply"])
    if result.get("input_image_png"):
        (out / "trellis_input.png").write_bytes(result["input_image_png"])

    def _write_frames(directory: Path, pngs: list[bytes]) -> None:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir()
        for i, png in enumerate(pngs):
            (directory / f"frame_{i:02d}.png").write_bytes(png)

    _write_frames(out / "scene_orbit",    result["scene_pngs"])
    _write_frames(out / "combined_orbit", result["combined_pngs"])
    _write_frames(out / "asset_close",    result["asset_close_pngs"])

    m = result["metrics"]
    print(
        f"\n=== Downloaded outputs ===\n"
        f"  Prompt      : {m.get('prompt', '')!r}\n"
        f"  Output dir  : {out.resolve()}\n"
    )
