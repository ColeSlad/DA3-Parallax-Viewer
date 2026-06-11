"""
Pure-Python back-projection and point-cloud utilities.

No Modal imports — this module is safe to import from FastAPI or locally.

DA3 extrinsics convention:
    World-to-camera: X_cam = R @ X_world + t
    So X_world = R.T @ (X_cam - t)
"""

from __future__ import annotations

import io
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


@dataclass
class ReconstructionResult:
    xyz: np.ndarray          # (N, 3) float32  world-space
    rgb: np.ndarray          # (N, 3) uint8
    raw_count: int
    filtered_count: int
    voxel_count: int
    conf_threshold: float
    duration_s: float


class PointCloudMetrics(NamedTuple):
    view_count: int
    raw_count: int
    filtered_count: int
    voxel_count: int
    conf_threshold: float
    duration_s: float


def _backproject_view(
    depth: np.ndarray,    # (H, W) float32
    conf: np.ndarray,     # (H, W) float32
    K: np.ndarray,        # (3, 3) float32
    E: np.ndarray,        # (3, 4) float32  world-to-cam [R|t]
    rgb: np.ndarray,      # (H, W, 3) uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz_world, colors, confidences) for all pixels in one view."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    # Camera-space coordinates
    d = depth  # (H, W)
    x_cam = d * (uu - cx) / fx
    y_cam = d * (vv - cy) / fy
    z_cam = d

    # Stack to (H*W, 3)
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)

    R = E[:, :3]   # (3, 3)
    t = E[:, 3]    # (3,)

    # X_world = R.T @ (X_cam - t)
    pts_world = (pts_cam - t[None, :]) @ R  # equivalent to R.T @ (x - t) per row

    colors = rgb.reshape(-1, 3)
    confs = conf.reshape(-1)

    return pts_world.astype(np.float32), colors, confs.astype(np.float32)


def build_point_cloud(
    depths: np.ndarray,          # (N, H, W)
    confs: np.ndarray,           # (N, H, W)
    intrinsics: np.ndarray,      # (N, 3, 3)
    extrinsics: np.ndarray,      # (N, 3, 4)
    processed_images: np.ndarray,  # (N, H, W, 3) uint8
    conf_percentile: float = 25.0,
    voxel_size: float = 0.02,
) -> ReconstructionResult:
    """
    Back-project all views into a shared world-frame point cloud.

    Args:
        conf_percentile: Keep only points above this percentile of the global
            confidence distribution. Range 0–100; higher = fewer, cleaner points.
        voxel_size: Open3D voxel grid leaf size (world units). Larger = fewer points.

    Returns:
        ReconstructionResult with filtered, downsampled cloud.
    """
    t0 = time.perf_counter()

    N = depths.shape[0]
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    for i in range(N):
        xyz, rgb, conf = _backproject_view(
            depths[i], confs[i], intrinsics[i], extrinsics[i], processed_images[i]
        )
        # Skip points with zero or negative depth (invalid)
        valid = xyz[:, 2] > 0
        all_xyz.append(xyz[valid])
        all_rgb.append(rgb[valid])
        all_conf.append(conf[valid])

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    conf = np.concatenate(all_conf, axis=0)
    raw_count = len(xyz)

    # Confidence filter
    threshold = float(np.percentile(conf, conf_percentile))
    mask = conf > threshold
    xyz = xyz[mask]
    rgb = rgb[mask]
    filtered_count = len(xyz)

    # Voxel downsample via open3d
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    xyz_ds = np.asarray(pcd.points, dtype=np.float32)
    rgb_ds = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    voxel_count = len(xyz_ds)

    duration_s = time.perf_counter() - t0

    return ReconstructionResult(
        xyz=xyz_ds,
        rgb=rgb_ds,
        raw_count=raw_count,
        filtered_count=filtered_count,
        voxel_count=voxel_count,
        conf_threshold=threshold,
        duration_s=duration_s,
    )


def write_ply(result: ReconstructionResult, path: str | Path) -> None:
    """Write a binary PLY file from a ReconstructionResult."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = result.xyz
    rgb = result.rgb
    n = len(xyz)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        # Interleave xyz (float32) and rgb (uint8) per vertex
        for i in range(n):
            f.write(struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))
            f.write(struct.pack("BBB", rgb[i, 0], rgb[i, 1], rgb[i, 2]))


def parse_point_cloud_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a point cloud PLY (x y z red green blue) back into (xyz, rgb)."""
    import io as _io
    f = _io.BytesIO(data)
    n_vertices = 0
    while True:
        line = f.readline().decode("ascii").strip()
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        if line == "end_header":
            break
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1)
    return xyz, rgb


def ply_to_bytes(result: ReconstructionResult) -> bytes:
    """Serialize a ReconstructionResult to PLY bytes (for transfer over Modal)."""
    buf = io.BytesIO()
    n = len(result.xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    buf.write(header.encode("ascii"))
    for i in range(n):
        buf.write(struct.pack("<fff", result.xyz[i, 0], result.xyz[i, 1], result.xyz[i, 2]))
        buf.write(struct.pack("BBB", result.rgb[i, 0], result.rgb[i, 1], result.rgb[i, 2]))
    return buf.getvalue()
