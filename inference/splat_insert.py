"""
Stage 2: Insert a Gaussian asset into a fitted 3DGS scene.

The asset is represented as gaussians, concatenated with the scene gaussians,
and rasterized together — gsplat's depth-sorted alpha compositing handles
occlusion automatically. No manual depth compositing.

Two asset paths:
  - make_placeholder_cube: simple colored cube for quick testing
  - place_asset: places a TRELLIS-generated (or any canonical-frame) GaussianScene
    into the scene world frame with correct axis alignment and scale
"""

from __future__ import annotations

import math

import numpy as np
import torch

from inference.splat_fit import GaussianScene


def make_placeholder_cube(
    center: tuple[float, float, float],
    size: float,
    color: tuple[float, float, float] = (1.0, 0.1, 0.1),  # bright red
    n_per_side: int = 6,
    opacity: float = 0.95,
) -> GaussianScene:
    """
    Solid-colored cube of Gaussians at `center` with side length `size`.

    Places n_per_side^3 Gaussians on a regular 3D grid inside the cube.
    Each Gaussian is sized to cover its grid cell, giving solid coverage.
    """
    half = size / 2.0
    coords = np.linspace(-half, half, n_per_side, dtype=np.float32)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N, 3)
    positions += np.array(center, dtype=np.float32)

    N = len(positions)
    scale = (size / n_per_side) * 0.6  # slightly larger than half-cell to avoid gaps

    means = torch.from_numpy(positions)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    log_scales = torch.full((N, 3), math.log(max(scale, 1e-6)))
    logit_opacities = torch.full((N,), math.log(opacity / (1.0 - opacity)))

    rgb = torch.tensor(color, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
    raw_colors = torch.log(rgb / (1 - rgb)).unsqueeze(0).expand(N, -1).contiguous()

    return GaussianScene(
        means=means,
        quats=quats,
        log_scales=log_scales,
        logit_opacities=logit_opacities,
        raw_colors=raw_colors,
    )


def snap_to_surface(
    center: tuple[float, float, float],
    size: float,
    scene_xyz: np.ndarray,       # (M, 3) point cloud in world space
    search_radius: float = 0.5,
) -> tuple[float, float, float]:
    """
    Adjust the asset center so its base rests on the scene surface below it.

    Finds the highest point cloud point within `search_radius` in XY of
    `center`, then shifts Z so the asset base (center - size/2) sits there.

    Returns an adjusted (x, y, z) center. Falls back to `center` unchanged
    if no nearby surface points are found.
    """
    cx, cy, _ = center
    xy_dist = np.sqrt((scene_xyz[:, 0] - cx) ** 2 + (scene_xyz[:, 1] - cy) ** 2)
    nearby = scene_xyz[xy_dist < search_radius]
    if len(nearby) == 0:
        return center
    surface_z = float(nearby[:, 2].max())
    adjusted_z = surface_z + size / 2.0
    return (cx, cy, adjusted_z)


# ---------------------------------------------------------------------------
# Coordinate-frame helpers for place_asset
# ---------------------------------------------------------------------------

def _rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix R such that R @ src ≈ dst (both unit vectors).
    Uses the Rodrigues / cross-product formula.
    """
    src = src / (np.linalg.norm(src) + 1e-8)
    dst = dst / (np.linalg.norm(dst) + 1e-8)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        # Parallel (c ≈ 1) or anti-parallel (c ≈ -1)
        if c > 0:
            return np.eye(3, dtype=np.float32)
        # 180° rotation: pick any perpendicular axis
        perp = np.array([1., 0., 0.], dtype=np.float32)
        if abs(np.dot(perp, src)) > 0.9:
            perp = np.array([0., 1., 0.], dtype=np.float32)
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32)
        return -np.eye(3, dtype=np.float32) + 2 * np.outer(axis, axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1.0 - c) / (s * s)))


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to unit quaternion [w, x, y, z]."""
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def _quat_left_mul_batch(q_frame: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    Left-multiply each quaternion in qs by q_frame.
    All quaternions in [w, x, y, z] convention.
    q_new = q_frame * q_i  (rotation of gaussian in new world frame)
    """
    w1, x1, y1, z1 = q_frame
    w2 = qs[:, 0]; x2 = qs[:, 1]; y2 = qs[:, 2]; z2 = qs[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Asset placement: canonical frame → scene world frame
# ---------------------------------------------------------------------------

def place_asset(
    asset: GaussianScene,
    target_center: tuple[float, float, float],
    target_size: float,
    world_up: np.ndarray,
    asset_up: np.ndarray | None = None,
    snap_xyz: np.ndarray | None = None,
    snap_radius: float = 0.5,
) -> GaussianScene:
    """
    Transform asset Gaussians from their canonical frame into the scene world frame.

    Designed for TRELLIS output, which uses Y-up, right-handed coordinates with
    positions in roughly [-0.5, 0.5]^3. Works for any canonical-frame asset.

    Steps:
      1. Center at origin (bounding-box center)
      2. Normalize to unit bounding box (longest axis → 1)
      3. Rotate: align asset_up to scene world_up  (axis realignment)
      4. Scale to target_size; adjust log_scales accordingly
      5. Translate to target_center (with optional surface snap)
      6. Rotate quaternions with the same frame rotation (left-multiply)

    Args:
        asset:         GaussianScene in canonical frame (e.g. TRELLIS output)
        target_center: (x, y, z) target position in scene world space
        target_size:   real-world bounding box size in meters (longest axis)
        world_up:      scene world-up vector from _infer_world_up(extrinsics)
        asset_up:      canonical up axis of the asset (default: [0,1,0] for TRELLIS)
        snap_xyz:      (M, 3) scene point cloud for surface snap; None = disabled
        snap_radius:   XY search radius for surface snap in meters

    Returns GaussianScene in scene world frame, ready for merge_gaussians().
    """
    if asset_up is None:
        asset_up = np.array([0., 1., 0.], dtype=np.float32)

    means = asset.means.numpy().copy()   # (N, 3)
    quats = asset.quats.numpy().copy()   # (N, 4) [w,x,y,z]

    # 1. Center at origin
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    means -= (bbox_min + bbox_max) * 0.5

    # 2. Normalize to unit longest axis
    extent = float((bbox_max - bbox_min).max())
    if extent < 1e-6:
        extent = 1.0
    means /= extent  # positions now in approximately [-0.5, 0.5]^3

    # 3. Rotate: TRELLIS Y-up → scene world_up
    # After this rotation means still have unit scale; only orientation changes.
    R = _rotation_from_vectors(asset_up, world_up)
    means = means @ R.T   # (R @ x.T).T = x @ R.T  (apply R to each row)
    q_R = _mat_to_quat_wxyz(R)
    quats = _quat_left_mul_batch(q_R, quats)

    # 4. Scale to target_size
    # means are unit-normalized (longest axis = 1), so * target_size → longest axis = target_size
    means *= target_size
    # Gaussian splat sizes scale the same way: add log(target_size / extent)
    scale_factor = target_size / extent
    log_scales = asset.log_scales + math.log(max(scale_factor, 1e-8))
    # Hard ceiling: no single Gaussian wider than 10% of the target bounding box.
    # Prevents large structural splats from rendering as a featureless blob.
    max_log_scale = math.log(target_size * 0.10)
    log_scales = log_scales.clamp(max=max_log_scale)
    print(
        f"[place_asset] extent={extent:.4f}  scale_factor={scale_factor:.4f}  "
        f"log_scale range [{log_scales.min():.2f}, {log_scales.max():.2f}]  "
        f"linear scale range [{log_scales.exp().min():.4f}m, {log_scales.exp().max():.4f}m]"
    )

    # 5. Translate to target_center (optionally snapping base to surface)
    center = np.array(target_center, dtype=np.float32)
    if snap_xyz is not None:
        cx, cy = float(center[0]), float(center[1])
        xy_dist = np.sqrt((snap_xyz[:, 0] - cx) ** 2 + (snap_xyz[:, 1] - cy) ** 2)
        nearby = snap_xyz[xy_dist < snap_radius]
        if len(nearby) > 0:
            surface_z = float(nearby[:, 2].max())
            center[2] = surface_z + target_size / 2.0
            print(f"[place_asset] surface snap: z={surface_z:.3f} → asset center z={center[2]:.3f}")

    means += center

    return GaussianScene(
        means=torch.from_numpy(means.astype(np.float32)),
        quats=torch.from_numpy(quats.astype(np.float32)),
        log_scales=log_scales,
        logit_opacities=asset.logit_opacities,
        raw_colors=asset.raw_colors,
    )


# ---------------------------------------------------------------------------

def merge_gaussians(scene: GaussianScene, asset: GaussianScene) -> GaussianScene:
    """Concatenate scene and asset gaussians into one combined scene."""
    return GaussianScene(
        means=torch.cat([scene.means, asset.means], dim=0),
        quats=torch.cat([scene.quats, asset.quats], dim=0),
        log_scales=torch.cat([scene.log_scales, asset.log_scales], dim=0),
        logit_opacities=torch.cat([scene.logit_opacities, asset.logit_opacities], dim=0),
        raw_colors=torch.cat([scene.raw_colors, asset.raw_colors], dim=0),
    )
