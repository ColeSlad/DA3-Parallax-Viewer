"""
Stage 2: Insert a placeholder Gaussian asset into a fitted 3DGS scene.

The asset is represented as gaussians, concatenated with the scene gaussians,
and rasterized together — gsplat's depth-sorted alpha compositing handles
occlusion automatically. No manual depth compositing.
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


def merge_gaussians(scene: GaussianScene, asset: GaussianScene) -> GaussianScene:
    """Concatenate scene and asset gaussians into one combined scene."""
    return GaussianScene(
        means=torch.cat([scene.means, asset.means], dim=0),
        quats=torch.cat([scene.quats, asset.quats], dim=0),
        log_scales=torch.cat([scene.log_scales, asset.log_scales], dim=0),
        logit_opacities=torch.cat([scene.logit_opacities, asset.logit_opacities], dim=0),
        raw_colors=torch.cat([scene.raw_colors, asset.raw_colors], dim=0),
    )
