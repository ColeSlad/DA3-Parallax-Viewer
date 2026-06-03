"""
Stage 1: Fit a 3DGS scene from DA3 outputs using gsplat.

Gaussian means and colors are initialised from the DA3 point cloud.
Camera poses and intrinsics come directly from DA3 extrinsics/intrinsics —
no COLMAP, no re-derived poses.

DA3 extrinsics convention (world-to-camera):
    X_cam = R @ X_world + t
    viewmat = [[R, t], [0, 0, 0, 1]]   (same convention gsplat expects)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


SH_C0 = 0.28209479177387814  # 0th-order SH coefficient for PLY export


@dataclass
class GaussianScene:
    """Trained 3DGS scene. All tensors are float32, stored on CPU."""
    means: torch.Tensor           # (N, 3)  world positions
    quats: torch.Tensor           # (N, 4)  normalised [w, x, y, z]
    log_scales: torch.Tensor      # (N, 3)  log of per-axis scale
    logit_opacities: torch.Tensor # (N,)    logit of opacity
    raw_colors: torch.Tensor      # (N, 3)  logit-space; sigmoid → [0, 1] RGB

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacities)

    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_colors)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_viewmats(extrinsics: np.ndarray) -> torch.Tensor:
    """(V, 3, 4) world-to-cam → (V, 4, 4) viewmats for gsplat."""
    V = extrinsics.shape[0]
    vm = np.zeros((V, 4, 4), dtype=np.float32)
    vm[:, :3, :] = extrinsics
    vm[:, 3, 3] = 1.0
    return torch.from_numpy(vm)


def _infer_world_up(extrinsics: np.ndarray) -> np.ndarray:
    """
    Estimate world-space up direction from camera poses.
    DA3 uses OpenCV convention: camera Y points down.
    World up = mean of (R.T @ [0,-1,0]) across cameras.
    """
    Rs = extrinsics[:, :3, :3]  # (V, 3, 3) world-to-cam
    # Camera -Y in world space = R.T @ [0,-1,0]
    cam_up_world = np.einsum('vji,j->vi', Rs, np.array([0., -1., 0.], dtype=np.float32))
    up = cam_up_world.mean(0)
    norm = np.linalg.norm(up)
    return up / norm if norm > 1e-6 else np.array([0., 0., 1.], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 1: fit
# ---------------------------------------------------------------------------

def fit_gaussians(
    xyz: np.ndarray,        # (N, 3) float32  point cloud positions
    rgb: np.ndarray,        # (N, 3) uint8    point colors
    images: np.ndarray,     # (V, H, W, 3) uint8  training images
    intrinsics: np.ndarray, # (V, 3, 3) float32
    extrinsics: np.ndarray, # (V, 3, 4) float32  world-to-cam [R|t]
    n_iters: int = 2000,
    init_scale: float = 0.02,
    device: str = "cuda",
) -> GaussianScene:
    """
    Optimise a 3DGS scene from DA3 point cloud + camera data.

    No adaptive densification — DA3 gives a good spatial prior, so the
    optimiser only refines positions, scales, opacities, and colours.
    """
    from gsplat import rasterization

    V, H, W, _ = images.shape
    N = len(xyz)
    print(f"[splat_fit] {N:,} gaussians  {V} views  {H}×{W}  {n_iters} iters")

    # --- initialise parameters ---
    means = torch.from_numpy(xyz.astype(np.float32)).to(device)

    rgb_f = np.clip(rgb.astype(np.float32) / 255.0, 1e-3, 1 - 1e-3)
    raw_colors = torch.from_numpy(np.log(rgb_f / (1 - rgb_f)).astype(np.float32)).to(device)

    log_scales = torch.full((N, 3), math.log(init_scale), dtype=torch.float32, device=device)

    quats = torch.zeros(N, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # w=1 → identity rotation

    logit_opacities = torch.full((N,), -2.0, dtype=torch.float32, device=device)  # ≈ 0.12

    for p in [means, raw_colors, log_scales, quats, logit_opacities]:
        p.requires_grad_(True)

    # --- cameras (fixed) ---
    viewmats = _to_viewmats(extrinsics).to(device)                        # (V, 4, 4)
    Ks = torch.from_numpy(intrinsics.astype(np.float32)).to(device)       # (V, 3, 3)
    gt = torch.from_numpy(images.astype(np.float32) / 255.0).to(device)   # (V, H, W, 3)

    optimizer = torch.optim.Adam([
        {"params": [means],            "lr": 1e-4},
        {"params": [raw_colors],       "lr": 5e-3},
        {"params": [log_scales],       "lr": 5e-3},
        {"params": [quats],            "lr": 1e-3},
        {"params": [logit_opacities],  "lr": 5e-2},
    ])

    for i in range(n_iters):
        optimizer.zero_grad()

        q = F.normalize(quats, dim=-1)
        renders, _, _ = rasterization(
            means=means,
            quats=q,
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(logit_opacities),
            colors=torch.sigmoid(raw_colors),
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            packed=False,
            backgrounds=torch.ones(V, 3, device=device),
        )
        loss = torch.abs(renders - gt).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  iter {i+1:4d}/{n_iters}  loss={loss.item():.4f}"
                  f"  opacity_mean={torch.sigmoid(logit_opacities).mean().item():.3f}")

    print(f"[splat_fit] done — {N:,} gaussians")
    return GaussianScene(
        means=means.detach().cpu(),
        quats=F.normalize(quats, dim=-1).detach().cpu(),
        log_scales=log_scales.detach().cpu(),
        logit_opacities=logit_opacities.detach().cpu(),
        raw_colors=raw_colors.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Orbit renderer
# ---------------------------------------------------------------------------

def render_orbit(
    scene: GaussianScene,
    ref_intrinsic: np.ndarray,          # (3, 3)  focal / principal point for orbit cams
    image_hw: tuple[int, int],          # (H, W)
    extrinsics: Optional[np.ndarray] = None,  # (V, 3, 4) — used to infer world up
    scene_center: Optional[np.ndarray] = None,
    n_frames: int = 12,
    elevation_deg: float = 20.0,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Render n_frames orbit views. Returns list of (H, W, 3) uint8 arrays."""
    from gsplat import rasterization

    H, W = image_hw
    means_np = scene.means.numpy()

    if scene_center is None:
        scene_center = means_np.mean(axis=0).astype(np.float32)

    extent = float(np.linalg.norm(means_np - scene_center, axis=-1).max())
    radius = max(extent * 2.5, 0.1)

    world_up = (
        _infer_world_up(extrinsics) if extrinsics is not None
        else np.array([0., 0., 1.], dtype=np.float32)
    )

    # Push to device
    means_t = scene.means.to(device)
    quats_t = scene.quats.to(device)
    scales_t = torch.exp(scene.log_scales).to(device)
    opacities_t = torch.sigmoid(scene.logit_opacities).to(device)
    colors_t = torch.sigmoid(scene.raw_colors).to(device)
    K = torch.from_numpy(ref_intrinsic.astype(np.float32)).unsqueeze(0).to(device)

    el = math.radians(elevation_deg)
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames

        # Camera position in world space
        cam_pos = scene_center + np.array([
            radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ], dtype=np.float32)

        # Look-at: camera points toward scene_center
        fwd = scene_center - cam_pos
        fwd /= np.linalg.norm(fwd)

        up = world_up - np.dot(world_up, fwd) * fwd  # Gram-Schmidt
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            # degenerate: camera looking straight up/down
            up = np.array([0., 1., 0.], dtype=np.float32)
            up -= np.dot(up, fwd) * fwd
            up /= np.linalg.norm(up)
        else:
            up /= up_norm

        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)  # recompute for orthogonality

        # World-to-cam: rows = [right, -up_cam, fwd] in OpenCV convention
        # OpenCV: X right, Y down, Z forward
        R = np.stack([right, -up, fwd], axis=0)  # (3, 3)
        t = -R @ cam_pos

        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R
        vm[:3, 3] = t
        vm_t = torch.from_numpy(vm).unsqueeze(0).to(device)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=vm_t, Ks=K, width=W, height=H, packed=False,
                backgrounds=torch.ones(1, 3, device=device),
            )
        frames.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# PLY export (3DGS / SuperSplat format)
# ---------------------------------------------------------------------------

def write_splat_ply(scene: GaussianScene, path: str | Path) -> None:
    """
    Write a 3DGS-format PLY (Inria convention, readable by SuperSplat / Luma).

    Properties: x y z  nx ny nz  f_dc_0..2  opacity  scale_0..2  rot_0..3
    opacity and scale are stored in logit/log space (raw optimiser params).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    N = scene.n
    means = scene.means.numpy()
    quats = scene.quats.numpy()
    log_scales = scene.log_scales.numpy()
    logit_opacities = scene.logit_opacities.numpy()
    colors_01 = scene.colors().numpy()
    f_dc = (colors_01 - 0.5) / SH_C0  # SH DC coefficient

    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
        ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4'),
        ('opacity', '<f4'),
        ('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
        ('rot_0', '<f4'), ('rot_1', '<f4'), ('rot_2', '<f4'), ('rot_3', '<f4'),
    ])
    arr = np.zeros(N, dtype=dtype)
    arr['x'] = means[:, 0]; arr['y'] = means[:, 1]; arr['z'] = means[:, 2]
    arr['f_dc_0'] = f_dc[:, 0]; arr['f_dc_1'] = f_dc[:, 1]; arr['f_dc_2'] = f_dc[:, 2]
    arr['opacity'] = logit_opacities
    arr['scale_0'] = log_scales[:, 0]; arr['scale_1'] = log_scales[:, 1]; arr['scale_2'] = log_scales[:, 2]
    arr['rot_0'] = quats[:, 0]; arr['rot_1'] = quats[:, 1]
    arr['rot_2'] = quats[:, 2]; arr['rot_3'] = quats[:, 3]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())

    print(f"[splat_fit] wrote {N:,} gaussians → {path}")
