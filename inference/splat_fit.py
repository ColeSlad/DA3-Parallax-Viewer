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
    scale_reg: float = 0.0,  # L1 penalty on mean Gaussian scale; prevents scale growth
    scale_cap: Optional[float] = None,  # hard ceiling on Gaussian scale; None → init_scale * 10
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

    logit_opacities = torch.full((N,), 0.0, dtype=torch.float32, device=device)   # ≈ 0.50

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
        if scale_reg > 0.0:
            loss = loss + scale_reg * torch.exp(log_scales).mean()
        loss.backward()
        optimizer.step()

        # Clamp scales to prevent the degenerate large+transparent solution:
        # without this, scales grow freely and optimizer reduces opacity to compensate.
        _cap = scale_cap if scale_cap is not None else init_scale * 10
        with torch.no_grad():
            log_scales.clamp_(max=math.log(_cap))

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


def render_orbit_with_cameras(
    scene: GaussianScene,
    image_hw: tuple[int, int],
    n_frames: int = 16,
    elevation_deg: float = 20.0,
    world_up: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a synthetic orbit and return (images, extrinsics, intrinsics).
    Used to generate multi-view training data for refit_asset_gaussians.
    """
    from gsplat import rasterization

    H, W = image_hw
    means_np = scene.means.numpy()
    scene_center = ((means_np.max(axis=0) + means_np.min(axis=0)) / 2).astype(np.float32)
    extent = float(np.linalg.norm(means_np - scene_center, axis=-1).max())
    radius = max(extent * 2.5, 0.1)

    if world_up is None:
        world_up = np.array([0., 0., 1.], dtype=np.float32)

    f = (W / 2.0) / math.atan2(extent * 0.6, radius)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float32)
    K_t = torch.from_numpy(K).unsqueeze(0).to(device)

    means_t = scene.means.to(device)
    quats_t = scene.quats.to(device)
    scales_t = torch.exp(scene.log_scales).to(device)
    opacities_t = torch.sigmoid(scene.logit_opacities).to(device)
    colors_t = torch.sigmoid(scene.raw_colors).to(device)

    el = math.radians(elevation_deg)
    frames, exts = [], []

    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames
        cam_pos = scene_center + np.array([
            radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ], dtype=np.float32)

        fwd = scene_center - cam_pos
        fwd /= np.linalg.norm(fwd)
        up = world_up - np.dot(world_up, fwd) * fwd
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            up = np.array([0., 1., 0.], dtype=np.float32)
            up -= np.dot(up, fwd) * fwd
            up /= np.linalg.norm(up)
        else:
            up /= up_norm
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        R = np.stack([right, -up, fwd], axis=0)
        t = -R @ cam_pos
        ext = np.concatenate([R, t[:, None]], axis=1).astype(np.float32)
        vm = np.eye(4, dtype=np.float32); vm[:3, :] = ext
        vm_t = torch.from_numpy(vm).unsqueeze(0).to(device)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=vm_t, Ks=K_t, width=W, height=H, packed=False,
                backgrounds=torch.ones(1, 3, device=device),
            )
        frames.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        exts.append(ext)

    imgs = np.stack(frames, axis=0)
    exts_np = np.stack(exts, axis=0)
    Ks_np = np.tile(K[None], (n_frames, 1, 1))
    return imgs, exts_np, Ks_np


def refit_asset_gaussians(
    asset: GaussianScene,
    mesh_xyz: Optional[np.ndarray] = None,
    mesh_rgb: Optional[np.ndarray] = None,
    n_views: int = 20,
    image_hw: tuple[int, int] = (512, 512),
    n_iters: int = 2000,
    init_scale: float = 5e-3,
    device: str = "cuda",
) -> GaussianScene:
    """
    Re-fit tight gsplat Gaussians against multi-view renders of the TRELLIS asset.

    Uses mesh surface points (if available) as both training view source and
    initialization — gives sharper results than the raw TRELLIS VAE Gaussians.
    Falls back to isotropic TRELLIS Gaussians when mesh is unavailable.
    """
    if mesh_xyz is not None and mesh_rgb is not None:
        N_mesh = len(mesh_xyz)
        rgb_clipped = np.clip(mesh_rgb, 1e-3, 1 - 1e-3).astype(np.float32)
        render_scene = GaussianScene(
            means=torch.from_numpy(mesh_xyz),
            quats=torch.cat([torch.ones(N_mesh, 1), torch.zeros(N_mesh, 3)], dim=1),
            log_scales=torch.full((N_mesh, 3), math.log(4e-3)),
            logit_opacities=torch.full((N_mesh,), 4.0),
            raw_colors=torch.from_numpy(np.log(rgb_clipped / (1 - rgb_clipped))),
        )
        init_xyz = mesh_xyz
        init_rgb = (np.clip(mesh_rgb, 0, 1) * 255).astype(np.uint8)
        print(f"[refit] mesh-based training views — {N_mesh:,} surface pts")
    else:
        iso_log = min(float(torch.quantile(asset.log_scales.flatten(), 0.50).item()), math.log(0.006))
        render_scene = GaussianScene(
            means=asset.means,
            quats=asset.quats,
            log_scales=torch.full_like(asset.log_scales, iso_log),
            logit_opacities=asset.logit_opacities,
            raw_colors=asset.raw_colors,
        )
        init_xyz = asset.means.numpy()
        init_rgb = (asset.colors().numpy() * 255).clip(0, 255).astype(np.uint8)
        print(f"[refit] isotropic fallback  scale={math.exp(iso_log)*1000:.1f}mm")

    print(f"[refit] rendering {n_views} side + 32 pole orbit views ...")
    imgs, exts, Ks = render_orbit_with_cameras(render_scene, image_hw=image_hw, n_frames=n_views, device=device)
    for elev in (60.0, 75.0, -55.0, -75.0):
        imgs_e, exts_e, Ks_e = render_orbit_with_cameras(render_scene, image_hw=image_hw, n_frames=8, elevation_deg=elev, device=device)
        imgs = np.concatenate([imgs, imgs_e], axis=0)
        exts = np.concatenate([exts, exts_e], axis=0)
        Ks   = np.concatenate([Ks,   Ks_e],  axis=0)

    print(f"[refit] fitting {len(init_xyz):,} Gaussians against {len(imgs)} views ...")
    result = fit_gaussians(
        xyz=init_xyz, rgb=init_rgb, images=imgs, intrinsics=Ks, extrinsics=exts,
        n_iters=n_iters, init_scale=init_scale,
        scale_reg=0.05, scale_cap=init_scale * 5,
        device=device,
    )
    print(f"[refit] done — {result.n:,} Gaussians")
    return result


def render_from_cameras(
    scene: GaussianScene,
    intrinsics: np.ndarray,   # (V, 3, 3)
    extrinsics: np.ndarray,   # (V, 3, 4) world-to-cam [R|t]
    image_hw: tuple[int, int],
    device: str = "cuda",
) -> list[np.ndarray]:
    """Render scene from the actual training camera positions.

    Unlike render_orbit, this only uses viewpoints where the scene was observed,
    so it avoids the degenerate streaks that appear from unseen angles in
    sparse (2–4 view) scenes.
    """
    from gsplat import rasterization

    H, W = image_hw
    means_t = scene.means.to(device)
    quats_t = scene.quats.to(device)
    scales_t = torch.exp(scene.log_scales).to(device)
    opacities_t = torch.sigmoid(scene.logit_opacities).to(device)
    colors_t = torch.sigmoid(scene.raw_colors).to(device)

    frames: list[np.ndarray] = []
    for i in range(len(extrinsics)):
        # Pad 3×4 → 4×4
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :] = extrinsics[i]
        vm_t = torch.from_numpy(vm).unsqueeze(0).to(device)
        K_t = torch.from_numpy(intrinsics[i].astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=vm_t, Ks=K_t, width=W, height=H, packed=False,
                backgrounds=torch.ones(1, 3, device=device),
            )
        frames.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    return frames


def render_around_asset(
    scene: GaussianScene,
    asset_center: np.ndarray,   # (3,) world-space center of the asset
    asset_size: float,          # bounding box diameter (meters / scene units)
    ref_intrinsic: np.ndarray,  # (3, 3) from one training camera
    image_hw: tuple[int, int],
    world_up: np.ndarray,
    n_frames: int = 8,
    elevation_deg: float = 20.0,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Orbit closely around the placed asset so its quality is clearly visible.

    Uses a radius of 3× the asset size and adjusts focal length so the asset
    fills roughly half the frame, regardless of scene scale.
    """
    from gsplat import rasterization

    H, W = image_hw
    radius = asset_size * 3.0

    # Scale focal length so asset_size subtends ~half the frame width
    # f = (W/2) / tan(half_fov);  half_fov = atan(asset_size/2 / radius)
    import math as _math
    half_angle = _math.atan((asset_size / 2.0) / radius)
    f_asset = (W / 2.0) / _math.tan(half_angle)
    cx, cy = W / 2.0, H / 2.0
    K_asset = np.array([[f_asset, 0, cx], [0, f_asset, cy], [0, 0, 1]], dtype=np.float32)

    means_t = scene.means.to(device)
    quats_t = scene.quats.to(device)
    scales_t = torch.exp(scene.log_scales).to(device)
    opacities_t = torch.sigmoid(scene.logit_opacities).to(device)
    colors_t = torch.sigmoid(scene.raw_colors).to(device)
    K_t = torch.from_numpy(K_asset).unsqueeze(0).to(device)

    el = _math.radians(elevation_deg)
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        az = 2 * _math.pi * i / n_frames

        cam_pos = asset_center + np.array([
            radius * _math.cos(el) * _math.cos(az),
            radius * _math.cos(el) * _math.sin(az),
            radius * _math.sin(el),
        ], dtype=np.float32)

        fwd = asset_center - cam_pos
        fwd /= np.linalg.norm(fwd)

        up = world_up - np.dot(world_up, fwd) * fwd
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            up = np.array([0., 1., 0.], dtype=np.float32)
            up -= np.dot(up, fwd) * fwd
            up /= np.linalg.norm(up)
        else:
            up /= up_norm

        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        R = np.stack([right, -up, fwd], axis=0)
        t = -R @ cam_pos
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = R; vm[:3, 3] = t
        vm_t = torch.from_numpy(vm).unsqueeze(0).to(device)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=means_t, quats=quats_t, scales=scales_t,
                opacities=opacities_t, colors=colors_t,
                viewmats=vm_t, Ks=K_t, width=W, height=H, packed=False,
                backgrounds=torch.ones(1, 3, device=device),
            )
        frames.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    return frames


# ---------------------------------------------------------------------------
# PLY import/export (3DGS / SuperSplat format)
# ---------------------------------------------------------------------------

def read_splat_ply(data: bytes) -> GaussianScene:
    """
    Deserialize a 3DGS PLY written by write_splat_ply back into a GaussianScene.

    Parses the 17-float-per-vertex binary format directly; no external deps.
    Reverses the SH-DC ↔ logit-color transform applied by write_splat_ply.
    """
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
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    arr = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)

    means = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
    # Reverse: f_dc = (colors_01 - 0.5) / SH_C0  →  colors_01 = f_dc * SH_C0 + 0.5
    colors_01 = np.clip(f_dc * SH_C0 + 0.5, 1e-3, 1 - 1e-3)
    raw_colors = np.log(colors_01 / (1 - colors_01)).astype(np.float32)
    log_scales = np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(np.float32)

    return GaussianScene(
        means=torch.from_numpy(means),
        quats=torch.from_numpy(quats),
        log_scales=torch.from_numpy(log_scales),
        logit_opacities=torch.from_numpy(arr["opacity"].astype(np.float32)),
        raw_colors=torch.from_numpy(raw_colors),
    )


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
