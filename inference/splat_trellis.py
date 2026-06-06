"""
Asset generation via TRELLIS image-to-3D Gaussian.

Text path:  prompt → SDXL-Turbo image → TRELLIS → GaussianScene
Image path: image_bytes → TRELLIS → GaussianScene

TRELLIS canonical frame (output convention):
  Up axis   : +Z  (render_utils uses [0,0,1] as camera up; GLB export applies Z→Y)
  Handedness: right-handed
  Scale     : positions in AABB [-0.5, -0.5, -0.5] → [1.0, 1.0, 1.0] (extent 1.5)
  Quaternions: [w, x, y, z]  (3DGS convention, same as our GaussianScene)
  _scaling  : PRE-SOFTPLUS space  (actual_scale = softplus(_scaling) + 9e-4, NOT exp)
  _opacity  : logit space (same as our logit_opacities)

The returned GaussianScene is in this canonical frame.
Call splat_insert.place_asset() to bring it into the scene world frame.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inference.splat_fit import GaussianScene

SH_C0 = 0.28209479177387814  # zeroth-order SH coefficient: 1 / (2*sqrt(pi))


# ---------------------------------------------------------------------------
# TRELLIS → GaussianScene
# ---------------------------------------------------------------------------

def _extract_trellis_gaussians(gs) -> GaussianScene:
    """
    Convert a TRELLIS Gaussian object to GaussianScene.

    TRELLIS Gaussian representation differences from standard 3DGS:
      _rotation : unnormalized [w,x,y,z] quaternion  (same as 3DGS)
      _scaling  : PRE-SOFTPLUS values  (3DGS uses log-space / exp; TRELLIS uses softplus)
      _opacity  : logit-space opacity, shape (N, 1)  (same as 3DGS)
      _features_dc: SH DC coefficients, shape (N, 1, 3)  (sh_degree=0, no _features_rest)

    Scale conversion: actual_scale = softplus(_scaling) + 9e-4 (minimum_kernel_size)
    Color conversion: linear_rgb = SH_C0 * f_dc + 0.5  →  logit → raw_colors
    """
    means = gs._xyz.detach().cpu().float()                             # (N, 3)
    quats = F.normalize(gs._rotation.detach(), dim=-1).cpu().float()  # (N, 4) [w,x,y,z]
    # TRELLIS uses softplus (not exp) for scaling activation; convert to log-space
    # so GaussianScene / gsplat can use exp(log_scales) correctly.
    scaling_raw = gs._scaling.detach().cpu().float()                   # (N, 3) pre-softplus
    actual_scales = F.softplus(scaling_raw) + 9e-4                    # match TRELLIS get_scaling
    print(
        f"[trellis] raw _scaling: min={scaling_raw.min():.3f}  max={scaling_raw.max():.3f}  "
        f"mean={scaling_raw.mean():.3f}  → actual_scale p50={actual_scales.median():.4f}  "
        f"p90={actual_scales.quantile(0.90).item():.4f}  max={actual_scales.max():.4f} (canonical units)"
    )
    log_scales = torch.log(actual_scales.clamp(min=1e-8))             # (N, 3) log-space
    logit_opacities = gs._opacity.detach().cpu().float().squeeze(-1)  # (N,)

    f_dc = gs._features_dc.detach().cpu().float()[:, 0, :]            # (N, 3)
    color_01 = (SH_C0 * f_dc + 0.5).clamp(1e-3, 1 - 1e-3)
    raw_colors = torch.log(color_01 / (1 - color_01))                 # (N, 3) logit

    opacity = torch.sigmoid(logit_opacities)        # (N,) in [0, 1]

    # Step 0: remove near-pure-white Gaussians.
    # TRELLIS's preprocess_image=True removes the background but leaves edge Gaussians
    # that picked up the white background colour. These contaminate the object colour.
    # Threshold: all three channels > 0.82 → near-white regardless of hue.
    not_white = ~(color_01 > 0.82).all(dim=-1)
    n_white = int((~not_white).sum())
    if n_white:
        means = means[not_white]; quats = quats[not_white]
        log_scales = log_scales[not_white]; logit_opacities = logit_opacities[not_white]
        raw_colors = raw_colors[not_white]; color_01 = color_01[not_white]
        opacity = opacity[not_white]
        print(f"[trellis] white filter: removed {n_white:,} near-white Gaussians")

    # Step 1: centroid-based floater rejection.
    # Use a generous 3× multiplier so extended features (wings, tails, antennae) are
    # kept; only truly distant background grid / streak artifacts are cut.
    hi_op_mask = opacity > 0.3
    core_means = means[hi_op_mask] if hi_op_mask.sum() > 100 else means
    core_centroid = core_means.mean(dim=0)
    core_extent = float((core_means.max(dim=0).values - core_means.min(dim=0).values).max())
    dist = (means - core_centroid).norm(dim=-1)
    in_core = dist < core_extent * 3.0
    print(
        f"[trellis] floater filter: core_extent={core_extent:.4f}  "
        f"removed {(~in_core).sum():,} of {means.shape[0]:,} Gaussians outside core"
    )

    means = means[in_core]; quats = quats[in_core]
    log_scales = log_scales[in_core]; logit_opacities = logit_opacities[in_core]
    raw_colors = raw_colors[in_core]; opacity = opacity[in_core]

    # Step 2: prune near-invisible Gaussians only.
    # Do NOT cap by scale — large Gaussians are interior fill and removing them
    # creates a hollow shell. The spatial filter above handles distant floaters.
    keep = opacity > 0.02

    means = means[keep]; quats = quats[keep]
    log_scales = log_scales[keep]; logit_opacities = logit_opacities[keep]
    raw_colors = raw_colors[keep]; opacity = opacity[keep]

    # Step 3: subsample to at most MAX_GAUSSIANS using random sampling.
    # Top-opacity subsampling preferentially keeps surface splats and discards
    # interior fill Gaussians, producing a hollow appearance. Random sampling
    # preserves the full density distribution TRELLIS learned.
    MAX_GAUSSIANS = 100_000
    n_after_filter = means.shape[0]
    if n_after_filter > MAX_GAUSSIANS:
        perm = torch.randperm(n_after_filter)[:MAX_GAUSSIANS]
        means = means[perm]; quats = quats[perm]
        log_scales = log_scales[perm]; logit_opacities = logit_opacities[perm]
        raw_colors = raw_colors[perm]

    n_final = means.shape[0]
    print(
        f"[trellis] {gs._xyz.shape[0]:,} raw  "
        f"→ {n_after_filter:,} after opacity filter  "
        f"→ {n_final:,} after subsample"
    )

    return GaussianScene(
        means=means,
        quats=quats,
        log_scales=log_scales,
        logit_opacities=logit_opacities,
        raw_colors=raw_colors,
    )


# ---------------------------------------------------------------------------
# Serialization helpers (no TRELLIS dependency — safe to import anywhere)
# ---------------------------------------------------------------------------

def gaussianscene_to_dict(gs: GaussianScene) -> dict:
    """Serialize GaussianScene to a dict of numpy arrays for Modal transfer."""
    return {
        "means": gs.means.numpy(),
        "quats": gs.quats.numpy(),
        "log_scales": gs.log_scales.numpy(),
        "logit_opacities": gs.logit_opacities.numpy(),
        "raw_colors": gs.raw_colors.numpy(),
    }


def dict_to_gaussianscene(d: dict) -> GaussianScene:
    """Reconstruct GaussianScene from serialized dict."""
    return GaussianScene(
        means=torch.from_numpy(d["means"]),
        quats=torch.from_numpy(d["quats"]),
        log_scales=torch.from_numpy(d["log_scales"]),
        logit_opacities=torch.from_numpy(d["logit_opacities"]),
        raw_colors=torch.from_numpy(d["raw_colors"]),
    )


# ---------------------------------------------------------------------------
# Main generation function (runs inside Modal TRELLIS container)
# ---------------------------------------------------------------------------

def _sample_mesh_points(
    mesh,
    gs_means: np.ndarray,
    gs_colors: np.ndarray,
    n_samples: int = 30_000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Sample surface points from a TRELLIS mesh and assign per-point colors.

    Colors are sourced in priority order:
      1. vertex_attrs['rgb']  — direct per-vertex RGB from TRELLIS decoder
      2. vertex_attrs['shs']  — SH DC term converted to RGB
      3. KNN from TRELLIS Gaussians — always available fallback

    Returns (xyz, rgb) float32 arrays or (None, None) on any failure.
    """
    try:
        import trimesh as _trimesh  # noqa: PLC0415
        from scipy.spatial import cKDTree  # noqa: PLC0415

        verts = getattr(mesh, 'vertices', None) or getattr(mesh, 'verts', None)
        faces = getattr(mesh, 'faces', None)
        if verts is None or faces is None:
            print("[mesh_sample] no vertices/faces on mesh — skipping")
            return None, None

        to_np = lambda t: t.detach().cpu().float().numpy() if hasattr(t, 'detach') else np.asarray(t, dtype=np.float32)
        verts_np = to_np(verts)
        faces_np = np.asarray(faces.detach().cpu().numpy() if hasattr(faces, 'detach') else faces, dtype=np.int32)

        tm = _trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        pts, face_idx = _trimesh.sample.sample_surface(tm, n_samples)
        pts = pts.astype(np.float32)

        # Attempt to get per-vertex colors
        vertex_colors = None
        color_src = None
        attrs = getattr(mesh, 'vertex_attrs', None) or {}
        if 'rgb' in attrs:
            vertex_colors = to_np(attrs['rgb']).clip(0, 1)
            color_src = "vertex_attrs['rgb']"
        elif 'shs' in attrs:
            SH_C0 = 0.28209479177387814
            vertex_colors = (to_np(attrs['shs'])[:, :3] * SH_C0 + 0.5).clip(0, 1)
            color_src = "vertex_attrs['shs'] DC"

        if vertex_colors is not None:
            bary = _trimesh.triangles.points_to_barycentric(
                triangles=verts_np[faces_np[face_idx]],
                points=pts,
            ).astype(np.float32)
            face_vcolors = vertex_colors[faces_np[face_idx]]  # (N, 3, 3)
            colors = np.einsum('ni,nij->nj', bary, face_vcolors).clip(0, 1).astype(np.float32)
        else:
            _, knn_idx = cKDTree(gs_means).query(pts, k=1)
            colors = gs_colors[knn_idx].astype(np.float32)
            color_src = "KNN from TRELLIS Gaussians"

        print(f"[mesh_sample] {n_samples:,} surface pts  colors={color_src}")
        return pts, colors

    except Exception as exc:
        print(f"[mesh_sample] failed ({exc}) — will use blurry Gaussian training views")
        return None, None


def generate_asset_gaussians(
    prompt: str,
    image_bytes: bytes | None = None,
    seed: int = 42,
    weights_dir: str = "/trellis-weights",
) -> tuple["GaussianScene", bytes, np.ndarray | None, np.ndarray | None]:
    """
    Generate 3D Gaussians from a text prompt or image using TRELLIS.

    Text path:  SDXL-Turbo/FLUX generates an image → TRELLIS converts to 3D.
    Image path: TRELLIS converts the provided image directly.

    Returns (GaussianScene, image_png_bytes, mesh_xyz, mesh_rgb).
    mesh_xyz/mesh_rgb are surface-sampled points from the TRELLIS mesh decoder
    (30k points, float32). Used by refit_asset_gaussians to render sharp training
    views instead of blurry Gaussian-rendered views.
    Returns None for mesh arrays if the mesh decoder fails.

    The caller must run splat_insert.place_asset() to bring into scene frame.
    Weights are cached in `weights_dir` (Modal Volume mount point).
    """
    import warnings
    # xformers uses the deprecated torch.library.impl_abstract API; suppress until xformers updates.
    warnings.filterwarnings("ignore", category=FutureWarning, module="xformers")

    if image_bytes is not None:
        print(f"[trellis] image-to-3D  ({len(image_bytes) // 1024} KB input)")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image_bytes = image_bytes  # return the caller's image unchanged
    else:
        print(f"[trellis] text-to-3D  prompt={prompt!r}")
        image = _text_to_image(prompt, cache_dir=weights_dir)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        input_image_bytes = buf.getvalue()

    # Deferred import: trellis is only installed in the trellis_image container
    from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415

    print("[trellis] loading TRELLIS pipeline ...")
    # from_pretrained only accepts path: str — no cache_dir param.
    # HF_HOME is already set above so hf_hub_download caches to our volume.
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large",
    )
    pipeline.cuda()
    print("[trellis] pipeline ready")

    outputs = pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],   # mesh gives sharp surface for refit training views
        preprocess_image=True,          # TRELLIS handles resize + background removal
        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 25, "cfg_strength": 3.0},
    )
    gs = outputs["gaussian"][0]
    n = gs._xyz.shape[0]
    print(f"[trellis] generated {n:,} gaussians")

    result = _extract_trellis_gaussians(gs)

    # Sample mesh surface for sharp refit training views
    mesh_xyz, mesh_rgb = None, None
    mesh_out = outputs.get("mesh", [])
    print(f"[trellis] mesh output: {type(mesh_out)}  len={len(mesh_out) if hasattr(mesh_out, '__len__') else 'N/A'}")
    if mesh_out:
        mesh_obj = mesh_out[0]
        print(f"[trellis] mesh[0] type={type(mesh_obj).__name__}  attrs={[a for a in dir(mesh_obj) if not a.startswith('_')][:15]}")
        gs_colors = torch.sigmoid(result.raw_colors).numpy()
        mesh_xyz, mesh_rgb = _sample_mesh_points(
            mesh_obj,
            gs_means=result.means.numpy(),
            gs_colors=gs_colors,
        )

    # Free GPU memory before returning
    del pipeline, gs, outputs
    torch.cuda.empty_cache()

    return result, input_image_bytes, mesh_xyz, mesh_rgb


# ---------------------------------------------------------------------------
# Text-to-image via SDXL-Turbo
# ---------------------------------------------------------------------------

def _text_to_image(prompt: str, cache_dir: str) -> Image.Image:
    """
    Generate a 1024x1024 RGB image from a text prompt using FLUX.1-schnell.
    FLUX.1-schnell is a distilled flow-matching model: 4 steps, no CFG, bfloat16.
    Substantially higher quality than SDXL-Turbo at the same step count.
    Frees GPU memory after generation so TRELLIS can load cleanly.
    """
    from diffusers import FluxPipeline  # noqa: PLC0415

    print(f"[trellis] generating image for prompt: {prompt!r}")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    pipe = pipe.to("cuda")

    # Wrap the prompt to bias toward a clean single-object render; TRELLIS
    # background-removes the result, so a neutral-bg product-photo style helps.
    # FLUX follows natural language well so a descriptive style works better than
    # comma-separated tags.
    wrapped = (
        f"{prompt}, single object on a pure white background, "
        "product photography, studio lighting, centered, no shadows"
    )
    with torch.inference_mode():
        result = pipe(
            prompt=wrapped,
            num_inference_steps=4,   # schnell: distilled, 4 steps sufficient
            guidance_scale=0.0,      # distilled model; CFG not needed
            width=1024,
            height=1024,
        )

    image = result.images[0]
    del pipe, result
    torch.cuda.empty_cache()
    print("[trellis] image generation done")
    return image
