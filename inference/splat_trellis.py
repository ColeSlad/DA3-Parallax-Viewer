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

    Scale: softplus(_scaling) + 9e-4  (TRELLIS uses softplus, not exp)
    Color: SH_C0 * f_dc + 0.5 → logit
    """
    means          = gs._xyz.detach().cpu().float()
    quats          = F.normalize(gs._rotation.detach(), dim=-1).cpu().float()
    log_scales     = torch.log((F.softplus(gs._scaling.detach().cpu().float()) + 9e-4).clamp(min=1e-8))
    logit_opacities = gs._opacity.detach().cpu().float().squeeze(-1)
    color_01       = (SH_C0 * gs._features_dc.detach().cpu().float()[:, 0, :] + 0.5).clamp(1e-3, 1 - 1e-3)
    raw_colors     = torch.log(color_01 / (1 - color_01))
    opacity        = torch.sigmoid(logit_opacities)

    # Build all filter masks on the raw arrays, then apply once.
    not_white = ~(color_01 > 0.82).all(dim=-1)

    hi_op  = opacity > 0.3
    core   = means[hi_op] if hi_op.sum() > 100 else means
    extent = float((core.max(dim=0).values - core.min(dim=0).values).max())
    in_core = (means - core.mean(dim=0)).norm(dim=-1) < extent * 3.0

    keep = not_white & in_core & (opacity > 0.02)
    fields = [means, quats, log_scales, logit_opacities, raw_colors]
    means, quats, log_scales, logit_opacities, raw_colors = [f[keep] for f in fields]

    # Random subsample — preserves TRELLIS's density distribution (top-opacity
    # subsampling discards interior fill Gaussians, producing a hollow appearance).
    N = means.shape[0]
    if N > 100_000:
        idx = torch.randperm(N)[:100_000]
        means, quats, log_scales, logit_opacities, raw_colors = [f[idx] for f in [means, quats, log_scales, logit_opacities, raw_colors]]

    print(f"[trellis] {gs._xyz.shape[0]:,} raw → {keep.sum():,} after filters → {means.shape[0]:,} final")

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
# Mesh surface sampler (feeds refit training views)
# ---------------------------------------------------------------------------

def _sample_mesh_points(
    mesh,
    gs_means: np.ndarray,
    gs_colors: np.ndarray,
    n_samples: int = 30_000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Sample surface points from a TRELLIS mesh and assign per-point colors.
    Returns (xyz, rgb) float32 arrays, or (None, None) on failure.
    """
    try:
        import trimesh as _trimesh  # noqa: PLC0415
        from scipy.spatial import cKDTree  # noqa: PLC0415

        verts = getattr(mesh, 'vertices', None) or getattr(mesh, 'verts', None)
        faces = getattr(mesh, 'faces', None)
        if verts is None or faces is None:
            return None, None

        to_np = lambda t: t.detach().cpu().float().numpy() if hasattr(t, 'detach') else np.asarray(t, dtype=np.float32)
        verts_np = to_np(verts)
        faces_np = np.asarray(faces.detach().cpu().numpy() if hasattr(faces, 'detach') else faces, dtype=np.int32)

        tm = _trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        pts, face_idx = _trimesh.sample.sample_surface(tm, n_samples)
        pts = pts.astype(np.float32)

        attrs = getattr(mesh, 'vertex_attrs', None) or {}
        vertex_colors = None
        if 'rgb' in attrs:
            vertex_colors = to_np(attrs['rgb']).clip(0, 1)
        elif 'shs' in attrs:
            vertex_colors = (to_np(attrs['shs'])[:, :3] * SH_C0 + 0.5).clip(0, 1)

        if vertex_colors is not None:
            bary = _trimesh.triangles.points_to_barycentric(
                triangles=verts_np[faces_np[face_idx]], points=pts,
            ).astype(np.float32)
            colors = np.einsum('ni,nij->nj', bary, vertex_colors[faces_np[face_idx]]).clip(0, 1).astype(np.float32)
        else:
            _, knn_idx = cKDTree(gs_means).query(pts, k=1)
            colors = gs_colors[knn_idx].astype(np.float32)

        print(f"[mesh_sample] {n_samples:,} surface pts sampled")
        return pts, colors

    except Exception as exc:
        print(f"[mesh_sample] failed ({exc})")
        return None, None


# ---------------------------------------------------------------------------
# Main generation function (runs inside Modal TRELLIS container)
# ---------------------------------------------------------------------------

def generate_asset_gaussians(
    prompt: str,
    image_bytes: bytes | None = None,
    seed: int = 42,
    weights_dir: str = "/trellis-weights",
) -> tuple["GaussianScene", bytes, np.ndarray | None, np.ndarray | None]:
    """
    Generate 3D Gaussians from a text prompt or image using TRELLIS.

    Text path:  FLUX generates an image → TRELLIS converts to 3D.
    Image path: TRELLIS converts the provided image directly.

    Returns (GaussianScene, input_image_png_bytes, mesh_xyz, mesh_rgb).
    mesh_xyz/mesh_rgb are 30k surface points from the TRELLIS mesh decoder,
    used by refit_asset_gaussians for sharp training views.
    Call splat_insert.place_asset() to bring into scene frame.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="xformers")

    if image_bytes is not None:
        print(f"[trellis] image-to-3D  ({len(image_bytes) // 1024} KB input)")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_image_bytes = image_bytes
    else:
        print(f"[trellis] text-to-3D  prompt={prompt!r}")
        image = _text_to_image(prompt, cache_dir=weights_dir)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        input_image_bytes = buf.getvalue()

    from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415

    print("[trellis] loading TRELLIS pipeline ...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    print("[trellis] pipeline ready")

    outputs = pipeline.run(
        image,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=True,
        sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 25, "cfg_strength": 3.0},
    )
    gs = outputs["gaussian"][0]
    print(f"[trellis] generated {gs._xyz.shape[0]:,} gaussians")

    result = _extract_trellis_gaussians(gs)

    mesh_xyz, mesh_rgb = None, None
    if outputs.get("mesh"):
        gs_colors = torch.sigmoid(result.raw_colors).numpy()
        mesh_xyz, mesh_rgb = _sample_mesh_points(
            outputs["mesh"][0],
            gs_means=result.means.numpy(),
            gs_colors=gs_colors,
        )

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
