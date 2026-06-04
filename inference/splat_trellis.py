"""
Asset generation via TRELLIS image-to-3D Gaussian.

Text path:  prompt → SDXL-Turbo image → TRELLIS → GaussianScene
Image path: image_bytes → TRELLIS → GaussianScene

TRELLIS canonical frame (output convention):
  Up axis   : +Y
  Handedness: right-handed
  Scale     : positions in approximately [-0.5, 0.5]^3
  Quaternions: [w, x, y, z]  (3DGS convention, same as our GaussianScene)
  _scaling  : log space (same as our log_scales)
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

    TRELLIS follows 3DGS conventions:
      _rotation : unnormalized [w,x,y,z] quaternion
      _scaling  : log-space scales
      _opacity  : logit-space opacity, shape (N, 1)
      _features_dc: SH DC coefficients, shape (N, 1, 3)

    Color conversion: linear_rgb = SH_C0 * f_dc + 0.5  →  logit → raw_colors
    (DC-only: view-independent average color, sufficient for compositing)
    """
    means = gs._xyz.detach().cpu().float()                             # (N, 3)
    quats = F.normalize(gs._rotation.detach(), dim=-1).cpu().float()  # (N, 4) [w,x,y,z]
    log_scales = gs._scaling.detach().cpu().float()                    # (N, 3)
    logit_opacities = gs._opacity.detach().cpu().float().squeeze(-1)  # (N,)

    f_dc = gs._features_dc.detach().cpu().float()[:, 0, :]            # (N, 3)
    color_01 = (SH_C0 * f_dc + 0.5).clamp(1e-3, 1 - 1e-3)
    raw_colors = torch.log(color_01 / (1 - color_01))                 # (N, 3) logit

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

def generate_asset_gaussians(
    prompt: str,
    image_bytes: bytes | None = None,
    seed: int = 42,
    weights_dir: str = "/trellis-weights",
) -> GaussianScene:
    """
    Generate 3D Gaussians from a text prompt or image using TRELLIS.

    Text path:  SDXL-Turbo generates an image → TRELLIS converts to 3D.
    Image path: TRELLIS converts the provided image directly.

    Returns GaussianScene in TRELLIS canonical frame (Y-up, ~[-0.5,0.5]^3).
    The caller must run splat_insert.place_asset() to bring into scene frame.

    Weights are cached in `weights_dir` (Modal Volume mount point).
    """
    import os
    os.environ.setdefault("HF_HOME", weights_dir)

    if image_bytes is not None:
        print(f"[trellis] image-to-3D  ({len(image_bytes) // 1024} KB input)")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        print(f"[trellis] text-to-3D  prompt={prompt!r}")
        image = _text_to_image(prompt, cache_dir=weights_dir)

    # Deferred import: trellis is only installed in the trellis_image container
    from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415

    print("[trellis] loading TRELLIS pipeline ...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large",
        cache_dir=weights_dir,
    )
    pipeline.cuda()
    print("[trellis] pipeline ready")

    outputs = pipeline.run(
        image,
        seed=seed,
        # Gaussian output only — skips the mesh decoder (no nvdiffrast needed)
        formats=["gaussian"],
        preprocess_image=True,  # TRELLIS handles resize + background removal
    )
    gs = outputs["gaussian"][0]
    n = gs._xyz.shape[0]
    print(f"[trellis] generated {n:,} gaussians")

    result = _extract_trellis_gaussians(gs)

    # Free GPU memory before returning
    del pipeline, gs, outputs
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Text-to-image via SDXL-Turbo
# ---------------------------------------------------------------------------

def _text_to_image(prompt: str, cache_dir: str) -> Image.Image:
    """
    Generate a single 512x512 RGB image from a text prompt using SDXL-Turbo.
    Frees GPU memory after generation so TRELLIS can load cleanly.
    """
    from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

    print(f"[trellis] generating image for prompt: {prompt!r}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir,
    )
    pipe = pipe.to("cuda")

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=4,  # turbo: 1–4 steps sufficient
            guidance_scale=0.0,     # distilled model; CFG not needed
            width=512,
            height=512,
        )

    image = result.images[0]
    del pipe, result
    torch.cuda.empty_cache()
    print("[trellis] image generation done")
    return image
