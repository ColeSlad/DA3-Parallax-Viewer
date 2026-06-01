# DA3-Parallax

Full-stack 3D reconstruction app built on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

## Current scope — GPU inference validation

This phase proves DA3 multi-view inference produces good reconstructions and that
back-projection + .ply export work before building the surrounding app.

### Prerequisites

1. A Modal account with an active token:
   ```bash
   pip install modal
   modal token new
   ```

2. Python 3.11+ locally (only needed to drive `modal run`).

### Run the validation

```bash
modal run inference/app.py::validate
```

This will:
- Spin up an L4 GPU container on Modal with DA3 installed
- Download model weights to a persistent Modal Volume (first run only; ~5 GB)
- Run multi-view inference against the bundled DA3 example scene (`assets/examples/SOH`)
- Back-project depth maps into a colored point cloud in world space
- Apply confidence filtering and voxel downsampling
- Write `output/validation.ply` to your local machine

Open `output/validation.ply` in [MeshLab](https://www.meshlab.net/) to inspect the result.

### Tuning quality

Two parameters control output quality:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--conf-percentile` | `25` | Keep only points above this confidence percentile. Higher = fewer points, less noise. |
| `--voxel-size` | `0.02` | Voxel grid leaf size in meters. Larger = more aggressive downsampling. |

```bash
modal run inference/app.py::validate --conf-percentile 40 --voxel-size 0.01
```

### Printed metrics

```
Views:           8
Raw points:      4_823_112
After conf filter: 2_104_788  (conf > p25 = 0.412)
After voxel ds:    312_440
Duration:        14.3 s
```

## Planned architecture

- React + Vite + TypeScript frontend, @react-three/fiber point-cloud viewer
- FastAPI backend as Modal ASGI app
- DA3 inference as a separate Modal GPU function (L4, scale-to-zero)
- Postgres (Neon) for job state: `queued → running → succeeded | failed`
- Cloudflare R2 for uploaded images and output `.ply` files
