# DA3-Parallax

Upload photos of a scene, reconstruct a 3D point cloud, then insert a generated 3D object into it using a text prompt — all rendered in the browser as a gaussian splat.

**Pipeline:** photos → [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) multi-view depth inference → point cloud (~5 min) → text prompt → [TRELLIS](https://github.com/microsoft/TRELLIS) 3D asset generation (A100) → combined gaussian splat rendered in-browser.

---

## Architecture

| Layer | What |
|-------|------|
| Frontend | React + Vite + TypeScript; React Three Fiber point cloud viewer; WebGL gaussian splat viewer (`@mkkellogg/gaussian-splats-3d`) |
| API | FastAPI served as a Modal ASGI app; async job queue backed by PostgreSQL (Neon) |
| Reconstruction worker | DA3-LARGE-1.1 on Modal L4; outputs point cloud PLY uploaded to Cloudflare R2 |
| Insertion worker | TRELLIS + gsplat on Modal L4/A100; converts point cloud to isotropic gaussians, generates and refits a 3D asset, merges and uploads combined PLY |
| Storage | Cloudflare R2 for input images and output PLY files; presigned URLs delivered to frontend |

### User flow

1. Upload photos in the browser → reconstruction job queued
2. DA3 runs on L4 GPU (~5 min); point cloud appears in the interactive viewer
3. User enters a text prompt, world-space position, and size → insertion job queued
4. TRELLIS generates a 3D asset on A100 (parallelised with scene prep); combined splat uploaded
5. Browser switches from point cloud viewer to gaussian splat viewer showing the merged result

---

## Setup

### Prerequisites

```bash
pip install modal
modal token new
```

### External resources

**Neon (Postgres)**
1. Create a project at [neon.tech](https://neon.tech)
2. Copy the pooler connection string from *Connection Details*

**Cloudflare R2**
1. Create a bucket
2. *R2 → Manage API Tokens* → token with **Object Read & Write** on that bucket
3. Note your Account ID, Access Key ID, Secret Access Key

### Modal secret

Create a secret named **`da3-parallax-secrets`** in your Modal dashboard:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | `postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require` |
| `R2_ENDPOINT` | `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | from R2 API token |
| `R2_SECRET_ACCESS_KEY` | from R2 API token |
| `R2_BUCKET` | your bucket name |
| `CORS_ORIGIN` | `http://localhost:5173` (or your frontend URL) |

You also need a **`huggingface-token`** secret with `HF_TOKEN` for DA3 model weights.

### Apply DB migrations

```bash
export DATABASE_URL=postgresql://...
pip install psycopg2-binary
python scripts/migrate.py
```

### Run locally

```bash
# Backend (hot reload)
modal serve api/main.py

# Frontend
cd web && npm install && npm run dev
```

### Deploy

```bash
modal deploy api/main.py
```

---

## API

```
POST /api/reconstructions
  multipart: images[]  (jpeg/png/webp, max 30 files, 20 MB each, 200 MB total)
  -> 202 { "job_id": "uuid", "status": "queued" }

GET  /api/reconstructions/{job_id}
  -> 200 {
       "status": "queued" | "running" | "succeeded" | "failed",
       "result": {
         "pointcloud_url": "presigned-r2-url (1hr)",
         "point_count": int,
         "view_count":  int,
         "duration_ms": int
       } | null
     }

POST /api/scenes/{scene_job_id}/insertions
  JSON: { "prompt": str, "position": [x,y,z], "size_m": float }
  -> 202 { "job_id": "uuid", "status": "queued" }
  -> 409 if scene job not yet succeeded

GET  /api/insertions/{job_id}
  -> 200 {
       "status": "queued" | "running" | "succeeded" | "failed",
       "result": {
         "combined_splat_url": "presigned-r2-url (1hr)",
         "duration_ms": int
       } | null
     }
```

### curl validation

```bash
BASE=https://<your-modal-url>

# 1. Upload photos and start reconstruction (~5 min)
JOB=$(curl -sf -X POST $BASE/api/reconstructions \
  -F "images=@images/001.jpg" \
  -F "images=@images/002.jpg")
SCENE_ID=$(echo $JOB | jq -r '.job_id')

# 2. Poll until succeeded
curl -sf $BASE/api/reconstructions/$SCENE_ID | jq .

# 3. Insert a generated asset (~10-15 min; position in DA3 world frame)
INS=$(curl -sf -X POST $BASE/api/scenes/$SCENE_ID/insertions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a small potted plant","position":[0.1,-0.2,0.0],"size_m":0.3}')
INS_ID=$(echo $INS | jq -r '.job_id')

# 4. Poll and download
curl -sf $BASE/api/insertions/$INS_ID | jq .
COMBINED_URL=$(curl -sf $BASE/api/insertions/$INS_ID | jq -r '.result.combined_splat_url')
curl -o output/combined.ply "$COMBINED_URL"
```

Open `output/combined.ply` in [SuperSplat](https://supersplat.xyz) to inspect.

---

## Offline pipeline (no API)

Run the full pipeline locally against Modal GPU functions, saving PLYs and orbit renders:

```bash
# Point cloud only
modal run inference/app.py::validate --image-dir ./images

# TRELLIS insertion into a scene
modal run --detach inference/app.py::trellis_insert \
  --prompt "a red fire hydrant" \
  --image-dir ./images \
  --asset-x 0.1 --asset-y -0.2 --asset-size 0.4

# Download outputs after a detached run
modal run inference/app.py::download_outputs
```

Outputs are written to `output/` and also persisted to a Modal volume (`da3-outputs`) so long-running jobs survive local disconnects.
