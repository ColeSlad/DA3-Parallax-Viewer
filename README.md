# DA3-Parallax

Full-stack 3D reconstruction app built on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

## Architecture

- React + Vite + TypeScript frontend, @react-three/fiber point-cloud viewer *(planned)*
- FastAPI backend as Modal ASGI app (`api/main.py`)
- DA3 inference as a Modal GPU function (L4, scale-to-zero, same Modal app)
- Postgres (Neon) for job state: `queued → running → succeeded | failed`
- Cloudflare R2 for uploaded images and output `.ply` files

---

## Part 1 — GPU inference validation

Proves DA3 multi-view inference and back-projection work end-to-end.

### Prerequisites

```bash
pip install modal
modal token new
```

### Run the validation

```bash
# Against the bundled DA3 SOH example images:
modal run inference/app.py::validate

# Against your own photos:
modal run inference/app.py::validate --image-dir ./images

# Tune quality:
modal run inference/app.py::validate --image-dir ./images --conf-percentile 40 --voxel-size 0.01
```

Writes `output/validation.ply`. Open in [MeshLab](https://www.meshlab.net/).

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--conf-percentile` | `25` | Keep points above this confidence percentile. Higher = fewer, cleaner. |
| `--voxel-size` | `0.02` | Voxel grid leaf size (meters). Larger = fewer points. |

---

## Part 2 — Async job API

FastAPI service + Postgres job lifecycle + R2 storage. Drive entirely with curl.

### External resources to provision

**Neon (Postgres)**
1. Create a project at [neon.tech](https://neon.tech)
2. Copy the connection string from *Connection Details → Pooler*:
   `postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require`

**Cloudflare R2**
1. Create a bucket
2. *R2 → Manage API Tokens* → create token with **Object Read & Write** on that bucket
3. Note your Account ID, Access Key ID, Secret Access Key

### Modal secret

Create one secret named **`da3-parallax-secrets`** in your Modal dashboard with these keys:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | `postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require` |
| `R2_ENDPOINT` | `https://<account_id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | from R2 API token |
| `R2_SECRET_ACCESS_KEY` | from R2 API token |
| `R2_BUCKET` | your bucket name |
| `CORS_ORIGIN` | `http://localhost:5173` (or your frontend URL) |

### Apply the DB migration

```bash
export DATABASE_URL=postgresql://...   # same string as above
pip install psycopg2-binary
python scripts/migrate.py
```

### Serve locally (hot reload)

```bash
modal serve api/main.py
```

Modal prints a URL like `https://coleslad--da3-parallax-web-app-dev.modal.run`.

### Deploy to production

```bash
modal deploy api/main.py
```

### Apply the DB migration (run once after deploying this feature)

```bash
export DATABASE_URL=postgresql://...
python scripts/migrate.py
```

### API contract

```
POST /api/reconstructions
  multipart: images[]  (jpeg/png/webp, max 30 files, 20 MB each, 200 MB total)
  -> 202 { "job_id": "uuid", "status": "queued" }

GET  /api/reconstructions/{job_id}
  -> 200 {
       "job_id": "uuid",
       "status": "queued" | "running" | "succeeded" | "failed",
       "created_at": "iso8601",
       "result": {
         "pointcloud_url": "presigned-r2-url (1hr)",
         "splat_url":      "presigned-r2-url (1hr)",
         "point_count": int,
         "view_count":  int,
         "duration_ms": int
       } | null,
       "error": "string" | null
     }
  -> 404 if no such job

POST /api/scenes/{scene_job_id}/insertions
  JSON: { "prompt": str, "position": [x,y,z], "size_m": float, "orientation": str|null }
  -> 202 { "job_id": "uuid", "status": "queued" }
  -> 404 if scene job not found
  -> 409 if scene job not yet succeeded

GET  /api/insertions/{job_id}
  -> 200 {
       "job_id": "uuid",
       "parent_id": "uuid",
       "status": "queued" | "running" | "succeeded" | "failed",
       "created_at": "iso8601",
       "result": {
         "combined_splat_url": "presigned-r2-url (1hr)",
         "duration_ms": int
       } | null,
       "error": "string" | null
     }
  -> 404 if no such insertion job
```

### End-to-end curl validation

#### Part A — Reconstruct a scene (point cloud + gsplat)

```bash
BASE=https://<your-url>

# 1. Upload photos
JOB=$(curl -sf -X POST $BASE/api/reconstructions \
  -F "images=@images/001.jpg" \
  -F "images=@images/002.jpg" \
  -F "images=@images/003.jpg")
echo $JOB
SCENE_ID=$(echo $JOB | jq -r '.job_id')

# 2. Poll until succeeded (~5 min DA3 + ~20-30 min gsplat fit; GPU cold start adds ~2 min)
curl -sf $BASE/api/reconstructions/$SCENE_ID | jq .

# 3. Download both outputs once succeeded
SPLAT_URL=$(curl -sf $BASE/api/reconstructions/$SCENE_ID | jq -r '.result.splat_url')
PC_URL=$(curl -sf $BASE/api/reconstructions/$SCENE_ID | jq -r '.result.pointcloud_url')

mkdir -p output
curl -o output/scene.ply "$SPLAT_URL"
curl -o output/pointcloud.ply "$PC_URL"
```

#### Part B — Insert a generated asset into the scene

```bash
# Use a world-space position from the reconstructed scene (e.g. scene centroid visible
# in a point cloud viewer, or from the DA3 extrinsics).  Example: x=0.1 y=-0.2 z=0.0

# 1. Create insertion job
INS=$(curl -sf -X POST $BASE/api/scenes/$SCENE_ID/insertions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a small potted plant","position":[0.1,-0.2,0.0],"size_m":0.3}')
echo $INS
INS_ID=$(echo $INS | jq -r '.job_id')

# 2. Poll until succeeded (~10 min TRELLIS A100 + ~10 min refit/placement)
curl -sf $BASE/api/insertions/$INS_ID | jq .

# 3. Download combined splat once succeeded
COMBINED_URL=$(curl -sf $BASE/api/insertions/$INS_ID | jq -r '.result.combined_splat_url')
curl -o output/combined.ply "$COMBINED_URL"
```

Open `output/scene.ply` and `output/combined.ply` in [SuperSplat](https://supersplat.xyz) or [MeshLab](https://www.meshlab.net/) to verify the asset is present and correctly placed.
