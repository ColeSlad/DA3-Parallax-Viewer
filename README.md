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
         "ply_url":     "presigned-r2-url (1hr expiry)",
         "point_count": int,
         "view_count":  int,
         "duration_ms": int
       } | null,
       "error": "string" | null
     }
  -> 404 if no such job
```

### End-to-end curl test

```bash
# 1. Upload images and capture job_id
JOB=$(curl -sf -X POST https://<your-url>/api/reconstructions \
  -F "images[]=@images/001.jpg" \
  -F "images[]=@images/002.jpg" \
  -F "images[]=@images/003.jpg")
echo $JOB
JOB_ID=$(echo $JOB | jq -r '.job_id')

# 2. Poll until done (takes ~30-90s including GPU cold start)
curl -sf https://<your-url>/api/reconstructions/$JOB_ID | jq .

# 3. Once status=succeeded, download the .ply
PLY_URL=$(curl -sf https://<your-url>/api/reconstructions/$JOB_ID | jq -r '.result.ply_url')
curl -o output/result.ply "$PLY_URL"

# Or run the full test in one shot:
./scripts/test_e2e.sh https://<your-url> ./images
```
