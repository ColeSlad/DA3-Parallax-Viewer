# DA3-Parallax — Frontend

React + Vite + TypeScript. Uploads photos, polls for job status, renders the resulting point cloud.

## Dev setup

```bash
cd web
cp .env.example .env.local
# Edit .env.local — set VITE_API_URL to your modal serve URL
npm install
npm run dev
```

Open http://localhost:5173.

The backend must be running in another terminal:
```bash
modal serve api/main.py
# Copy the printed URL into web/.env.local as VITE_API_URL
```

## R2 CORS (required before the point cloud will load)

The browser fetches the `.ply` directly from R2 via a presigned URL. R2 blocks
cross-origin requests by default — set this policy first or the viewer will silently fail.

Cloudflare dashboard → R2 → your bucket → Settings → CORS Policy:

```json
[
  {
    "AllowedOrigins": ["http://localhost:5173"],
    "AllowedMethods": ["GET"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

When you deploy the frontend, add its production origin to `AllowedOrigins`.
