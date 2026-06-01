#!/usr/bin/env bash
# End-to-end API smoke test.
#
# Usage:
#   ./scripts/test_e2e.sh <api_url> <image_dir>
#
# Examples:
#   # With modal serve (copy the URL it prints):
#   ./scripts/test_e2e.sh https://coleslad--da3-parallax-web-app-dev.modal.run ./images
#
#   # With deployed endpoint:
#   ./scripts/test_e2e.sh https://coleslad--da3-parallax-web-app.modal.run ./images
#
# Requires: curl, jq

set -euo pipefail

API_URL="${1:?Usage: $0 <api_url> <image_dir>}"
IMAGE_DIR="${2:?Usage: $0 <api_url> <image_dir>}"

if ! command -v jq &>/dev/null; then
  echo "jq is required: brew install jq" >&2
  exit 1
fi

# Collect image files
mapfile -t IMAGE_FILES < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort)

if [[ ${#IMAGE_FILES[@]} -eq 0 ]]; then
  echo "No images found in $IMAGE_DIR" >&2
  exit 1
fi

echo "=== Uploading ${#IMAGE_FILES[@]} images to $API_URL ==="

# Build -F args
FORM_ARGS=()
for f in "${IMAGE_FILES[@]}"; do
  FORM_ARGS+=(-F "images[]=@${f}")
done

response=$(curl -sf -X POST "$API_URL/api/reconstructions" "${FORM_ARGS[@]}")
echo "POST response: $response"

job_id=$(echo "$response" | jq -r '.job_id')
echo "Job ID: $job_id"

# Poll until terminal state
echo ""
echo "=== Polling for completion ==="
while true; do
  status_resp=$(curl -sf "$API_URL/api/reconstructions/$job_id")
  status=$(echo "$status_resp" | jq -r '.status')
  echo "  status: $status"

  if [[ "$status" == "succeeded" ]]; then
    echo ""
    echo "=== Job succeeded ==="
    echo "$status_resp" | jq '.result'
    ply_url=$(echo "$status_resp" | jq -r '.result.ply_url')
    break
  elif [[ "$status" == "failed" ]]; then
    echo ""
    echo "Job failed:" >&2
    echo "$status_resp" | jq -r '.error' >&2
    exit 1
  fi

  sleep 5
done

# Download .ply
mkdir -p output
out_file="output/e2e_result.ply"
echo "Downloading .ply to $out_file ..."
curl -sf -o "$out_file" "$ply_url"
size=$(wc -c < "$out_file")
echo "Downloaded $size bytes → $out_file"
echo ""
echo "Open in MeshLab: open $out_file"
