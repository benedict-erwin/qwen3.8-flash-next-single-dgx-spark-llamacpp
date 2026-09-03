#!/usr/bin/env bash
# Fetch the Unsloth llama.cpp prebuilt -- the only build that loads the MTP draft
# head (upstream rejects it for a missing output_hc_norm.weight). No compile needed.
# Verified working on this box: GB10 / aarch64 / CUDA 13 / driver 580.173.02.
set -uo pipefail
cd "$(dirname "$0")"

TAG=b10715-mix-86bd2d3
ASSET="app-${TAG}-linux-arm64-cuda13-portable.tar.gz"
URL="https://github.com/unslothai/llama.cpp/releases/download/${TAG}/${ASSET}"
SIZE=188945299
DEST=forks/unsloth-b10715

if [ -x "$DEST/llama-server" ]; then
  echo "[skip] $DEST/llama-server already present"; exit 0
fi
mkdir -p forks
echo "[get ] $ASSET ($SIZE bytes)"
until [ "$(stat -c%s forks/$ASSET 2>/dev/null || echo 0)" -eq "$SIZE" ]; do
  aria2c --continue=true --max-connection-per-server=16 --split=16 --min-split-size=1M \
    --max-tries=5 --retry-wait=5 --file-allocation=none --console-log-level=warn \
    --auto-file-renaming=false --allow-overwrite=true \
    --dir=forks --out="$ASSET" "$URL" || echo "  aria2c exited $?, resuming..."
  sleep 3
done
mkdir -p "$DEST" && tar xzf "forks/$ASSET" -C "$DEST"
rm -f "forks/$ASSET"          # extracted tree is all that is needed
LD_LIBRARY_PATH="$PWD/$DEST" "$DEST/llama-server" --version 2>&1 | head -2
echo "FORK READY -> $DEST   (use: BUILD=fork ./serve.sh --mtp --nmax 3)"
