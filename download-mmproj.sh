#!/usr/bin/env bash
# Download the Qwen3.8-Flash-Next vision projector (mmproj) so llama-server can take
# images: ./serve.sh --vision. 0.85 GiB sidecar; the UD-Q4_K_XL text weights are unchanged.
# Same aria2c strategy as download-parts.sh: resumable, size-verified, re-runnable.
set -uo pipefail
cd "$(dirname "$0")"

REPO=unsloth/Qwen3.8-Flash-Next-GGUF
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST=models/mmproj
mkdir -p "$DEST"

# BF16 rather than F16: same size, and BF16 is what the encoder was trained in.
f="mmproj-BF16.gguf"
want=907542944
path="$DEST/$f"
have=$(stat -c%s "$path" 2>/dev/null || echo 0)
if [ "$have" -eq "$want" ]; then
  echo "[skip] $f already complete ($have bytes)"
else
  echo "[get ] $f (have $have / want $want)"
  until [ "$(stat -c%s "$path" 2>/dev/null || echo 0)" -eq "$want" ]; do
    aria2c \
      --continue=true \
      --max-connection-per-server=16 --split=16 --min-split-size=1M \
      --max-tries=0 --retry-wait=5 \
      --timeout=30 --connect-timeout=30 --lowest-speed-limit=1K \
      --file-allocation=none --summary-interval=15 \
      --auto-file-renaming=false --allow-overwrite=true \
      --console-log-level=warn \
      --dir="$DEST" --out="$f" \
      "$BASE/$f" || echo "  aria2c exited $?, resuming..."
    sleep 3
  done
  echo "[done] $f complete ($(stat -c%s "$path") bytes)"
fi
echo "MMPROJ READY -> $DEST"
