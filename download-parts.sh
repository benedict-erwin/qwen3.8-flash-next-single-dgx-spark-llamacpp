#!/usr/bin/env bash
# Resumable per-part download of Qwen3.8-Flash-Next UD-Q4_K_XL via aria2c.
# aria2c -x16 -s16: 16 parallel connections per file (beats HF CDN per-connection
# throttle); -c continues an existing partial file (incl. one started by curl).
# Sequential per part; each finished part is verified by exact size. Re-runnable.
set -uo pipefail
cd "$(dirname "$0")"

REPO=unsloth/Qwen3.8-Flash-Next-GGUF
BASE="https://huggingface.co/${REPO}/resolve/main/UD-Q4_K_XL"
DEST=models/UD-Q4_K_XL
mkdir -p "$DEST"

declare -A SIZE=(
  [00001]=10946624
  [00002]=49859583136
  [00003]=49376141504
  [00004]=12087983520
)

for n in 00001 00002 00003 00004; do
  f="Qwen3.8-Flash-Next-UD-Q4_K_XL-${n}-of-00004.gguf"
  path="$DEST/$f"
  want=${SIZE[$n]}
  have=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then
    echo "[skip] $f already complete ($have bytes)"
    continue
  fi
  echo "[get ] $f (have $have / want $want)"
  until [ "$(stat -c%s "$path" 2>/dev/null || echo 0)" -eq "$want" ]; do
    aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=1M \
      --max-tries=0 --retry-wait=5 \
      --timeout=30 --connect-timeout=30 \
      --lowest-speed-limit=1K \
      --file-allocation=none \
      --summary-interval=15 \
      --auto-file-renaming=false --allow-overwrite=true \
      --console-log-level=warn \
      --dir="$DEST" --out="$f" \
      "$BASE/$f" || echo "  aria2c exited $?, resuming..."
    sleep 3
  done
  echo "[done] $f complete ($(stat -c%s "$path") bytes)"
done

echo "ALL PARTS COMPLETE"
