#!/usr/bin/env bash
# Download the Qwen3.8-Flash-Next MTP draft head (sidecar for speculative decoding).
# Pairs with the existing UD-Q4_K_XL target model - no need to re-download the 104 GiB base.
# Same aria2c strategy as download-parts.sh (Decision #3): resumable, size-verified.
set -uo pipefail
cd "$(dirname "$0")"

REPO=unsloth/Qwen3.8-Flash-Next-GGUF
BASE="https://huggingface.co/${REPO}/resolve/main/MTP"
DEST=models/MTP
mkdir -p "$DEST"

# Use the STANDALONE variants, not the "-shared-" ones. The shared heads omit
# token_embd.weight and rely on a detached-head loader that borrows the target's
# embeddings; that loader is not in upstream llama.cpp yet (only in unslothai's
# fork), so upstream fails with "check_tensor_dims: tensor 'token_embd.weight'
# not found". Verified on 0ba6499, 2026-09-03.
# Q8_0 is the default: 66.1% draft acceptance vs 64.4% for Q4_K_M.
declare -A SIZE=(
  [mtp-Qwen3.8-Flash-Next-Q8_0.gguf]=4137429120
  [mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf]=2786204800
)

FILES=("mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
[[ "${1:-}" == "--both" ]] && FILES+=("mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf")

for f in "${FILES[@]}"; do
  path="$DEST/$f"
  want=${SIZE[$f]}
  have=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then
    echo "[skip] $f already complete ($have bytes)"
    continue
  fi
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
done

echo "MTP HEAD READY -> $DEST"
