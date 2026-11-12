#!/usr/bin/env bash
# Make a lighter MTP draft head: requantize only the head's own vocabulary projection
# (output.weight, 2560 -> 248320) from Q8_0 to Q4_K. That one tensor is read once per
# drafted token, three times per step at --nmax 3; halving it measured 0-10% more decode
# depending on the build and the workload (OPTIMIZATION.md, 2026-11-12). The target model verifies every
# draft, and GSM8K / HumanEval+ base were unchanged with this head. Every other tensor
# keeps its type; the downloaded head is left untouched. Takes a few seconds, no download.
#
#   ./requant-mtp-head.sh            # writes models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-outQ4_K.gguf
#   FORK=unsloth-hcf ./requant-mtp-head.sh   # take llama-quantize from another build under forks/
#
# Then point the server at it, per start or as a line in stack.local:
#   MTP_MODEL=models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-outQ4_K.gguf
set -euo pipefail
cd "$(dirname "$0")"

IN=models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
OUT=models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0-outQ4_K.gguf
FORKDIR="forks/${FORK:-unsloth-b10715}"
BIN="$FORKDIR/llama-quantize"

[ -f "$IN" ]  || { echo "missing $IN -- run ./download-mtp.sh" >&2; exit 1; }
[ -x "$BIN" ] || { echo "missing $BIN -- run ./download-fork.sh (or set FORK=<build under forks/>)" >&2; exit 1; }
if [ -f "$OUT" ]; then echo "[skip] $OUT already present"; else
  export LD_LIBRARY_PATH="$PWD/$FORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # File type Q8_0 keeps the other tensors as they are; only output.weight is overridden.
  "$BIN" --allow-requantize --output-tensor-type q4_k "$IN" "$OUT.part" Q8_0 "$(nproc)" 2>&1 \
    | grep -E 'output\.weight|quant size' || true
  [ -s "$OUT.part" ] || { echo "llama-quantize wrote nothing" >&2; exit 1; }
  mv "$OUT.part" "$OUT"
fi
echo "HEAD READY -> $OUT ($(stat -c%s "$OUT") bytes)"
echo "use it:  MTP_MODEL=$OUT ./stack.sh start llamacpp     (or put that MTP_MODEL= line in stack.local)"
