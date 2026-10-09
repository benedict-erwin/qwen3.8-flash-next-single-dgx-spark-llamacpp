#!/usr/bin/env bash
# Build the Unsloth llama.cpp prebuilt's exact source composition from scratch, plus the
# one-line MMQ fix (upstream PR #27044, issue #27792) that stops the prefill crash in
# MUL_MAT_ID -- so the server can run at the default -ub instead of -ub 256.
#
#   ./build-fork.sh                      # ~15 min on GB10; stop the server first if RAM is tight
#   FORK=unsloth-mixfix UBATCH=512 ./stack.sh start llamacpp
#
#   QSA_GATHER=1 ./build-fork.sh         # also applies patches/unsloth-pr165-qsa-gather.diff:
#   FORK=unsloth-qsa UBATCH=512 ./stack.sh start llamacpp
# The second patch is the port of Unsloth PR #165 (decode attention over the top-k cells only)
# extended to MTP verification batches; measured on GB10 it cuts the verification step 12-13%
# at 32k-64k context and does nothing below 24k (see OPTIMIZATION.md 2026-09-06). Optional.
#
#   HC_FUSE=1 QSA_GATHER=1 ./build-fork.sh   # also applies patches/hc-combine-fusion.diff:
#   FORK=unsloth-hcf UBATCH=1024 ./stack.sh start llamacpp
# The third patch fuses the hyper-connection combine (REPEAT, SCALE, sigmoid, SCALE, MUL, ADD over
# the residual) into one CUDA kernel and reorders the graph so the chain is contiguous; +5% prefill
# at -ub 1024 (OPTIMIZATION.md 2026-10-09/10). Written against the QSA tree, so it needs
# QSA_GATHER=1. Also applies patches/topk-deterministic.diff: with GGML_CUDA_TOPK_DETERMINISTIC=1
# in the environment the sparse-attention block selection uses a stable sort instead of
# cub::DeviceTopK, which makes greedy output reproducible beyond ~4k context (-1..2% decode).
# Optional.
#
# How: the fork's release tag is a manifest, not a tree. CI assembles upstream <tag> + the
# PR commits pinned in scripts/unsloth/pr-set.json (merged in order). patches/compose-mix.py
# reproduces that composition on a full upstream clone; then the patch is applied and the
# tree is built with the flags that matter on GB10 (CUB 3.2, native sm_121a only).
# Measured 2026-09-06: at -ub 512 the 1..600 ubatch sweep that crashed the prebuilt at 367 and
# 512 runs clean; b10798 base: TTFT 21.1 s / 39.0 tok/s vs the shipped 26.2 s / 36.7 tok/s.
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-b10798-mix-659e406}"        # fork release tag: its pr-set.json defines the mix (b10798: +6% decode vs b10715, measured)
UPSTREAM_TAG="${TAG%%-*}"               # b10715
QSA_GATHER="${QSA_GATHER:-0}"
HC_FUSE="${HC_FUSE:-0}"
[ "$HC_FUSE" = 1 ] && [ "$QSA_GATHER" != 1 ] && { echo "HC_FUSE=1 needs QSA_GATHER=1 (the patch is written against that tree)" >&2; exit 2; }
OUT="forks/${FORK_OUT:-$([ "$HC_FUSE" = 1 ] && echo unsloth-hcf || { [ "$QSA_GATHER" = 1 ] && echo unsloth-qsa || echo unsloth-mixfix; })}"
MANIFEST="forks/src-manifest-$TAG"
SRC="forks/src-mix-$UPSTREAM_TAG"
PATCH=patches/mmq-ids-tail-padding.patch
PATCH_QSA=patches/unsloth-pr165-qsa-gather.diff
PATCH_HCF=patches/hc-combine-fusion.diff
PATCH_TOPK=patches/topk-deterministic.diff   # applied with HC_FUSE=1: GGML_CUDA_TOPK_DETERMINISTIC=1 (OPTIMIZATION.md 2026-10-10)

command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit first (13.0 was used here)" >&2; exit 1; }
[ -d "$MANIFEST/.git" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/unslothai/llama.cpp "$MANIFEST"
[ -d "$SRC/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$SRC"   # full clone: merges need history

( cd "$SRC" && git config merge.conflictStyle diff3 \
  && python3 "../../patches/compose-mix.py" "$UPSTREAM_TAG" "../../$MANIFEST/scripts/unsloth/pr-set.json" "../../$MANIFEST/scripts/unsloth/additive_merge.py" )

( cd "$SRC"
  if git apply --check "../../$PATCH" 2>/dev/null; then git apply "../../$PATCH"; echo "[patch] applied $PATCH"
  elif git apply --reverse --check "../../$PATCH" 2>/dev/null; then echo "[patch] already applied"
  else echo "!! $PATCH does not apply to the composed tree; upstream may have fixed or moved the line" >&2; exit 1; fi
  if [ "$QSA_GATHER" = 1 ]; then
    if git apply --check "../../$PATCH_QSA" 2>/dev/null; then git apply "../../$PATCH_QSA"; echo "[patch] applied $PATCH_QSA"
    elif git apply --reverse --check "../../$PATCH_QSA" 2>/dev/null; then echo "[patch] already applied $PATCH_QSA"
    else echo "!! $PATCH_QSA does not apply; it was written against mix b10798-mix-659e406" >&2; exit 1; fi
  elif git apply --reverse --check "../../$PATCH_QSA" 2>/dev/null; then
    git apply --reverse "../../$PATCH_QSA"; echo "[patch] removed $PATCH_QSA (QSA_GATHER=0)"
  fi
  if [ "$HC_FUSE" = 1 ]; then
    if git apply --check "../../$PATCH_HCF" 2>/dev/null; then git apply "../../$PATCH_HCF"; echo "[patch] applied $PATCH_HCF"
    elif git apply --reverse --check "../../$PATCH_HCF" 2>/dev/null; then echo "[patch] already applied $PATCH_HCF"
    else echo "!! $PATCH_HCF does not apply; it was written against the QSA tree of mix b10798-mix-659e406" >&2; exit 1; fi
  elif git apply --reverse --check "../../$PATCH_HCF" 2>/dev/null; then
    git apply --reverse "../../$PATCH_HCF"; echo "[patch] removed $PATCH_HCF (HC_FUSE=0)"
  fi
  if [ "$HC_FUSE" = 1 ]; then
    if git apply --check "../../$PATCH_TOPK" 2>/dev/null; then git apply "../../$PATCH_TOPK"; echo "[patch] applied $PATCH_TOPK"
    elif git apply --reverse --check "../../$PATCH_TOPK" 2>/dev/null; then echo "[patch] already applied $PATCH_TOPK"
    else echo "!! $PATCH_TOPK does not apply" >&2; exit 1; fi
  elif git apply --reverse --check "../../$PATCH_TOPK" 2>/dev/null; then
    git apply --reverse "../../$PATCH_TOPK"; echo "[patch] removed $PATCH_TOPK (HC_FUSE=0)"
  fi
  cmake -B build -S . -DGGML_CUDA=ON -DGGML_CUDA_CUB_3DOT2=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF
  cmake --build build -j"${JOBS:-8}" --target llama-server )

mkdir -p "$OUT"
for f in "$SRC"/build/bin/*; do ln -sf "$PWD/$f" "$OUT/$(basename "$f")"; done
echo "llama.cpp version: mix $TAG composed locally ($(git -C "$SRC" rev-parse --short HEAD)) + $PATCH$([ "$QSA_GATHER" = 1 ] && echo " + $PATCH_QSA")$([ "$HC_FUSE" = 1 ] && echo " + $PATCH_HCF + $PATCH_TOPK")" > "$OUT/BUILD_INFO.txt"
LD_LIBRARY_PATH="$PWD/$OUT" "$OUT/llama-server" --version 2>&1 | head -1
echo "FORK READY -> $OUT   (use: FORK=$(basename "$OUT") UBATCH=512 ./stack.sh start llamacpp)"
