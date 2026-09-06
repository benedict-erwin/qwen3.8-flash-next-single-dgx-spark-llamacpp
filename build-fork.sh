#!/usr/bin/env bash
# Build the Unsloth llama.cpp prebuilt's exact source composition from scratch, plus the
# one-line MMQ fix (upstream PR #27044, issue #27792) that stops the prefill crash in
# MUL_MAT_ID -- so the server can run at the default -ub instead of -ub 256.
#
#   ./build-fork.sh                      # ~15 min on GB10; stop the server first if RAM is tight
#   FORK=unsloth-mixfix UBATCH=512 ./stack.sh start llamacpp
#
# How: the fork's release tag is a manifest, not a tree. CI assembles upstream <tag> + the
# PR commits pinned in scripts/unsloth/pr-set.json (merged in order). patches/compose-mix.py
# reproduces that composition on a full upstream clone; then the patch is applied and the
# tree is built with the flags that matter on GB10 (CUB 3.2, native sm_121a only).
# Measured 2026-09-06: this build matches the prebuilt's TTFT and decode, and the 1..600
# ubatch sweep that crashed the prebuilt at 367 and 512 runs clean at -ub 512.
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-b10715-mix-86bd2d3}"        # fork release tag: its pr-set.json defines the mix
UPSTREAM_TAG="${TAG%%-*}"               # b10715
OUT="forks/${FORK_OUT:-unsloth-mixfix}"
MANIFEST="forks/src-manifest-$TAG"
SRC="forks/src-mix-$UPSTREAM_TAG"
PATCH=patches/mmq-ids-tail-padding.patch

command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit first (13.0 was used here)" >&2; exit 1; }
[ -d "$MANIFEST/.git" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/unslothai/llama.cpp "$MANIFEST"
[ -d "$SRC/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$SRC"   # full clone: merges need history

( cd "$SRC" && git config merge.conflictStyle diff3 \
  && python3 "../../patches/compose-mix.py" "$UPSTREAM_TAG" "../../$MANIFEST/scripts/unsloth/pr-set.json" "../../$MANIFEST/scripts/unsloth/additive_merge.py" )

( cd "$SRC"
  if git apply --check "../../$PATCH" 2>/dev/null; then git apply "../../$PATCH"; echo "[patch] applied $PATCH"
  elif git apply --reverse --check "../../$PATCH" 2>/dev/null; then echo "[patch] already applied"
  else echo "!! $PATCH does not apply to the composed tree; upstream may have fixed or moved the line" >&2; exit 1; fi
  cmake -B build -S . -DGGML_CUDA=ON -DGGML_CUDA_CUB_3DOT2=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF
  cmake --build build -j"${JOBS:-8}" --target llama-server )

mkdir -p "$OUT"
for f in "$SRC"/build/bin/*; do ln -sf "$PWD/$f" "$OUT/$(basename "$f")"; done
echo "llama.cpp version: mix $TAG composed locally ($(git -C "$SRC" rev-parse --short HEAD)) + $PATCH" > "$OUT/BUILD_INFO.txt"
LD_LIBRARY_PATH="$PWD/$OUT" "$OUT/llama-server" --version 2>&1 | head -1
echo "FORK READY -> $OUT   (use: FORK=$(basename "$OUT") UBATCH=512 ./stack.sh start llamacpp)"
