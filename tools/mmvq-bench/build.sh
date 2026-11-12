#!/usr/bin/env bash
# Build mmvq_bench against a llama.cpp tree that was built with CUDA (default: the tree
# ./build-fork.sh composes). TREE=<path> picks another one.
set -euo pipefail
cd "$(dirname "$0")"
TREE="${TREE:-../../forks/src-mix-b10798}"
LIB="$(cd "$TREE/build/bin" && pwd)" || { echo "no build under $TREE -- run ./build-fork.sh first" >&2; exit 1; }
g++ -O2 -std=c++17 -I "$TREE/ggml/include" mmvq_bench.cpp -o mmvq_bench \
  -L "$LIB" -lggml -lggml-base -lggml-cuda -Wl,-rpath,"$LIB"
echo "built ./mmvq_bench against $TREE"
