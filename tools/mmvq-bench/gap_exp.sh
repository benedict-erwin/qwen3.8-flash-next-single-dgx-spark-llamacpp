#!/usr/bin/env bash
# Post-idle penalty: GAP=<n> tiny dependent ops (COMPUTE=<n> L2-resident matmuls) before each GEMV.
# Needs ./build.sh first.
cd "$(dirname "$0")"
for gap in 0 10 20 40; do
  for mk in "2560 6144" "6144 2560" "2560 10240" "2560 32"; do
    set -- $mk; echo -n "GAP=$gap  "; GAP=$gap ./mmvq_bench q8_0 $1 $2 4 17 2>/dev/null | grep us/op
  done
done
