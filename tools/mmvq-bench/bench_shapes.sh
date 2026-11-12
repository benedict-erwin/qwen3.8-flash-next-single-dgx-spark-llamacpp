#!/usr/bin/env bash
# Cold-L2 GB/s for the model's decode GEMV shapes (dense Q8_0 projections and the routed experts).
# usage: bench_shapes.sh <label> [nlist]     nlist = token columns per launch, default "1 2 3 4"
# Needs ./build.sh first. GGML_CUDA_MMVQ_PREFETCH=0 turns the Spark prefetch off for an A/B.
cd "$(dirname "$0")"
label=$1; nlist=${2:-"1 2 3 4"}
for n in $nlist; do
  for mk in "2560 10240" "2560 6144" "6144 2560" "2560 12288" "2560 512" "320 10240" "10240 320" "2560 640" "640 2560" "2560 248320"; do
    set -- $mk; echo -n "$label  "; ./mmvq_bench q8_0 $1 $2 $n 2>/dev/null | grep us/op
  done
  echo -n "$label  "; ./mmvq_bench q4_K 2560 640 $n moe 512 10 2>/dev/null | grep us/op
  echo -n "$label  "; ./mmvq_bench q5_1 640 2560 $n moe 512 10 2>/dev/null | grep us/op
done
