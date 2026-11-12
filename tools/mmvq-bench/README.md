# mmvq-bench

Cold-L2 benchmark for the CUDA GEMV kernels (`mul_mat_vec_q`, and `mul_mat_id` for the routed
experts) that decode spends about 80% of its GPU time in. `test-backend-ops perf` re-reads one
matrix, so anything under the GB10's 24 MB L2 reports 700–1200 GB/s; this harness multiplies N
distinct weight copies (≥ 256 MB per graph) so every launch streams from DRAM, as a decode step
does. Background and conclusions: `OPTIMIZATION.md`, 2026-10-11.

```bash
./build.sh                       # needs a CUDA build of llama.cpp, default forks/src-mix-b10798 (./build-fork.sh)
./mmvq_bench q8_0 2560 6144 4    # dense: W[2560 -> 6144], 4 token columns
./mmvq_bench q4_K 2560 640 4 moe 512 10   # experts: 512 experts, 10 used per token
./bench_shapes.sh base           # every decode shape of this model, 1-4 columns
GGML_CUDA_MMVQ_PREFETCH=0 ./bench_shapes.sh pf0   # same, Spark prefetch off (HC_FUSE=1 builds)
./gap_exp.sh                     # how a DRAM-idle phase before a GEMV slows it down
```

Stop the server first (`./stack.sh stop`): the harness allocates on the same GPU and the numbers
are bandwidth measurements. `results/` holds the 2026-10-11 output behind the tables in
`OPTIMIZATION.md` (`shapes-base.txt` before the prefetch patch, `shapes-pf.txt` with it,
`gap-base.txt` the post-idle experiment).
