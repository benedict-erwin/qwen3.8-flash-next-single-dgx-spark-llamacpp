# Optimal recipe for Qwen3.8-Flash-Next on DGX Spark GB10

> Follow-up research 2026-09-02. Supersedes part of the conclusions in `RESULTS.md` (see "Correction" below).
> All tensor size numbers in this document are parsed directly from the local GGUF
> (`models/UD-Q4_K_XL/*.gguf`), not from the model card.

> ## ⚠️ EXECUTION RESULTS 2026-09-03 — several claims below are REFUTED
>
> The plan in this document has been executed through step 5. In short: **not a single
> path produced a speedup.** Details in the "Execution results" section
> below. What has to be read with a correction:
>
> - **"+3-10% from the build update"** → actually **+0.4%** (within noise). The
>   +24.7% figure I reported at one point is an artefact of the benchmark method, not a speedup.
> - **"MTP available as a 2.60 GB sidecar"** → true that the file exists, **false**
>   that upstream can use it. Both variants are rejected by `llama.cpp` 0ba6499.
> - **"Projected 31-39 tok/s"** → not proven and currently blocked.
> - **"origin/master already wires MTP into qwen4exp"** → the graph is there, but
>   the loader for a separate draft head is not. Grepping 35 nextn/mtp references is not enough
>   as evidence; it should have been tested, not inferred.
> - The correct **roofline efficiency**: **57%** (23.71 / 41.4), not 51% or 66%.

## TL;DR

1. **MTP turns out to be AVAILABLE for the model we already have.** The old conclusion ("MTP failed, we have to swap GGUFs, another big download") is **wrong**. Unsloth released the MTP head as a **separate 2.60 GB sidecar file** that is attached alongside our current UD-Q4_K_XL — no need to re-download 104 GB.
2. **Our llama.cpp is 40 commits behind**, and `origin/master` **already** wires MTP into the `qwen4exp` architecture. When we tested on 2026-09-01, that support genuinely was not in our build. A `git pull` + rebuild is enough.
3. **Our decode is already memory-bandwidth bound, not misconfigured.** Our efficiency is 51–53% of roofline — the same as other publicly reported GB10 systems. That means there is no "magic flag" left; the only way to gain significantly is to **reduce bytes per token** (a smaller quant) or **raise tokens per byte** (speculative decoding).
4. **Recommendation: a combined recipe** — UD-Q4_K_XL (what we already have) + MTP sidecar + `-lm mmap` + latest llama.cpp. Projected **31–39 tok/s** `[estimate]` with retention still 92.3%, resident down from ~104 → **~77 GiB**.

---

## 1. The physical limit: where we actually stand

### Model anatomy (parsed directly from the local GGUF)

| Tensor group | GiB | % of file | Read per token? |
|---|---|---|---|
| MoE experts (`*_exps`) | 71.73 | 69.2% | **No** — only 10 of 512 experts |
| PLE (`per_layer_token_embd`) | 26.82 | 25.9% | No — just a row lookup |
| attn / ssm / norm | 3.86 | 3.7% | **Yes, in full** |
| `token_embd` + `output` | 1.26 | 1.2% | Partly (`output` in full) |
| **TOTAL** | **103.68** | | |

Validation: tensor total 103.68 GiB = file size on disk 103.69 GiB. PLE measures **4.500 bits/element** for 51.2 billion elements — matching the "26.82 GB lookup table" figure that circulates in llama.cpp discussions.

### Decode roofline

Bytes that must be read per token = full attn/ssm + 10/512 experts + output head:

```
attn/ssm/norm (full)      3.86 GiB   63%
active experts (10/512)   1.40 GiB   23%
output head               0.63 GiB   10%
--------------------------------------------
per token                 5.90 GiB = 6.34 GB
```

GB10 = 273 GB/s → **roofline ≈ 43 tok/s**.

| Configuration | Size | Measured | % of roofline |
|---|---|---|---|
| UD-Q4_K_XL — ours, 2026-09-01 | 103.69 GiB | 21.8 tok/s | **51%** |
| UD-Q4_K_XL — NVIDIA forum | 103.69 GiB | 25.0 tok/s | 58% |
| UD-IQ1_S — kubesimplify (tg128) | 67.55 GiB | 34.5 tok/s | 54% |

**This is the most important finding of this research.** Two different quants, on the same hardware, land at almost identical bandwidth efficiency (51–54%). The speed ratio 34.5/21.8 = 1.58 is almost exactly the size ratio 103.69/67.55 = 1.54.

Meaning: **our setup is not misconfigured.** The 34.5 tok/s figure going around is not a "better recipe", it is simply a **much smaller and much more damaged quant** (IQ1_S = 3.28 bpw effective). Stop looking for a missing flag — look for ways to reduce bytes/token or raise tokens/byte.

---

## 2. Correction to `RESULTS.md`

`RESULTS.md` states: *"MTP FAILED — the Unsloth GGUF has no MTP/nextn head... To really get MTP: you have to switch to an MTP-bearing GGUF, a different quant, another big download."*

**The first part is right, the conclusion is wrong.** What is actually true:

- The main model indeed has no `nextn` tensors (grep = 0 tensors — still true today).
- But the Flash-Next MTP head is **a whole qwen4exp block (4B params)** that can stand as a **separate draft file** and be attached with `-md`. Not something that has to be fused into the target GGUF.
- Unsloth has already released it in `unsloth/Qwen3.8-Flash-Next-GGUF` under the `MTP/` directory:

  | File | Size | Draft acceptance |
  |---|---|---|
  | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.60 GB | 66.1% |
  | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.78 GB | 64.4% |
  | `mtp-Qwen3.8-Flash-Next-shared-BF16.gguf` | 4.87 GB | 66.5% |

- There is also a third-party route, `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` (2.5 GB Q4_K_M), explicitly **"tested with unsloth's UD-Q4_K_XL and UD-IQ4_XS"** — exactly our files.

**Why it failed back then:** our build `5d4a3be` (2026-08-31) did not yet have the MTP graph for `qwen4exp`. Checked today: in our commit, the `LLM_ARCH_QWEN4EXP` case has **0** nextn/mtp references; in `origin/master` today there are **35**. The error was correct, but its cause was the llama.cpp version, not the absence of an MTP head.

---

## 3. What we are behind on in llama.cpp (40 commits)

Relevant commits between `5d4a3be` and `origin/master`:

| Commit | Content | Impact |
|---|---|---|
| `0eadefe` | `qwen4exp: support recurrent state rollback` (#28123) | **MTP prerequisite** — SSM state rollback when a draft is rejected |
| `36b1015` | `qwen4exp: fix seq_cp, block position keying, cuda abort` (#27941) | Stability |
| `09412af` | `qwen4exp: sum the indexer heads by slices` (#28023) | Indexer perf |
| `3466812` | `cuda: fuse MoE weighted expert reduction` (#25952) | **Faster MoE decode** |
| `e4b9af0` | `CUDA: XOR swizzle flash attn K,V smem fp16 tiles` (#25635) | Faster flash-attention |

The last two give a direct speedup with no MTP at all `[estimate]` +3–10%.

---

## 4. Ranking the options

### ⭐ A+. UD-Q4_K_XL (existing) + MTP sidecar + mmap — **recommended**

```bash
llama-server \
  -m models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  -ngl 999 -lm mmap -ot "per_layer_token_embd.weight=CPU" \
  -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 32768
```

- Download: **2.60 GB** only. The existing 104 GB model is used as-is.
- Retention stays **92.3%** — MTP is *lossless*: wrong drafts are rejected, output is identical to non-spec.
- Projected **31–39 tok/s** `[estimate]`, basis: Strix Halo (unified memory, similar bandwidth class) 20.3 → 35.8 tok/s (+76%) on code; M3 Max 27.4 → 38.8 (+42%). Prose will be at the low end, code at the high end.
- `-lm mmap` puts the 26.82 GiB PLE in **page cache** (reclaimable) instead of resident → resident ~**77 GiB**.

### A. UD-Q4_K_XL as-is (status quo)
21.8 tok/s, 92.3%, resident ~104 GiB. The proven safe point.

### B. blazux NVFP4 + vLLM
Resident ~76 GiB, decode 26 tok/s (NVFP4) / **31 tok/s** (hybrid fp8 side-layer), **prefill 1,500–2,000 tok/s** — 20–40× our prefill. A docker + patched-vLLM stack, not llama.cpp.
- This is the outright winner **if prompts are long** (agentic, RAG, codebase). Our ~50–800 tok/s prefill is the biggest weakness of option A, and MTP **does not improve prefill at all**.

### C. UD-IQ4_XS + MTP
93.7 GiB, retention 89.6%, per-token ~5.3 GiB → baseline ~24 tok/s, with MTP ~34–43 `[estimate]`. Trades 2.7 points of retention for ~10% speed. Less attractive than A+.

### D. UD-IQ1_S
34.5 tok/s but 3.28 bpw — retention far below the 90% target. Only for draft/experiments.

---

## 5. Can we build our own recipe? Yes — and the analysis points in the opposite direction from intuition

Per-token traffic breakdown, by tensor family (excluding experts & PLE):

| Tensor family | GiB | % of traffic per token |
|---|---|---|
| `attn_qkv.weight` | 0.934 | 15.2% |
| `output.weight` | 0.629 | 10.2% |
| `attn_gate.weight` | 0.560 | 9.1% |
| `ssm_out.weight` | 0.560 | 9.1% |
| `attn_q.weight` | 0.374 | 6.1% |
| `ffn_gate_inp.weight` (router) | 0.234 | 3.8% |
| `attn_output.weight` | 0.187 | 3.0% |
| `hc_*` (hyper-connections) ×4 | 0.624 | 10.2% |

**The usable insight:** the conventional quant rule — "keep attention at high precision, squeeze the experts" — is **inverted for decode speed on this model**:

- Experts = **69% of the file** but only **23% of traffic/token**. Squeezing experts Q4→Q3 cuts ~18 GiB off the file but decode only +6% `[estimate]`, at a quality cost that spreads across every domain.
- attn/ssm = **3.7% of the file** but **63% of traffic/token**. Dropping the top 4 families (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q` = 2.43 GiB = 41% of traffic) from ~5 bpw to 4 bpw cuts ~0.49 GiB/token → **+9% decode** with the file size almost unchanged.

So a custom quant recipe for this hardware would take the shape: **leave experts as high as possible (quality is cheap there), squeeze the per-token tensors**. This is the opposite of the Baekpica recipe, which raises the precision of edge layers.

**But:** the maximum result is only ~+10–15%, while MTP gives +40–76% with no quality sacrifice at all. And rolling our own quant needs either a Q8_0 188 GB or BF16 355 GB source (big download → must confirm first) plus hours of imatrix runs.

**Conclusion: do MTP first. A custom quant only makes sense if it is still too slow after MTP**, and it is better aimed at saving *memory* (so it fits alongside Ollama) than at speed.

---

## 6. Staged execution plan

| # | Step | Cost | Expectation | Risk |
|---|---|---|---|---|
| 1 | `git pull` llama.cpp + CUDA rebuild | ~15 min | +3–10%, unlocks MTP | Low — separate build, old binary kept |
| 2 | Re-bench the baseline with the new build | ~10 min | A clean comparison number | — |
| 3 | Test `--spec-type ngram-mod` (**never tried** — yesterday we used `ngram-cache`) | ~10 min | Free; sparkrun reports up to ~45 tok/s on copy-heavy work | — |
| 4 | Download the Q8_0 MTP head (2.60 GB) | ~5 min | — | — |
| 5 | Bench MTP `n-max` 2 vs 3, prose vs code | ~20 min | **31–39 tok/s** | Medium — spec can regress on certain backends |
| 6 | Add `-lm mmap`, measure resident | ~10 min | ~104 → ~77 GiB | Low |
| 7 | *(optional)* blazux vLLM stack if prefill becomes the problem | ~2 hours + docker | prefill 20–40× | High — custom stack |

Total for steps 1–6: **± 1 hour**, 2.6 GB download. Below the "ask first" threshold (>30 min of processing / >10 GB download) except for step 7.

### Memory note — relevant to the 2026-09-01 incident

`-lm mmap` is not only about saving. The 26.82 GiB PLE in page cache is **reclaimable**: when memory gets tight, the kernel drops the page cache and the model slows down. Without mmap that allocation is resident and running out of memory ends in `NVRM NV_ERR_NO_MEMORY` → stall → watchdog panic, exactly what happened on 2026-09-01.

**`-lm mmap` changes the failure mode from a kernel panic to merely slow.** That alone is reason enough to use it, regardless of the resident numbers.

Even so, resident ~77 GiB + KV is still not comfortable alongside the current Ollama load (~36.5 GiB): 77 + 36.5 = 113.5 of 121.7 GiB, and the PLE page cache would be evicted continuously → decode collapses. Decision #4 (stop Ollama before serving) still stands.

## 7. Not yet verified

- The Unsloth MTP speedup figure (1.67×, "83.2 → 138.8 tok/s") is clearly **not** from a GB10 — the hardware is not stated `[Unverified]`. The only thing transferable to our case is the **acceptance rate ~66%**.
- The 31–39 tok/s projection is derived from Strix Halo & M3 Max (both unified memory) `[estimate]`. PR #27836 notes that CUDA unified memory gets "similar gains", but I found no GB10-specific numbers.
- `--spec-type ngram-mod` we have never measured at all.
- NVFP4 accuracy retention (option B): **measured ourselves 2026-09-04**, on par with UD-Q4_K_XL
  on GSM8K and HumanEval+ — see the "Accuracy retention: measured" section below. The RadixArk
  model card reports GSM8K 97.27 vs BF16 97.12–97.50 (vendor-reported).
- There is no DFlash/DSpark draft head for Flash-Next (the ones circulating are only for Qwen3.8-27B) — that route is closed for now.

## Sources

- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md
- https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF
- https://github.com/ggml-org/llama.cpp/pull/27836 · https://github.com/ggml-org/llama.cpp/discussions/27864
- https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/qwen38-flash-next-image.md
- https://blog.kubesimplify.com/running-qwen3-8-flash-next-on-dgx-spark-and-rtx-pro-6000
- https://github.com/blazux/qwen3.8-Flash-DGX
- https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720
- https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/


---

# Execution results 2026-09-03

All numbers from `runs/bench-2026-09-03.jsonl`, 3 runs per configuration, `n_predict` 256, temperature 0.

## Methodological finding: the old benchmark overstated by 14.6%

The old `bench.sh` sent **the same prompt 3 times at temperature 0**. Because
the output was identical every run, the server could exploit slot/KV reuse — and for
speculative decoding, the n-gram cache simply copied the previous run's output.

| Method | old build | new build | + ngram-mod |
|---|---|---|---|
| A: same prompt repeated 3× | 21.80 | **27.18** | **86.05** |
| B: 5 different prompts rotated | 23.62 | **23.71** | **23.58** |

The A vs B gap on the same build: **+14.6%** without speculation, and **+263%** with
ngram-mod. `bench.sh` now rotates 5 different prompts; method A must not be used again.

## The real results (method B)

| Configuration | Median tok/s | vs baseline | Draft accepted |
|---|---|---|---|
| build 5d4a3be (old) | 23.62 | — | — |
| build 0ba6499 (63 commits newer) | 23.71 | **+0.4%** | — |
| + `--spec-type ngram-mod` | 23.58 | **-0.6%** | 0 |
| + `--spec-type draft-mtp` | — | **FAILED TO LOAD** | — |

**The llama.cpp update gives no measurable decode speedup.** `cuda: fuse MoE
weighted expert reduction` (#25952) and `XOR swizzle flash attn` (#25635) change
nothing for this model on a single-stream load.

**ngram-mod gives no gain on prose** — zero drafts accepted with varied prompts,
consistent with the `ngram-cache` result of 2026-09-01 (16% acceptance). The old conclusion that
ngram "has never been tried and might be different" is now answered: it is the same.

## MTP is blocked in upstream llama.cpp

Both Unsloth head variants are rejected by build 0ba6499:

| Variant | Size | Tensors | Error |
|---|---|---|---|
| `mtp-...-shared-Q8_0.gguf` | 2.60 GiB | 32 | `tensor 'token_embd.weight' not found` |
| `mtp-...-Q8_0.gguf` (standalone) | 3.85 GiB | 34 | `tensor 'output_hc_norm.weight' not found` |

Verified by diffing the tensor set against the target model (1224 tensors):

| Tensor | target | MTP standalone | MTP shared |
|---|---|---|---|
| `token_embd.weight` | present | present | **absent** |
| `output.weight` | present | present | **absent** |
| `output_hc_norm.weight` | present | **absent** | **absent** |

Upstream treats the draft model as a complete `qwen4exp` model and requires the full
tensor set (`LLM_TENSOR_HC_HEAD_NORM` is registered at `llama-arch.cpp:522`). The Unsloth
head is built for the `unslothai/llama.cpp` fork, which knows it is a partial head.

**Remaining paths to MTP:**
1. Build the `unslothai/llama.cpp` fork at tag `b10715-mix-86bd2d3` or newer.
2. Wait for PR #27836 to merge — still in draft status, plus it needs `crusaderky`'s
   detached-head loader commit a82a58a.
3. Add `output_hc_norm.weight` to the head GGUF ourselves (tensor patch) — untested,
   and not necessarily sufficient since there may be other required tensors.

## Memory status (measured, not estimated)

With the default `--load-mode auto` (which turns out to already be mmap):

- used with the model loaded: **83.4-84.3 GiB**, not ~104 GiB as first assumed
- page cache: ~24 GiB
- available left: ~35 GiB
- load time: **44.5 seconds** (warm)

The claim that `-lm mmap` would drop resident from 104 → 77 GiB is **irrelevant** — the default
is already mmap. Any additional benefit from that flag is unmeasured.

## Conclusions that survive

What still stands from the initial analysis: **decode is memory-bandwidth bound**. The honest
efficiency is 57% of a 41.4 tok/s roofline. But there is no cheap path to raise it —
build update zero, ngram zero, MTP blocked. What genuinely remains:

1. **The unslothai fork** for MTP — the only path with actual numbers on comparable
   hardware (Strix Halo 20.3 → 35.8).
2. **blazux NVFP4 + vLLM** — 31 tok/s hybrid, prefill 20-40× faster. A different stack.
3. **Accept 23.7 tok/s** with 92.3% retention.

---

# Unsloth fork results + coding-agent workload (2026-09-03, follow-up session)

The real workload: **a coding agent with long prompts**. `bench.sh` (prose, ~30-token
prompts) is not representative of that, so `bench-code.sh` was written: ~10.9K-token prompts from
real `llama.cpp` source + a coding task, 3 different files & 3 different tasks per run.

The Unsloth fork is used via the **prebuilt** `app-b10715-mix-86bd2d3-linux-arm64-cuda13-portable`
(180 MB, no compile) — `BUILD=fork ./serve.sh`.

## MTP WORKS on the fork

Prompt 10,886 tokens, output 256 tokens, median of 3 runs:

| Configuration | Prefill tok/s | TTFT | Decode tok/s | Acceptance | RAM |
|---|---|---|---|---|---|
| upstream `0ba6499` | 337.8 | 33.1s | 22.11 | — | 84.3 GiB |
| fork `b10715` | **525.3** | **20.8s** | 24.07 | — | 84.3 GiB |
| fork + MTP n-max 2 | 505.6 | 21.7s | **36.45** | **94.3%** | 88.8 GiB |
| fork + MTP n-max 3 | 505.7 | 21.7s | **37.81** | 92.8% | 89.1 GiB |

**Two separate wins:**

1. **MTP: decode 24.07 → 37.81 tok/s (+57%)**, acceptance **92–94%** on code. That is far above
   the expectation from the Strix Halo data (0.90 on code) and explains why yesterday's prose test
   was misleading — prose is the worst case for MTP, code the best case. Lossless: wrong
   drafts are rejected, output identical to non-spec.
2. **The fork prefills faster: 337.8 → 525.3 tok/s (+55%)**, TTFT 33.1s → 20.8s, even without
   MTP. This was unexpected and is not an MTP effect.

MTP's cost: **+4.5 GiB** RAM (84.3 → 88.8 GiB) for the Q8_0 head.

## End-to-end for a single agent call

| Configuration | TTFT | Decode | Total | Prefill share |
|---|---|---|---|---|
| upstream | 33.1s | 11.6s | **44.7s** | 74% |
| fork | 20.8s | 10.6s | 31.5s | 66% |
| fork + MTP n3 | 21.7s | 6.8s | **28.5s** | **76%** |

**End-to-end improvement 44.7s → 28.5s (+57%).** The recommended configuration:
`BUILD=fork ./serve.sh --mtp --nmax 3`.

## But prefill remains the dominant problem

After MTP, **76% of the time per call goes to prefill**. Decode is no longer the bottleneck.
For a coding agent, this is what determines how it feels to use:

- prompt 10.9K → TTFT ~21s
- prompt 32K → TTFT ~63s `[estimate]`, linear extrapolation from 505 tok/s

blazux NVFP4+vLLM claims prefill of **1,500–2,000 tok/s** (3–4× this fork) plus prefix caching
with TTFT 1.4s for 20K tokens it has already seen. For an agent that calls repeatedly
with a growing context, that prefix caching could matter more than the raw
prefill number.

**Tiered recommendation:**
1. **Now:** use `BUILD=fork ./serve.sh --mtp --nmax 3`. Free, already measured, +57%.
2. **Next:** evaluate blazux vLLM specifically for prefill + prefix caching. Requires downloading
   ~122 GiB of NVFP4 weights (confirm first) and NVFP4 accuracy retention has not been measured by a third party.
3. Consequence of using the fork: behind upstream, dependent on Unsloth releases. For now
   it is worth it — upstream cannot load the MTP head at all.

---

# Final comparison: llama.cpp fork+MTP vs vLLM NVFP4 (2026-09-03 evening)

Measured from the client side over streaming SSE for both stacks (`bench-stream.py`), identical
~10.9K-token prompts from real source, 256 tokens of output, `finish_reason=length` in every
run. Token counts taken from the `usage.completion_tokens` reported by the server.

## Results

| Stack | TTFT | Decode | Total/call | RAM | Load |
|---|---|---|---|---|---|
| llama.cpp fork+MTP n3 | 21.59s | **37.77** | 28.4s | **88 GiB** | **45s** |
| vLLM NVFP4 | **11.30s** | 30.27 | **19.8s** | 112 GiB | 14 min |
| llama.cpp — prefix cache hit | 1.82s | 35.83 | **9.0s** | | |
| vLLM — prefix cache hit | 1.86s | 30.71 | 10.2s | | |

- **Cold** (a new context every call): vLLM is **1.44× faster** — a 1.9× prefill covers
  a decode that is 1.25× slower.
- **Warm** (same prefix): llama.cpp is **1.14× faster**, and **prefix caching is
  practically equal on both (1.82s vs 1.86s)**.
- **Break-even: vLLM wins when >12% of calls are cold.**

## Correction: prefix caching is NOT a vLLM-specific advantage

Throughout this research I treated prefix caching as the main reason to move to vLLM,
citing the blazux claim of "TTFT 1.4s for 20K tokens". Measured: **llama.cpp has the same
capability and is just as fast** (1.82s vs 1.86s on a 10.9K prompt). The argument for switching stacks because of
prefix caching does not hold.

## Three methodological flaws found today

All of them produced numbers that looked plausible while measuring something else:

1. **Repeated prompt + temperature 0** → 14.6% inflation without speculation, 263% with ngram-mod.
2. **`"model": "x"`** → vLLM replies HTTP 404 (llama.cpp ignores this field). The model name is now
   read from `/v1/models`.
3. **Counting SSE chunks as tokens** → vLLM packs ~2.6 tokens per chunk, making decode
   look like 9.7 tok/s when it was 30.3. Now uses `usage.completion_tokens`.

Plus: the first request after server start sometimes fails/comes back empty and was included in the median.
`bench-stream.py` now discards one warm-up run and rejects failed runs explicitly.

## Recommendation: stay on llama.cpp fork + MTP

```bash
BUILD=fork ./serve.sh --mtp --nmax 3
```

vLLM's 1.44× advantage applies only to cold calls. For a coding agent with a stable system prompt
and a context that grows at the end, the majority of calls are cache hits — the region where
llama.cpp is in fact slightly faster. Beyond speed:

| | llama.cpp fork | vLLM NVFP4 |
|---|---|---|
| RAM | 88 GiB (33 left) | 112 GiB (~8 left) |
| Restart | 45 seconds | 14 minutes |
| Accuracy retention (measured, thinking off) | GSM8K 97.3%, HumanEval+ 93.9% | GSM8K 97.0%, HumanEval+ 95.7% |
| Stack | a single binary | docker + patched vLLM |

vLLM's memory margin (~8 GiB) is dangerous on this machine given the 2026-09-01 watchdog incident.

**Moving to vLLM only makes sense if** the workload turns out to be dominated by always-new
context (>12% cold), e.g. scanning many different files each call. That can be
measured from real usage patterns, not guessed now: `./cache-ratio.py -v` reads
`runs/serve-current.log` after a real session and reports the ratio of cold calls against this
break-even point. The benchmark log is deliberately 100% cold (a different prompt every run), so it is not representative.
**Measured 2026-09-05** (bottom-most section): 8.7% cold, 95.9% of tokens from cache.

---

## 2026-09-04: CUDA crash during prefill — fork AND upstream, mitigated with `-ub 256`

Found while running `bench-accuracy.py` (HumanEval+): llama-server died on problem
`HumanEval/68` with

```
CUDA error: an internal operation failed
  in function ggml_cuda_mul_mat_cublas_impl at ggml-cuda.cu:1623 (cublasGemmEx)
```

Deterministic reproduction: freshly started server, one request with that prompt (381 prompt tokens,
thinking off), dead within 1 second. Not OOM — 116 GiB available at the time.

**Isolation** (`runs/crash-sweep-2026-09-04.jsonl`, crash logs in `runs/serve-crash-*.log`):

| Configuration | Prompt HumanEval/68 |
|---|---|
| Unsloth fork b10715 + MTP | crash |
| Unsloth fork b10715 without MTP | crash |
| upstream `0ba6499` (build-new) `--ngram-mod` | crash |
| fork + MTP, **`-ub 256`** | **runs** (474-token completion) |

So it is neither MTP nor the fork — a llama.cpp CUDA backend bug on GB10 (sm_121), present in
upstream too. A neutral prompt sweep (raw token ids via `/completion`, `cache_prompt`
off) of sizes 300–600 at the default `-ub`: only **367 and 512** crash; but a neutral 381
does *not* crash even though the HumanEval/68 prompt (381 tokens) does. The trigger is **data-dependent,
not merely batch size** — `[Inference]` the per-expert GEMM shape (how many tokens are
routed to each expert within one ubatch) hitting the problematic cuBLAS path.

**Mitigation:** `-ub 256`. With it, the 1–600 sweep is clean and HumanEval/68 runs. It is now the
default in `stack.sh` (`UBATCH=256`, overridable). This avoids every case we could
reproduce, not a guarantee — the root cause is not fixed upstream.
Why this never showed up in the 2026-09-03 benchmark: the ~10.9K-token prompts are processed
in full 2048/512 ubatches; only the last ubatch has a random size, and three prompts
were not enough to hit it. A coding agent sending hundreds of varied prompts **will**
hit it sooner or later — 468 eval requests hit it on request 369.

**2026-09-06, the kernel that actually fails: `MUL_MAT_ID`, not cuBLAS.** Retested at `-ub 512`
with a raw-token probe of size 367 (`runs/crash-sweep-2026-09-04.jsonl`, the 2026-09-06 entry):
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` f32 / bf16 / f16 all crash at the same size, and this time
the error is "illegal memory access" surfacing in `launch_mul_mat_q`, not "internal operation
failed" in `cublasGemmEx` as on the 4th. CUDA errors of this kind are asynchronous, and the reported location
is the next call that happens to check status, so neither report points at the
faulty kernel. With `CUDA_LAUNCH_BLOCKING=1` (errors become synchronous) the culprit reads out unambiguously:

```
ggml_cuda_compute_forward: MUL_MAT_ID failed
CUDA error: an illegal memory access was encountered
```

`MUL_MAT_ID` = the MoE expert matmul via ggml's MMQ kernel (Q4_K experts). This is exactly
**ggml-org/llama.cpp#27792** (still OPEN): the tail padding of the `src1_q8_1` buffer in the `mul_mat_id` path
is computed from `ne11`, which is always 1 for MoE, so the last tile of the last expert reads past
the end of the allocation; whether it is visible or not depends on VMM pool slack, which is why only certain
ubatch sizes (367, 512) and why it is data-dependent (expert routing). The fix already exists as PR
**#27044** ("size MMQ ids-path tail padding from the flattened row count") — opened 2026-08-13,
not yet merged. Our report #28377 is therefore a duplicate of #27792 with a symptom whose
location is misleading. Consequence: `GGML_CUDA_FORCE_MMQ` is in fact the wrong path;
`GGML_CUDA_FORCE_CUBLAS` (compile-time) or the #27044 patch in a self-built binary is what avoids it.
A newer Unsloth prebuilt (`b10798-mix-659e406`, 2026-09-04) is tested below.

Reported upstream as **ggml-org/llama.cpp#28377** (2026-09-04) with the sweep reproduction
above. Related candidates: #28251 (same call site, `cublasGemmEx` in the MoE path, different cuBLAS
status, RTX 3070) and #27792 (OOB read in MMQ `mul_mat_id`, a different kernel path
but likewise ubatch-dependent) `[Speculation]`.

**The cost of `-ub 256`** (`runs/bench-stream2.jsonl`, labels `llamacpp-mtp-ub256` vs `-ub512`,
~10.9K-token prompts, median of 2 valid runs): TTFT **26.2 s vs 21.9 s** (+20% prefill),
decode 36.7 vs 37.2 tok/s (equivalent). Prefill does get more expensive; that is the price of
stability until upstream fixes the kernel. Note: run 4 (source
`speculative.cpp`) failed on both configurations — the server returned 1 token then stopped.
The same prompt ran normally on vLLM on 2026-09-03. Not yet investigated `[Unverified]`.

---

## 2026-09-04: Accuracy retention — measured, not from the model card

The question left over from the backend comparison: is NVFP4 (RadixArk) worse than
UD-Q4_K_XL (Unsloth)? BF16 360 GB does not fit on GB10, so what was measured is **both quants
on an identical task set, harness and decoding settings**, then compared against each other
and against published BF16 numbers. Tool: `bench-accuracy.py`; data:
`runs/accuracy-2026-09-04.jsonl`; per-problem samples in `tmp/accuracy/<label>/` (not tracked).

Settings: thinking **off** (`chat_template_kwargs.enable_thinking=false`, both servers
honour it — 27 identical prompt tokens), temperature 0, seed 1234. GSM8K: lm-eval
`gsm8k_cot_zeroshot`, first 300 test problems, max 1024 tokens. HumanEval+: evalplus, 164 problems,
greedy, max 2048 tokens (the default 768 truncated two answers that reason at length first).

| | llama.cpp UD-Q4_K_XL + MTP | vLLM NVFP4 (RadixArk) | BF16 published |
|---|---|---|---|
| GSM8K (300, final-paragraph) | **97.3%** (292) | **97.0%** (291) | 97.1–97.5 (RadixArk card, thinking on) |
| GSM8K (300, lm-eval flexible-extract) | 86.3% | 85.3% | — |
| HumanEval+ pass@1 base | 96.3% (158/164) | 97.0% (159/164) | — |
| HumanEval+ pass@1 plus | **93.9%** (154/164) | **95.7%** (157/164) | — |

GSM8K stderr ≈ 2.0 points; HumanEval+ ≈ 1.9 points. **Conclusion: the two quants are equivalent within
noise.** Stronger evidence than the aggregate numbers is *which problems* failed:

- GSM8K: 7 problems fail on both; llama.cpp fails 1 additional problem, vLLM 2. The remaining failures
  are model limits (e.g. 36.36 vs 36, rounding), not a quant effect.
- HumanEval+: 7 problems fail on both; **vLLM's failure set is a strict subset** of
  llama.cpp's, which fails 3 additional problems (38, 116, 124). A 1.8-point gap, within stderr,
  but the direction is consistent: NVFP4 is not worse, if anything slightly better.

The backend comparison table row above ("not yet measured") has been corrected. Unsloth's 92.3%
figure is not comparable to these (a "top-1% accuracy" metric vs BF16, not a task score).

**Two harness flaws found and fixed** (continuing the flaw list from 2026-09-03):

4. lm-eval `flexible-extract` takes the *last* number in the response. This model closes with
   "Kylar needs to pay **$64** for the 16 glasses" → read as 16. Also `$26.00` ≠ `26`. Of the 48
   lm-eval "failures", 37 were correct answers. The `acc_final_para` scorer (last number
   in the last paragraph, bold numbers preferred) fixes it without turning a single correct
   verdict into a wrong one. Both numbers are stored; read the final-paragraph one.
5. llama.cpp + MTP with 4 parallel slots: 96.3% vs 97.3% with 1 slot, and one response
   stopped after 5 characters ("Let $"). Matches the symptoms of upstream #28286 (cross-slot
   contamination with draft-mtp). The official result uses 1 slot; vLLM may use 4 in parallel.

---

## 2026-09-05: Real cache hit ratio — first coding agent session

The number that was previously blank in the Recommendation section. Source:
`runs/cache-ratio-2026-09-05.jsonl` (23 requests, reconstructed from the llama-server log by
`cache-ratio.py`; the `draft_acceptance` column added from the same `print_timing` lines). Load: two
Pi sessions from a laptop over an NVIDIA Sync port-forward — one one-shot "mini spreadsheet" (a
single HTML file, formula parser + dependency graph) then a follow-up session (bug fix + new
feature) on the same file. The model wrote code, ran Node tests, fixed its own tests; 23 requests
total, context grew from 1.6k to 40k tokens.

| | Value |
|---|---|
| Prompt tokens total / cached | 606 854 / 582 177 (**95.9%**) |
| Cold calls (cached < 50%) | **2 / 23 = 8.7%** — below the 12% break-even point |
| Prefill wall total | 66.0 s, 2.87 s per call; **37.1 s** of that in a single call |
| Prefill per call if that call is excluded | ~1.3 s (22 calls, 29 s) |
| Completion tokens total / decode wall | 35 461 / 1076 s → **33.0 tok/s** aggregate |
| Decode per call | 26.6–38.5 tok/s; ≥30k context tends toward 27–30 |
| Aggregate MTP draft acceptance | 0.856 (20 670 / 24 145), 0.78–0.99 per call |

**Conclusion:** the usage pattern of a real coding agent falls in llama.cpp + MTP territory, just as
the Recommendation assumed. Small sample (one user, two sessions, one kind of task), but the
direction is clear: 21 of 23 calls hit ≥ 82% cache, and 17 of those ≥ 98%.

**Those two cold calls are not the ones you would guess.** The first is the initial probe (expected).
The second, task 6875, is the **second turn** of the first session: an 18 364-token prompt, only
1 988 cached, 16 376 tokens re-prefilled over 37 s. The size that got re-prefilled is almost exactly
the size of the previous response (16 342 tokens, 456 s decode — the one-shot response contained a
long thinking block plus the whole file). `[Inference]` Pi sends the content of that response back as
history, but the prefix cache does not match: candidate causes are (a) the generated tokens ≠ the
result of re-tokenizing the same text, so one differing token at the start invalidates everything
after it, or (b) the chat template re-renders the thinking block in a format different from the one
that was generated. Which one is right has not been tested; after that turn, every subsequent turn
hit ≥ 90%, so the cache works normally — this miss only happens once per conversation, right after
the first response. If the pattern is consistent, the cost of a session = one prefill the size of the
first response; for short responses the cost is small, for a 16k-token one-shot like this it is 37 s.

**Narrowed down the same day.** Two sources: a capture of Pi traffic on the laptop (socat) and
`cache-probe.py` on the server.

*Client (socat capture, 2-turn Pi conversation without tools):* Pi **does send `reasoning_content`
back** (28 occurrences in the request body), and its turn 2 hit 99.0% (task 15198). So the "Pi drops
the thinking" hypothesis is REJECTED for ordinary text turns.

*Server (`cache-probe.py`, `usage.prompt_tokens_details.cached_tokens` read directly, turn 2
`max_tokens=1`):* every form of turn that is sent back as-is **HITs**:

| Form of turn 1 | non-streaming | streaming (deltas reassembled, arguments re-serialised) |
|---|---|---|
| thinking + text | HIT | HIT |
| thinking + tool call | HIT | HIT |
| thinking + text + tool call | HIT | HIT |
| ... with a 3.6k-token tool call (HTML file) | HIT | HIT |
| long thinking (puzzle) + tool call | — | HIT |

And the diff of the re-rendered history tokens vs the generated tokens: **identical** for every form.
Only three things break the cache (deterministic diff via `/apply-template`): `reasoning_content`
being dropped or moved to another field (`reasoning` is ignored by the template); the key order of
`arguments` changing; a trailing newline in a parameter value being stripped. The Pi source
(`packages/ai/src/api/openai-completions.ts`) does none of the three: the field follows the signature
it received (`reasoning_content`), `arguments` = `JSON.stringify` of the parsed object (key order
preserved), an empty `content` is dropped (identical render).

**Why one mismatch costs an entire response.** This model is `qwen4exp`: SSM layers
(`ssm.state_size` 128, `full_attention_interval` 4 → 3 of every 4 layers are linear attention). A
recurrent state cannot be rewound to an arbitrary position, so llama-server can only go back to a
**context checkpoint** (`--ctx-checkpoints` 32, `--checkpoint-min-step` 8192; created during prompt
processing, not during generate). A mismatch anywhere in a turn → rollback to the end-of-prompt
checkpoint of the previous turn → re-prefill the entire response. Visible in the probe: the variants
without reasoning always show `cached` = the size of the turn-1 prompt (or an earlier checkpoint),
not the mismatch position. The task 6875 pattern (`cached` 1988 ≈ the 1992-token turn-1 prompt) is
the signature of this rollback; the mismatch itself could be anywhere inside those 16k tokens.

**Status: not yet reproduced from the server side.** The most faithful reproduction (the original
spreadsheet prompt via `write_file`, streaming) hit `max_tokens` 24000 without finishing (682 s
decode), so it is inconclusive. The decisive next step gets executed during a real Pi session: the
server itself can print the tokens around the mismatch. In the Sync launch script:

```bash
LLAMA_SERVER_SLOTS_DEBUG=1 LLAMA_SERVER_SLOTS_N_DIFF=12 API_KEY=<key> ./stack.sh start llamacpp
```

then repeat the Pi task that ends with a large tool call, and look in `runs/serve-current.log` for
the lines `old: ... | ...` / `new: ... | ...` (WARN, shown without verbose) — the tokens before the
`|` match, what follows is the mismatch. The env vars are inherited `stack.sh` → `serve.sh` →
`llama-server`.

`cache-ratio.py` note: the "vLLM territory" 18.5% verdict after this session is polluted by synthetic
probes (3 of 5 cold calls are probes). Without the probes: 3/25 = 12.0%, and two of those are the
first turn of a new conversation, which is always cold anyway. The cold ratio therefore depends on
conversation length, not only on the task pattern.

**Decode note:** 33 tok/s aggregate vs 36.7 tok/s in the benchmark. The gap is consistent with a much
longer context (benchmark ~2k, this session up to 40k) and a lower MTP acceptance on code that has
never existed before (0.78–0.89) compared to re-copying existing code (0.99). The benchmark number is
not wrong; this is the number that represents real usage.

---

## 2026-09-05: Vision (mmproj) on the llama.cpp path — works, MTP unaffected

This model is a VLM (27-layer vision tower, `qwen3vl_merger` projector, `image_size` 768,
`patch_size` 16). Unsloth ships `mmproj-BF16.gguf` (907 542 944 bytes) in the same GGUF repo; this
recipe previously only fetched the text shards + MTP head. Fork b10715 brings `libmtmd`, so
`--mmproj` in `serve.sh` (`--vision`) is enough — no rebuild.

Test 2026-09-05 (`runs/vision-2026-09-05.jsonl`; server `--mtp --nmax 3 -ub 256 --vision`):

| Request | prompt tok | prefill | completion | decode | MTP acceptance | result |
|---|---|---|---|---|---|---|
| 336² PNG with three colour bands, "what colours, top to bottom?" | 196 | 1.39 s | 318 | 42.6 tok/s | 0.977 | correct: red, green, blue |
| 336² PNG with a black 7 on white, "what character?" | 190 | 0.75 s | 51 | 41.5 tok/s | 0.971 | correct: 7 |
| text-only control, same server | 58 | 0.26 s | 87 | 31.5 tok/s | 0.909 | correct |

Notes:
- **MTP stays active on image requests** — in contrast to vLLM (Mia repo): the draft model there does
  not receive multimodal embeddings and falls back to non-speculative decode. In llama.cpp the MTP
  draft only sees the text tokens after the image block, and its acceptance is actually high (0.97)
  because descriptive answers are predictable.
- Memory: `MemAvailable` 29 → 27.9 GiB, about 1 GiB (0.85 GiB projector weights + encode buffers).
- A small image ≈ 140 prompt tokens. llama.cpp warns: "Qwen-VL models require at minimum 1024 image
  tokens to function correctly on grounding tasks; try `--image-min-tokens 1024`". Not made the
  default because it raises prefill per image; relevant only for bounding boxes / precise
  localisation. `[Unverified]` grounding accuracy at the default setting.
- The `LLAMA_SERVER_SLOTS_DEBUG` dump also shows `<|vision_start|> | [mtmd]...` when two different
  image requests land in the same slot — that is an expected mismatch (different images), not a cache
  anomaly.
- Images only. Model card: image + video, no audio. mtmd llama-server accepts still images; video has
  to be sampled into frames by the client. vLLM (Mia repo) accepts `video_url` directly.
- Decision: `stack.sh` turns on `--vision` automatically if `models/mmproj/` exists (`VISION=0`
  disables it). The 1 GiB cost is judged worth a capability that was previously assumed absent.

---

## 2026-09-06: Many-file load — 7 marked bugs in a single Pi session, and the root cause of the cache miss found

The "many-file load" test that has been pending since 2026-09-05. Repo `markedjs/marked` (13 source
files, 410 spec files), a snapshot without git history from commit `c6119b3c` with `test/` from HEAD,
so that **seven bug-fixes** that landed upstream on 4–5 September 2026 (clearly outside the training
data) are ripped out at once. One prompt: fix them all until `npm test` is green, do not touch
`test/`, do not use git history or the network. The `pi/AGENTS.md` rules were active, thinking
`medium`, server 131k + vision. Source of the numbers:
`runs/cache-ratio-2026-09-06-marked.jsonl` (94 requests, from log line 807 onward).

### Model results

**7/7 bugs fixed, 1801 spec + 191 unit tests green, no test file touched, one 31-minute turn**
(28 minutes of that thinking), 94 requests, context grew 2.6k → 70k tokens. The final report uses the
format and the rules labels (`[Inferensi]`, "Diverifikasi / Tidak diverifikasi") in Indonesian —
`AGENTS.md` was obeyed right up to the last turn. Comparison with the upstream fixes:

| Bug | Model fix vs upstream | Note |
|---|---|---|
| HTML tag name (#4083) | identical | both rules (block + inline) exactly the same |
| fence indent (#4074) | identical | same `Math.min(...)` |
| empty code block (#4073) | identical | without the comment |
| nested bracket (#4064) | equivalent | one nesting level inlined, upstream uses a separate sub-rule |
| email autolink (#4063) | different, valid | `(?![a-zA-Z0-9]*[-_])` vs `(?![\w-])`; its backtracking analysis is correct |
| ATX heading tab (#4084) | **passes the tests, latent defect** | the model widened `endingSpaceChar` to `/[ \t]$/`; upstream added a new rule because `endingSpaceChar` is also used by code spans (`Tokenizer.ts:848`), which per CommonMark may only be spaces. No spec catches the space+tab combination |
| character reference autolink (#4053) | different design | the model escapes the text in the **tokenizer** and adds an `&` regex in the renderer; upstream keeps the raw token and escapes in the **renderer** via the `autolink` flag. Passes every spec, but consumers of custom tokens would receive already-escaped text — a reviewer would reject it |

Quality conclusion: 5 fixes on par with upstream, 2 that pass the tests but lose on design, one of
them a latent bug. For a local 4-bit model with no help, this is the level of a contributor whose PR
needs one review round, not one that needs rewriting.

### Cache and speed

| | Value |
|---|---|
| Requests / total prompt tokens | 94 / 3 123 134 |
| Cached | **99.1%**; cold calls **0/94** |
| Prefill wall | 109 s total, 1.16 s per call; longest 9.3 s (4.2k-token tool result) |
| Completion tokens / decode | 45 900 tok / 1700 s → **27.0 tok/s** aggregate at 30–70k context |
| MTP draft acceptance | 0.861 |
| Final context | 70 043 tokens, no compaction (threshold 106k) |

Decode 27 tok/s vs 33 in the spreadsheet session (≤40k) vs 36.7 in the benchmark (~2k): the drop is
consistent with context length. cache-ratio verdict: llama.cpp + MTP territory, by a wide margin.

### Root cause of the turn-2 cache miss (yesterday's task 6875) — ANSWERED: token boundary drift

The `LLAMA_SERVER_SLOTS_DEBUG` trap caught **2 real mismatches out of 93 requests** in the session
(2.2%), plus 1 at the start of the session that is expected (the cwd in the system prompt changed
`marked` → `marked-eval`). Both are the same pattern: **identical text, different token ids**.

- task 6211: inside the tool call command `sed -n '/^const atx/,/$/p' src/rules.ts | head -12`, the
  split is around `/,` + ` /$/p'`. Rollback of 225 tokens, 0.7 s.
- task 14580: inside reasoning containing a regex, ``...or end. ` `` + `@` — the cache has a `` ` ``
  token then `@...`, the new prompt has a `` `@ `` token (ids `... 13151 75370 ...` vs
  `... 74988 5431 ...`). Rollback to the checkpoint = **the entire previous response (2 597 tokens) +
  the tool result**, 2 667 tokens, 7.4 s.

So hypothesis (a) is the right one, not (b) the template and not (c) the client: **the generated
tokens are not the canonical tokenization of their text**. The client (Pi) sends the text back, the
server re-tokenizes canonically, and at the drift position the token ids differ even though the text
is the same. In this hybrid model a mismatch that small rewinds to the previous end-of-prompt
checkpoint, so the cost = the length of the previous response: 0.7 s for a 200-token response, 7 s
for 2.6k, **37 s for yesterday's 16k-token one-shot response**. Frequency ~2% of requests, tending to
occur on text dense with punctuation (regex, shell, backticks). All the synthetic `cache-probe.py`
tests HIT because their output is ordinary prose and code.

`[Speculation]` The source of the drift is probably MTP: the draft head proposes tokens, and the
accepted sequence need not match what the tokenizer would produce for the same text. The test: a
similar session with `--spec-type` disabled and count the mismatches — not run yet because decode
without MTP is far slower. Mitigation available today: none that is cheap; the cost is tied to the
length of the previous response, so thinking `medium` (shorter responses) also shrinks the cost of a
miss. The real fix is in the server: comparing the cache by text rather than by token id, or periodic
checkpoints during generate. Worth reporting upstream with the dump above.

---

## 2026-09-06: Idea from the Mia repo — reduced-vocabulary drafting (not possible in llama.cpp yet)

The MiaAI-Lab repo (vLLM) raised single-stream decode 36.9 → 46.3 tok/s (+25%) with one change: the
MTP draft head computes argmax over only the 65 536 most frequently used tokens instead of the whole
248 320 vocab (FR-Spec style, `files/patch_mtp_draft_vocab.py`). The reasoning: the drafter's
`lm_head` is read once per draft step, three times per engine step at MTP 3, and decode is already
against the bandwidth wall, so the bytes saved turn into time almost one-to-one. Output accuracy does
not change because the target verifies every draft; a wrong draft only lowers acceptance (MGSM, 250
problems per language: EN 94.8% vs 93.6%, ZH 86.4% vs 86.4%, their numbers, single-run).

Relevance to this recipe `[Inference]`: llama.cpp's `draft-mtp` path also computes full logits over
248 320 tokens for each draft token. `[estimate]` a Q8_0 head = 248 320 × 2 560 ≈ 0.67 GB per draft,
~2 GB per step at `--nmax 3`, compared to ~6.35 GB per target token from the roofline (273 GB/s ÷ 43
tok/s) — about a quarter of the bytes per step. If it could be trimmed as in vLLM, decode could
potentially reach the 45 tok/s range. **There is no llama-server flag for this**; it needs a code
change in the speculative path (draft argmax over a vocab subset). Recorded as an idea, not a plan:
nothing in the recipe can change today, and an upstream contribution from me is impossible (see the
llama.cpp AI policy). If upstream ever adds it, that is the first flag worth testing.

Two other things from their update that do not apply here: PLE page-fault prefetch (`posix_fadvise`)
— our PLE is resident, not mmap; and `CUDAGRAPH_CAPTURE_SIZES=auto` — vLLM-specific.

---

## 2026-09-06: Option A — speculative flag sweep: `--spec-draft-p-min 0.50` +10% decode

The only speculative knob that had never been swept. `serve.sh` now accepts `--pmin` (default 0.75,
the value that was previously hardcoded) and `stack.sh` passes `NMAX`/`PMIN` through from the
environment. All numbers from `runs/bench-stream2.jsonl`, labels `sweepA-*` and `sweepA2-*`, server
restarted per configuration (empty cache), 3 requests per start, warm-up discarded.

**Round 1, five configurations, 2 valid runs each:**

| nmax / p-min | median decode | vs baseline |
|---|---|---|
| 3 / 0.75 (today's baseline) | 38.1 tok/s | — |
| 4 / 0.75 | 39.1 | +2.5%, noise |
| **3 / 0.50** | **41.7** | **+9.4%** |
| 2 / 0.75 | 36.8 | −3.4%, noise |
| 3 / 0.90 | 33.8 | invalid — the failing prompt differs from the baseline |

**Round 2, alternating A/B with 3 restarts per side** (labels `sweepA2-*`):

| | p-min 0.75 | p-min 0.50 |
|---|---|---|
| valid runs | 5 | 4 |
| prompt `llama-kv-cache.cpp`, 3 restarts | 34.0 / 36.0 / 35.0 → **35.0** | 39.0 / 39.1 / 38.1 → **38.8** (+10.7%) |
| prompt `arg.cpp` | 41.5 / 41.2 | 46.2 (+11.7%, 1 run) |
| median TTFT | 28.25 s | 28.25 s |
| draft acceptance | 0.931 | 0.840 |
| mean accepted length | 3.21 | 3.30 |

The mechanism is coherent: a lower p-min makes the drafter propose more tokens per step (approaching
`nmax` 3 every time), more of them get rejected (acceptance drops), but the number of accepted tokens
per step actually rises (3.21 → 3.30) and that is what determines tok/s. Output quality is untouched:
the target verifies every draft, so p-min only tunes cost, not the result. On the same prompt, three
consecutive restarts, the ranges do not overlap (34.0–36.0 vs 38.1–39.1). `nmax` 4 does not help; the
remaining gain from p-min 0.50 is about +10%, not the +25% of reduced-vocab drafting in vLLM.

**Decision (2026-09-06, after option C):** `p-min 0.50` becomes the default in `stack.sh`/`serve.sh`;
`PMIN=0.75` restores the old value.

**The "prompt `speculative.cpp` → 1 token" anomaly is ANSWERED (an old item in the tracker).** In all
11 starts of this sweep, the first request (warm-up, prompt `speculative.cpp`, 10 892 tokens) ended
after 1 token, and the 3rd run (same prompt, the RAM cache restoring the state → only 4 tokens
prefilled) also gave 1 token. The `arg.cpp` prompt does the same thing intermittently (3 of 6
starts). `bench-stream.py` uses raw `/v1/completions` without a chat template, and on those prompts
the model immediately emits an end token. This is a harness artefact, not a server bug, and it does
not depend on the speculative parameters. The consequence: each start yields only 1–2 valid runs out
of 3. Harness flaw #6 to fix if this benchmark is used again: use chat completions or add prompts.

---

## 2026-09-06: Greedy is not deterministic — and in llama.cpp the cause is MTP

Trigger: issue #28 in the Mia repo (an independent reproduction on another Spark): on their vLLM,
five identical `temperature=0` requests produced five different outputs, **even with MTP off**, and
on a "continue the story" prompt the model sometimes reads the context back verbatim, which inflates
MTP acceptance (0.93 on recital vs 0.37–0.41 on honest generation) and with it their prose decode
numbers.

Tested on our server, two prompts (code and prose), five identical requests per prompt, `max_tokens`
220, greedy, all requests landing in the same slot (slot 3, prefix-cache hit of 4 tokens) so slot
rotation is not a variable:

| | MTP 3 (installed profile) | MTP off (`MTP=0 ./stack.sh start llamacpp`) |
|---|---|---|
| code, differing outputs | **5/5**, diverging after ~275 characters | **1/5** — byte-for-byte identical |
| prose, differing outputs | **3/5**, diverging after ~236 characters | **1/5** |

So it differs from vLLM: **llama.cpp's non-speculative path is deterministic**, and the
non-determinism comes from speculative decoding. `[Inference]` During verification the target
processes 1+n tokens in one batch, not a single token; a different batch shape takes a different
kernel path / reduction order, and on near-tied logits the argmax can differ. Since the draft length
per step depends on `p-min` and on the content, the batch shape keeps changing, and one differing
token changes everything that follows. This is also a consistent explanation for the token boundary
drift (the 2026-09-06 section above): the draft proposes a non-canonical token split, and the target
accepts it on a tie.

Practical consequences:
- Output stays equally good on average (each token is the target's argmax for that batch), but it is
  **not bit-reproducible** while MTP is active. Our accuracy eval (2026-09-04) was run with MTP on,
  so the 97.3% / 93.9% numbers have a small run-to-run variance that has not been measured; for evals
  that must be reproducible, use `MTP=0`.
- Decode benchmarks on prompts that invite quoting (e.g. "explain this file") raise acceptance. Our
  bench number of 0.93 vs 0.86 in a real coding session is consistent with that effect; the real
  session number is the representative one.
- `stack.sh` now accepts `MTP=0` for diagnostics like this; decode without MTP is ~24 tok/s.

Two other things from the Mia issue that do not apply here: the 3 200-token QSA attention block that
makes prefix-cache hits impossible below ~6 400 prompt tokens — in llama.cpp our probe hits on a
100-token prompt; and verbatim recital, which we have not observed in a coding session (acceptance
0.84–0.86, not 0.93).

---

## 2026-09-06: Option C done — own build with a patch, crash gone, speed on par with prebuilt

The question answered: can the "Mia way" (runtime patch + own build) be used in llama.cpp to
eliminate the `MUL_MAT_ID` crash without `-ub 256`, and at what price. The answer: yes, a one-line
patch, and once the source composition is right, without losing speed. But getting there taught a few
things about the Unsloth prebuilt binaries that are not visible from the outside.

### What was tried, in chronological order (`runs/bench-stream2.jsonl`, labels in the first column; crash probe =
raw tokens of size 367/512 at `-ub 512`; server restarted per row)

| Label | Source | Compiler | Crash 367 | TTFT 10.9k | Decode |
|---|---|---|---|---|---|
| `llamacpp-mtp-ub256` | prebuilt b10715 (installed profile) | nvcc 13.3 (CI) | avoided via ub 256 | 26.2 s | 36.7 |
| `llamacpp-mtp-ub512` | prebuilt b10715 | — | **crash** | 21.9 s | 37.2 |
| `b10798-ub256` | prebuilt b10798 (2026-09-04) | — | crash at ub 512 | 25.8 s | 39.3 |
| `mmqfix-ub512` | branch `mtp/qwen4exp-nextn` + patch | nvcc 13.0 | **no crash**, sweep 1–600 clean | 33.6 s | 33.1 |
| `mmqfix-cub-ub512/256` | same + CUB 3.2 | nvcc 13.0 | no crash | 36.1 / 38.7 s | 33.6 / 35.1 |
| `mmqfix-n133-ub512/256` | same + CUB 3.2 | nvcc 13.3 (local redist) | no crash | 36.0 / 41.5 s | 34.8 / 34.2 |
| **`mixfix-ub256`** | **b10715 mix reassembled + patch** | nvcc 13.0 | — | **26.1 s** | **38.0** |
| **`mixfix-ub512`** | same | nvcc 13.0 | **no crash** | **23.4 s** | **36.2** |

### Three things learned

1. **The Unsloth "mix" release tag is not a source tree.** `git clone --branch b10715-mix-86bd2d3`
   gives a manifest commit without `qwen4exp`. The prebuilt binaries are assembled by CI from
   upstream b10715 + 14 PR commits pinned in `scripts/unsloth/pr-set.json`, merged in order, add/add
   conflicts resolved by their `additive_merge.py`. `patches/compose-mix.py` repeats that process
   locally; the result is a commit with a different hash but the same content, and all 14 merges are
   clean (two via add/add resolution).
2. **The fork's MTP branch (`mtp/qwen4exp-nextn`) is not an equivalent substitute.** A build from
   that branch fixes the crash but prefill is 30–50% slower and decode −10%, and that is **not** the
   compiler: nvcc 13.0 vs 13.3, CUB 3.0 vs 3.2, and CPU backend configurations were all tried with no
   effect. Nor is it the upstream base (prebuilt b10798 is newer and is in fact the fastest). The
   difference is in the source composition: that branch carries MTP commits different from the
   repinned #144 in the mix (among others "key the CUDA graph cache by shape" and the *borrow target
   tensors* path from PR #142, which is absent in the mix). `[Inference]` exactly which one is slow
   has not been isolated; no need, since the reassembled mix is already on par with the prebuilt.
3. **Memory harness.** An `nvcc -j8` build alongside a 91 GiB server turned out to be safe on the
   system (no `NV_ERR_NO_MEMORY`), but a memory guard in the terminal session killed the build process several
   times; the practical solution is to run the build detached (`setsid nohup`) with the server
   stopped.

### Results and artefacts

- `patches/mmq-ids-tail-padding.patch`: one line in `ggml/src/ggml-cuda/mmq.cu`, identical to
  upstream PR #27044, for the `mul_mat_id` path only (the dense path line is left alone).
- `patches/compose-mix.py` + `build-fork.sh`: full upstream clone, assemble the mix per the tag
  manifest, apply the patch, build (`GGML_CUDA_CUB_3DOT2=ON`, native arch), result in
  `forks/unsloth-mixfix/`. The script refuses if the patch no longer applies, so an upstream fix will
  be visible.
- Usage: `FORK=unsloth-mixfix UBATCH=512 ./stack.sh start llamacpp`. All other features (MTP, vision,
  `--pmin`) are the same because `serve.sh` only swaps the binary directory.
- Measured benefit vs the installed profile: cold TTFT 26.2 → 23.4 s (−11%) on a 10.9k prompt, decode
  on par, and the main one: **no ubatch size crashes any more**. Sweep 1–600 at `-ub 512`:
  **600/600 sizes pass, 0 `CUDA error` lines in the log**, and a re-bench after the sweep of
  23.5 s / 36.0 tok/s.
- The installed profile is **unchanged**: prebuilt + `-ub 256` remains the default because it needs
  no build. This is an option for those who want it, and it becomes unnecessary once #27044 is merged
  upstream and reaches the Unsloth prebuilds.

### 2026-09-06 (continued): b10798 base + patch, and a CUDA graphs test

The next two items from the follow-up optimisation list, done with the same pipeline
(`TAG=b10798-mix-659e406 ./build-fork.sh`; `runs/bench-stream2.jsonl` labels `b10798fix-*`,
`runs/crash-sweep-2026-09-04.jsonl` b10798 entries):

| | prebuilt b10715 + `-ub 256` (installed) | b10715 mix + patch, `-ub 512` | **b10798 mix + patch, `-ub 512`** |
|---|---|---|---|
| Ubatch sweep 1–600 | crashes without ub 256 | 600/600 | **600/600, 0 CUDA errors** |
| TTFT 10.9k | 26.2 s | 23.4 s | **21.1 s (−20%)** |
| Decode | 36.7 tok/s | 36.2 | **39.0 (+6%)** |

The decode gain comes from the 83 upstream commits between b10715 and b10798 — MoE fusion extended
to speculative decoding, fused weighted expert reduction, flash attention K/V tile swizzle, per-slice
qwen4exp indexer — not from the patch. The same as measured on the b10798 prebuilt (39.3 tok/s at ub
256), only now without the crash. This is the best build available right now; `build-fork.sh` uses
this tag as its default.

**CUDA graphs are not hidden overhead.** A/B on the same build, ub 256:
`GGML_CUDA_DISABLE_GRAPHS=1` → 37.5 vs 38.9 tok/s, about −4%, same TTFT. So graphs are indeed active
in the MTP verification path and their contribution is small; the per-step overhead that remains is
elsewhere (PLE gather on the CPU, QSA indexer, draft synchronisation). The "graph cache" patch
candidate is crossed off. What is left to reach 40+ at short context: `PMIN=0.50` (+10% measured) on
top of this build `[estimate]` ~43 tok/s; for long context: Unsloth PRs #150/#165 (QSA decode) have
not been tried.

Note: b10798 contains upstream #28123 "qwen4exp: support recurrent state rollback". `[Inference]`
this could cut the cost of the token-drift cache miss (the rollback to the previous end-of-prompt
checkpoint); not yet re-measured with `cache-probe.py`.

## 2026-09-06: Unsloth PRs #150/#165 (QSA decode at long context) — tested, ported, not adopted

Two open PRs in the Unsloth fork target the long-context decode drop that we measured ourselves
(39 tok/s at short prompts → 27 tok/s at 30–70k in the marked session). Their claims on the discrete
cards: #150 computes the QSA layout input once per ubatch instead of 48× per layer (decode at 131k
16.2 → 21.1 tok/s on 2× RTX 3090); #165 replaces masked attention over the whole cache with a gather
of 2 048 selected cells (141k: 7.1 → 18–22 tok/s on 2× A6000), plus CUDA top-k batching, a kv-cells
scan, and a compact mask. Both target the fork's `qwen4exp/qwen3.8-flash-next` branch, not the
upstream tree used by the mix.

### Findings before building

1. **The idea behind #150 is already in b10798** in upstream form: #27941 (2026-09-01) shares one QSA
   input set across all layers (`qsa_inps` in `qwen4exp.cpp`, with the comment "the layers sharing a
   ratio share one input set") and its host scan is 865 µs per ubatch at 33k according to a TODO in
   `llama-memory-hybrid-idx.cpp`. The prebuilt b10715 (installed) does not contain it yet. Merging
   the PR into the mix fails in 11 files (parallel histories: the mix pins vs the fork branch), and
   it is unnecessary.
2. **Upstream sparse flash attention (#27970) does not apply to this model yet.** `qwen4exp.cpp` has
   a commented-out line `build_attn_mha(..., top_k->ne[0], ...)` with the comment "TODO: enable
   sparse attention when we are ready", but the CUDA kernel exists only for head dim 512/576
   (`may_use_sparse` in `fattn-mma-f16.cuh`, i.e. DeepSeek V4). Enabling that line changes nothing
   here.
3. **#165 can be ported, with one semantic adjustment.** Four files apply cleanly via a 3-way apply
   (`qwen4exp.cpp`, `llama-graph.cpp`, `models.h`, `top-k.cu`); the `kv-cells.h` hunk is obsolete
   (its function has been replaced upstream); the mix's `set_input_qsa` was rewritten entirely, so
   `mask_row` was added by hand. The PR's per-block top-k shortcut is DISABLED: `blk_cells` in the
   mix only contains fully populated groups, and the not-yet-full tail cells sit in one spare block
   whose row is zero, so expanding block → cell would discard the tail and repeatedly attend to cell
   0. The per-cell gather path (the core of the saving) is kept. Result:
   `patches/unsloth-pr165-qsa-gather.diff` (398 lines) on top of the `build-fork.sh` b10798 tree +
   crash patch; clean build.

### Results (`runs/longctx-2026-09-06.jsonl`, `bench-longctx.py`, 256 output tokens, 2 different prompts per point)

| decode tok/s | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| installed b10715 `-ub 256`, MTP | 34.7 / 43.6 | 44.4 / 29.2 | 40.6 / 37.8 | 33.2 / 26.0 |
| b10798+patch `-ub 512`, MTP | 38.0 / 37.2 | 37.1 / 34.3 | 37.0 / 39.4 | 32.5 / 30.1 |
| #165 port, gather ON, MTP | 45.0 / 37.4 | 32.2 / 36.2 | 38.2 / 37.5 | 28.8 / 33.3 |
| #165 port, gather OFF, MTP | 45.1 / 34.4 | 35.3 / 38.9 | 39.9 / 38.0 | 32.5 / 34.1 |
| #165 port, gather ON, **MTP=0** | 24.7 / 24.8 | 23.9 / 24.1 | **22.7 / 22.5** | **19.5 / 19.6** |
| #165 port, gather OFF, **MTP=0** | 24.9 / 24.9 | 23.6 / 23.6 | 21.1 / 21.3 | 16.9 / 16.8 |

Prefill was recorded too: 359 tok/s at 64k on the installed build, 458–500 on the `-ub 512` build
(the effect of the crash patch, not the gather; the #165 top-k batching is not visible in the prefill
numbers).

**The gather works, but only without MTP.** With `MTP=0` the gain is +7% at 32k and +15% at 64k, zero
at 8k (the `n_kv >= 2×width` gate is not passed yet). Attention over the whole cache is indeed the
part that grows: without the gather decode drops 24.9 → 16.9 from 8k to 64k, with the gather
24.7 → 19.5, so the gather recovers about a third of that loss, not 2.5× as on the A6000 —
`[Inference]` on the GB10 unified memory makes the 18 MB/token mask and the host loop far cheaper, so
there is less left to save. With MTP on, the gather ON and OFF numbers are identical within noise
(the spread of draft acceptance 0.66–0.96 shifts decode by ±8 tok/s), and the reason is in the code:
the gather path only activates if the ubatch contains one token per stream
(`gather = n_tokens == n_stream && ...`), whereas MTP verification sends 1 + draft tokens at once. So
on the installed profile #165 practically never runs.

**The output stays correct, within numerical limits.** Exactly the same prompt, greedy, `MTP=0`
(`bench-longctx.py --fixed`, labels `qsa-fixed-*`; `tmp/div-*.json` for the logprobs): 8k, 32k and
48k are byte-identical gather ON vs OFF over 128–160 tokens. 64k differs starting at token 8 on a
near-tied choice (`run` −0.64 vs `in` −0.77 nats on one side, reversed on the other); the per-token
logprob difference before that point is at most 0.18 nats. That is the pattern of a different
numerical path — the gather dequantizes q8_0 K/V to F32 via `get_rows` and then to F16 for flash
attention, while the masked path reads q8_0 directly — not the pattern of missing cells or a wrong
mask, which would show up far earlier and at 32k as well. Consistent with the PR's claim
(byte-identical on their card, where both paths use the same types).

### Multi-token extension (built in the same session, at the user's request)

The PR's gate was replaced: the gather is active for ubatches of up to 8 tokens per stream
(`QWEN4EXP_QSA_GATHER_MAX_TPS`), i.e. ordinary decode and the MTP verification batch (1 + draft). K/V
is gathered per token through a single flattened index list `[n_topk*n_tps]` and then reshaped to
`[hd, n_head_kv, n_topk, n_tps*ns]`; the mask goes through `get_rows` over `mask_row` viewed as
`[1, n_kv, n_tps, ns]`; Q splits on its own into `[hd, n_head, 1, n_tps*ns]` because `build_attn_mha`
splits Q according to `k->ne[3]`. Prompt chunks of hundreds of tokens still take the masked path.

**Numerically correct.** Prompts were built at 31 748 and 63 492 tokens (≡ 4 mod 512) so that the
last prefill chunk contains 4 tokens and goes through the multi-token gather path; `MTP=0`, greedy,
logprobs (`tmp/div-mt-*.json`): 160-token output byte-identical gather ON vs OFF at both sizes, with
a first-token logprob difference of 0.000 and 0.006 nats.

**The effect under MTP, measured per verification step** (tok/s depends too much on acceptance;
`predicted_ms / (predicted_n − draft_n_accepted)` = ms per step, `runs/longctx-2026-09-06.jsonl`):

| ms per step (2 prompts) | 8k | 16k | 32k | 64k |
|---|---|---|---|---|
| control: b10798+patch, port gather OFF, 1-token port (6 runs) | 69–77 | 73–77 | 85–89 | 101–105 |
| extension, no threshold (`qsa2-gather1-mtp`) | 78.0 / 73.8 | 82.4 / 74.2 | **77.3 / 75.6** | **92.4 / 86.9** |
| extension + 24k threshold, gather ON (`qsa3-gather1-mtp`) | 74.5 / 69.9 | 71.4 / 78.0 | 77.3 / 83.8 | **86.5 / 92.6** |
| same binary, `QWEN4EXP_QSA_GATHER=0` (`qsa3-gather0-mtp`) | 68.4 / 74.8 | 79.0 / 76.2 | 85.1 / 81.4 | 96.5 / 101.1 |

Without a threshold, gathering 4 tokens × 2 304 rows per layer makes 8k–16k 3–5% slower, while
32k–64k are 12–13% faster; so the gate got a threshold of `n_kv ≥ 24 576`
(`QWEN4EXP_QSA_GATHER_MIN_KV`), between the two points. With that threshold, A/B on the same binary:
64k **−9%** per step (89.5 vs 98.8 ms), 32k −3%, 16k −4% and 8k +1% (both on the masked path, so that
is the size of the noise, ±4%). Combining the two runs: 64k is 9–13% shorter per step ≈ +10–15%
decode at the same acceptance; 32k somewhere between 3 and 12%, uncertain; below 24k zero by design.
This is consistent with the `MTP=0` gains (+7% at 32k, +15% at 64k).

**The 100k point** (99 960-token prompt, same binary, labels `qsa3-*` targeting 100000): the gain
keeps growing with context depth.

| 100k | gather ON | gather OFF |
|---|---|---|
| MTP, decode tok/s (2 prompts) | **32.6 / 35.5** | 27.8 / 27.0 |
| MTP, ms per verification step | **104.8 / 104.5** | 124.5 / 124.6 |
| `MTP=0`, decode tok/s | **16.6** | 13.7 |
| prefill tok/s | 412–429 | 412–429 |

−16% per step, +21–25% decode. With the gather, decode at 100k equals decode at 64k without it.

**Accuracy at that depth is on par** (`runs/accuracy-2026-09-06.jsonl`, `bench-accuracy-longctx.py`):
the ordinary eval does not touch this patch at all because the GSM8K/HumanEval+ prompts are only
1–2k tokens, below the threshold. So the first 100 GSM8K problems (prompt `gsm8k_cot_zeroshot`,
thinking off, greedy, seed 1234, same as `bench-accuracy.py`) were given a 30 000-token prefix of the
same source code so that the gather path is active for every answer (the prefix is served by the
prompt cache: on average 29 337 tokens cached, 753 re-prefilled per problem). Result: gather ON
**98/100**, gather OFF **97/100**, with the two wrong answers being the same in both (#93, #98) and
OFF getting one more wrong (#12). 43/100 responses byte-identical; the rest differ because greedy +
MTP simply is not deterministic (the 2026-09-06 section "Greedy is not deterministic"). Without a
prefix, the old build scored 96.3% over 300 problems (2026-09-04), so a 30k prefix breaks nothing.
HumanEval+ was not repeated: evalplus builds its own prompts and cannot be given a prefix.

### Decision

Not made the recipe default (user decision, 2026-09-06): it needs an own build, its patch is pinned
to the b10798 mix and only we maintain it, and the gain only shows above 24k. But for long sessions
this is the RECOMMENDED profile: `QSA_GATHER=1 ./build-fork.sh` applies
`patches/unsloth-pr165-qsa-gather.diff` (the #165 port + multi-token extension + threshold) on top of
the b10798 mix + crash patch, then `FORK=unsloth-qsa UBATCH=512 ./stack.sh start llamacpp`; the
prebuilt remains the default for a first installation. For coding sessions that live at 30–70k
(marked: 27 tok/s) `[estimate]` +10% at the upper end, +25% if it reaches 100k. The other road still
waits for upstream to enable sparse FA for this model's head dim (the TODO in `qwen4exp.cpp`).
`bench-longctx.py` and `bench-accuracy-longctx.py` remain for re-measuring.
