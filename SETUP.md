# Setup from zero (fresh clone)

This repo holds no model weights, no llama.cpp build, no fork binaries and no
third-party clones — all of that is `.gitignore`d because it is large and can be
fetched again. This document is what makes a clone on another machine runnable.

**Target hardware:** NVIDIA DGX Spark GB10, 128 GB unified memory, aarch64, CUDA 13.
The benchmark numbers in [`OPTIMIZATION.md`](OPTIMIZATION.md) apply to that
combination; other hardware will differ.

## Prerequisites

```bash
aria2c --version        # every download (resumable, size-verified)
jq --version            # bench.sh / bench-code.sh
python3 --version       # 3.12 here; the bench-*.py scripts use the stdlib only
git --version           # benchmark prompts are read from a llama.cpp checkout
nvidia-smi              # driver; tested on 580.173.02
```

Only for the optional paths:

```bash
nvcc --version          # build-fork.sh (CUDA toolkit 13.0 here)
uv --version            # the accuracy eval venv
docker --version        # the vLLM path
docker run --rm --gpus all <image> nvidia-smi -L    # GPU visible inside a container
```

## Path A — llama.cpp + MTP (recommended, ~110 GB)

This is the stack that won the benchmark: decode 36.7 tok/s, 88 GiB resident,
~45 s to ready.

```bash
# 1. Main model, 103.7 GiB in 4 parts (~3 h at 11 MB/s)
./download-parts.sh

# 2. MTP draft head, 3.85 GiB  -- speculative decoding
./download-mtp.sh

# 3. Unsloth prebuilt, ~189 MB download / 223 MB extracted -- REQUIRED for MTP
./download-fork.sh

# 4. (optional) vision projector, 0.85 GiB -- makes image_url requests work.
#    stack.sh loads it automatically once the file exists; VISION=0 for text-only.
./download-mmproj.sh

# 5. Run it
./stack.sh start llamacpp        # port 18080, OpenAI-compatible API
./stack.sh status
./stack.sh stop
./stack.sh help                  # flags (e.g. --clock-cap, needs sudo) and env vars
```

Every download script is re-runnable: each file is verified by exact byte size and
resumed with `aria2c -c`, so an interrupted transfer costs nothing.
`./download-mtp.sh --both` also fetches the Q4_K_M head (66.1% draft acceptance for
Q8_0 vs 64.4% for Q4_K_M, so Q8_0 is the default).

### What the default profile is

`./stack.sh start llamacpp` calls `serve.sh` with the settings that were measured,
not guessed:

| | Value | Set by |
|---|---|---|
| Binary | Unsloth prebuilt `b10715-mix-86bd2d3` (`forks/unsloth-b10715`) | `download-fork.sh`, `serve.sh` (`BUILD=fork`) |
| Weights | UD-Q4_K_XL, experts on GPU, `per_layer_token_embd.weight` pinned to CPU | `serve.sh` |
| Speculative | MTP draft head Q8_0, `--nmax 3`, `--pmin 0.50` | `stack.sh` (`NMAX=`, `PMIN=`, `MTP=0`) |
| KV cache | q8_0 for K and V, flash attention on | `serve.sh` |
| Context | 131072 | `serve.sh` (`CTX=`) |
| Physical batch | `-ub 256` | `stack.sh` (`UBATCH=`) |
| Vision | projector loaded whenever `models/mmproj/mmproj-BF16.gguf` exists | `stack.sh` (`VISION=0`) |
| Model id | `qwen3.8-flash-next` (via `--alias`) | `serve.sh` (`ALIAS=`) |
| Bind address | `127.0.0.1:18080`, no auth | `serve.sh` (`HOST=`, `API_KEY=`) |

`p-min 0.50` has been the default since 2026-09-06: +10% decode over 0.75 on matched
prompts, with the same answers, because the target model verifies every drafted token.
`PMIN=0.75` restores the old threshold. `MTP=0` turns speculative decoding off
entirely — decode drops to ~24 tok/s, and it is the only way to get bit-reproducible
greedy output (`OPTIMIZATION.md`, 2026-09-06).

### Why the fork and not upstream llama.cpp

Upstream rejects the Unsloth MTP head: `output_hc_norm.weight` is not in the head file
and upstream requires the complete `qwen4exp` tensor set. Without MTP, decode falls
from ~37 tok/s to ~24. Details in `OPTIMIZATION.md`.

`TAG=<release tag> ./download-fork.sh` fetches another Unsloth prebuilt side by side
(it lands in `forks/unsloth-<upstream tag>`, e.g. `TAG=b10798-mix-659e406` →
`forks/unsloth-b10798`), selected at run time with
`FORK=unsloth-b10798 ./stack.sh start llamacpp`.

### Optional: build the fork yourself and drop `-ub 256`

The prefill crash that forces `-ub 256` is upstream issue ggml-org/llama.cpp#27792
(the MMQ `mul_mat_id` path under-sizes a buffer). Its one-line fix, PR #27044, is
unmerged as of 2026-09-06, so no prebuilt carries it. `build-fork.sh` reproduces the
Unsloth prebuilt's exact source composition (upstream tag + the PR set pinned in the
fork's `scripts/unsloth/pr-set.json`, replayed by `patches/compose-mix.py`), applies
the patch and builds:

```bash
./build-fork.sh                                   # ~15 min, ~1.3 GB of source clones
FORK=unsloth-mixfix UBATCH=512 ./stack.sh start llamacpp
```

Needs `nvcc`; stop the server first if memory is tight, and use `JOBS=` to change the
`-j8` default. The default `TAG=b10798-mix-659e406` measured cold TTFT 26.2 → 21.1 s
and decode 36.7 → 39.0 tok/s on the 10.9k-token prompt, and the 1..600 ubatch sweep
that crashes the prebuilt at 367 and 512 runs clean (`runs/bench-stream2.jsonl`,
labels `b10798fix-*`). `TAG=b10715-mix-86bd2d3` rebuilds the shipped prebuilt's base
instead. Everything else (MTP, vision, `PMIN`) is unchanged, because only the binary
directory moves.

For long sessions there is a second, additive build:

```bash
QSA_GATHER=1 ./build-fork.sh                      # ~15 min, on top of the crash fix
FORK=unsloth-qsa UBATCH=512 ./stack.sh start llamacpp
```

This adds `patches/unsloth-pr165-qsa-gather.diff` — a port of Unsloth PR #165
(attend over the top-k cells instead of masking the whole cache), extended to MTP
verification batches and gated to contexts above 24k cells. Measured 2026-09-06
(`runs/longctx-2026-09-06.jsonl`): nothing below 24k by design, ~3% at 32k, 9–13%
shorter verification step at 64k, 16% at 100k where decode goes 27 → 33–36 tok/s;
GSM8K behind a 30k prefix scores 98/100 with the patch and 97/100 without
(`runs/accuracy-2026-09-06.jsonl`). Recommended for long sessions, not the default:
it needs a build, and the patch is pinned to the b10798 mix and maintained only here.

### Optional: build upstream llama.cpp for comparison

Only needed to compare against upstream (which means no MTP):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -B llama.cpp/build-new -S llama.cpp \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF
cmake --build llama.cpp/build-new -j8        # ~4 min on GB10
UBATCH=256 BUILD=build-new ./serve.sh        # no --mtp; upstream cannot load the head
```

`stack.sh start llamacpp` always uses `BUILD=fork`, so an upstream build is driven
through `serve.sh` directly. `CMAKE_CUDA_ARCHITECTURES=native` matters: it compiles
sm_121 only instead of a multi-architecture fatbin, which is what makes the build 4
minutes rather than 40.

## Path B — vLLM NVFP4 (optional, ~150 GB)

1.44x faster on cold calls only; it loses on prefix-cache hits and takes 112 GiB,
leaving ~8 GiB. Full comparison in `OPTIMIZATION.md`.

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git vllm-dgx
cd vllm-dgx && docker build -t qwen38-flash-dgx . && cd ..     # base image ~23 GB
# an HF token is optional but avoids the anonymous rate limit:
#   printf '%s' 'hf_xxx' > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
vllm-dgx/scripts/download-weights.sh                           # 126 GB, ~3 h
./stack.sh start vllm                                          # port 18300, ~14 min to load
```

## Operating rules (important)

The 121.7 GiB of unified memory is shared by CPU and GPU. **Never run two backends at
once, or one backend next to an Ollama that is holding a model** — that is what caused
`NVRM NV_ERR_NO_MEMORY` → stall → watchdog kernel panic on 2026-09-01. `stack.sh`
refuses to start when it detects either condition, or when less memory is available
than the backend needs (95 GiB for llama.cpp, 118 GiB for vLLM). Do not bypass it.

```bash
./stack.sh status        # check before anything
```

### `-ub 256`

`stack.sh` starts llama-server with `-ub 256`. Without it the CUDA backend (fork and
upstream alike) aborts on certain prefill batch sizes — one HumanEval prompt reproduced
it every time, and a sweep found 367 and 512 also crash. The error first surfaced in
`cublasGemmEx` and was reported as ggml-org/llama.cpp#28377, but `CUDA_LAUNCH_BLOCKING=1`
later pinned the real culprit: the MMQ `MUL_MAT_ID` kernel, upstream issue #27792, whose
one-line fix (#27044) is still unmerged. The workaround costs ~4 s of cold TTFT; the
`build-fork.sh` path above removes the crash without that cost. If you call `serve.sh`
directly, set `UBATCH=256` yourself — `serve.sh` leaves the batch size at llama.cpp's
default otherwise.

### Server slots

llama-server's `--parallel` defaults to auto, which picks **4 slots with a unified KV
pool** here (`n_slots = 4, kv_unified = 'true'` in `runs/serve-current.log`), so each
slot can still see the whole context. That is fine for normal use, but with `draft-mtp`
active the slots leak into each other: upstream issue ggml-org/llama.cpp#28286
(cross-slot content contamination, still open). It cost a full point of GSM8K here —
96.3% on 4 slots against 97.3% on one, with one answer truncated after five characters
(`runs/accuracy-2026-09-04.jsonl`). Accuracy evals therefore run `--concurrent 1`, and
a single-slot queue can be forced with:

```bash
LLAMA_ARG_N_PARALLEL=1 ./stack.sh start llamacpp
```

### Context

The default is `CTX=131072`, not the model's 262144 native: the q8_0 KV cache costs
~6.4 GiB at 128k against ~12.8 GiB at 262k, and 32k turned out too tight for a coding
agent pasting real files. Running at the native context is an optional tip, not part of
the recipe:

```bash
CTX=262144 ./stack.sh start llamacpp
```

Measured here at that setting (`runs/longctx-2026-09-06.jsonl`, label `ctx262k-check`),
a 140k-token prompt sat at 99–100 GiB used and still decoded at 33.5 tok/s. If you raise it, raise the client's window too — for Pi that is
`contextWindow` in `~/.pi/agent/models.json`; the two numbers have to move together.

## Re-running the benchmarks

Benchmark prompts are read from real `llama.cpp` sources so the prompt is identical
across runs and machines. `llama.cpp/` is gitignored, so a fresh clone needs it first
(no build required):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

```bash
./bench-code.sh <label> 3                    # via llama.cpp's /completion
python3 bench-stream.py <port> <label> 3     # streaming; works against both backends
HOST=100.x.y.z API_KEY=<key> python3 bench-stream.py 18080 remote 3   # from another machine
./bench-longctx.py <label>                   # decode vs context depth, llama.cpp A/B only
./cache-ratio.py -v                          # real cache-hit ratio from runs/serve-current.log
tmp/eval-venv/bin/python bench-accuracy.py 18080 <label> --concurrent 1   # accuracy, see below
```

Every script `cd`s into its own directory first, so they can be called from anywhere.
`HOST` is passed straight to the client — unlike `serve.sh` and `stack.sh` it is not
resolved, so give the benchmarks a literal address, not `tailscale`.

Three rules, learned the hard way (the three methodology defects are written up in
`OPTIMIZATION.md`):

- **A distinct prompt per run.** Repeating a prompt at temperature 0 inflated results
  by 14.6%, and by 263% with speculative decoding on.
- **Token counts from `usage.completion_tokens`,** never from counting SSE chunks —
  vLLM packs several tokens into one chunk, which understated its decode rate by 2.6x.
- **Cross-backend comparisons only through `bench-stream.py`.** `bench.sh`,
  `bench-code.sh` and `bench-longctx.py` read llama.cpp's own server-side `timings`,
  which are not defined the same way as vLLM's. Discard warm-up runs, drop failed ones.

`cache-ratio.py` answers the question `OPTIMIZATION.md` leaves open: what share of real
calls is cold. It reads the llama-server log rather than a benchmark, reconstructs the
cached tokens per request from `prompt eval` against `n_tokens` at release, and compares
the result with the 12% break-even point between the two backends. Run it after a real
coding session, not after a benchmark — benchmark prompts differ by design, so they are
always cold.

### Accuracy (`bench-accuracy.py`)

Needs a throwaway venv under `tmp/` (untracked):

```bash
uv venv tmp/eval-venv --python 3.12
VIRTUAL_ENV=$PWD/tmp/eval-venv uv pip install "lm_eval[api]" evalplus
```

That pulls in `datasets`, which `bench-accuracy-longctx.py` needs too. Datasets (GSM8K,
HumanEval+) are downloaded into `tmp/hf-cache` and `tmp/evalplus-cache` rather than
`~/.cache/huggingface`, which the vLLM container owns as root. Run with `--concurrent 1`
on llama.cpp + MTP (upstream #28286, above); `--concurrent 4` is safe on vLLM. GSM8K 300
items takes ~38 min and HumanEval+ 164 items ~18 min per backend
(`runs/accuracy-2026-09-04.jsonl`). Read `acc_final_para`, not
`acc_flexible_extract` — the reason is in `OPTIMIZATION.md`.

`bench-accuracy-longctx.py` runs the same GSM8K items behind a fixed long prefix, for
attention changes such as the QSA gather patch that only engage deep in the context:

```bash
tmp/eval-venv/bin/python bench-accuracy-longctx.py <label> --n 100 --prefix 30000
```

## Connecting a coding agent / harness

Both backends expose an **OpenAI-compatible API**, so any harness that accepts a custom
base URL works. Verified against the running llama.cpp fork: `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, SSE streaming, and tool calling
(it returns `tool_calls` with `finish_reason: "tool_calls"`).

```
OPENAI_BASE_URL = http://127.0.0.1:18080/v1     # vLLM: 18300
OPENAI_API_KEY  = anything (not checked unless API_KEY was set)
model           = qwen3.8-flash-next
```

**This is a reasoning model.** Answers come back in two fields: `reasoning_content`
(the thinking) and `content` (the final answer). A harness that only reads `content`
works normally, it just does not see the thinking. To turn it off, or to save tokens,
add `--reasoning off` or `--no-reasoning-preserve` to the `serve.sh` command line.

### Reaching the server from another machine

The server binds to loopback by default. `HOST=tailscale` resolves this node's tailnet
address and binds there instead:

```bash
HOST=tailscale API_KEY=<your-key> ./stack.sh start llamacpp
```

That is reachable from every device on the tailnet and from nowhere else — not the WiFi
subnet, not the docker or kube bridges. `API_KEY` is optional but recommended as soon as
the port leaves loopback; without it the server is unauthenticated.

**If the laptop runs NVIDIA Sync instead of the Tailscale app,** `HOST=tailscale` cannot
be reached from it: Sync embeds its own Tailscale node inside the application, so the
laptop's OS gets no `100.x` interface. Use Sync's per-application port forward instead
(verified 2026-09-04):

1. In Sync, connect to the DGX, then **Settings → Custom → Add New**.
2. Set the port to `18080` (Sync binds local 18080 to remote 18080).
3. Launch script — leave `HOST` unset, because Sync's forward targets `127.0.0.1` on
   the DGX and the default loopback bind is exactly right:

   ```bash
   cd <path-to-this-repo> && API_KEY=<your-key> ./stack.sh start llamacpp
   ```

   Alternatively, start the server yourself over SSH and give the Custom Application a
   launch script that only reports status, so a Sync reconnect never triggers a start:

   ```bash
   # on the DGX, once
   LLAMA_SERVER_SLOTS_DEBUG=1 LLAMA_SERVER_SLOTS_N_DIFF=12 API_KEY=<your-key> ./stack.sh start llamacpp
   # Sync launch script
   cd <path-to-this-repo> && ./stack.sh status
   ```

   The two `LLAMA_SERVER_SLOTS_*` variables are worth setting either way: they cost
   nothing while the prefix cache hits, and on a miss the server logs the tokens around
   the mismatch into `runs/serve-current.log`.
4. **Click the new entry** in the Custom list. Adding it only saves the definition; the
   forward is not active until the entry is clicked and its status dot turns green.
   Until then `127.0.0.1:18080` on the laptop refuses connections even though the server
   is up on the DGX.
5. On the laptop the base URL is then `http://127.0.0.1:18080/v1`.

More detail, including the same-WiFi alternatives, in README, *Reaching the server from
another machine*.

### Pi (https://pi.dev) — ready-made config

[`pi-models.json`](pi-models.json) in this directory defines two providers
(`qwen38-llamacpp` on 18080, `qwen38-vllm` on 18300). Copy or merge it into
`~/.pi/agent/models.json`, then pick the model with `/model`:

```bash
mkdir -p ~/.pi/agent
cp pi-models.json ~/.pi/agent/models.json      # CAREFUL: do not overwrite an existing config
```

Edit `baseUrl` — it ships with a placeholder host so no tailnet address is published.
Through the Sync port forward it is simply `http://127.0.0.1:18080/v1`; on the tailnet,
find your own address on the DGX with `tailscale ip -4` or
`tailscale status --json | jq -r .Self.DNSName`.

`apiKey` is `"$QWEN38_API_KEY"` — Pi's syntax for reading an environment variable
(`$VAR`, `${VAR}` or `!command`). The `{env:VAR}` form used by an earlier revision is
**not** understood by Pi: it sends the placeholder literally and the server answers 401.
Export the variable in the same shell you start `pi` from. If the server runs without
`API_KEY`, set the variable to any non-empty string anyway — Pi hides models that have
no key configured.

Every field in the file was set by probing a running server (2026-09-04), not copied
from documentation:

| Probed | Result | Consequence in models.json |
|---|---|---|
| role `developer` | HTTP 200 | `supportsDeveloperRole` left at its default (true) |
| `reasoning_effort` | HTTP 200 | `supportsReasoningEffort` left at its default (true) |
| `max_tokens` / `max_completion_tokens` | both 200 | `maxTokensField` need not be set |
| `stream_options.include_usage` | `usage` is sent | `supportsUsageInStreaming` true |
| thinking shape | `reasoning_content` | `"thinkingFormat": "deepseek"` |
| tool calling | `tool_calls` + `finish_reason` | works with no adjustment |
| images | answered correctly | `"input": ["text", "image"]` on the llama.cpp provider |

Note that llama.cpp *accepts* `reasoning_effort` without an error but probably ignores
it. For Pi that is enough — what matters is that the request is not rejected.

`contextWindow` is 131072 to match the server's `-c`; raise both together if you run
with `CTX=262144`. Give the model a generous `maxTokens`: with a small budget the
thinking consumes all of it and `content` comes back empty.

A bonus from the probe: `usage.prompt_tokens_details.cached_tokens` is reported, so a
client can watch directly how much of each prompt hit the prefix cache — the number that
decides whether llama.cpp or vLLM suits your usage (break-even at 12% cold calls, see
`OPTIMIZATION.md`). `./cache-ratio.py -v` computes the same thing server-side.

[`pi/`](pi/) holds the optional global `AGENTS.md` and `settings.json` the maintainer
runs with this model; [`pi/README.md`](pi/README.md) explains each value. Personal
preferences, not part of the recipe.
