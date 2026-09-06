# Qwen3.8-Flash-Next on DGX Spark (GB10)

A measured recipe for serving **Qwen3.8-Flash-Next** (MoE 125B / 6B active + 51B PLE)
on a single NVIDIA DGX Spark GB10 with 128 GB unified memory, and for pointing a
coding agent at it.

Two backends were benchmarked head to head. **llama.cpp + MTP wins** for this
workload; vLLM is kept as an option. Numbers, method and the paths that failed are
in [`OPTIMIZATION.md`](OPTIMIZATION.md).

| Backend | TTFT (10.9k prompt) | Decode | RAM | Start |
|---|---|---|---|---|
| **llama.cpp fork + MTP** ⭐ | 21.6s / **1.8s** cached | **37.8 tok/s** | 88 GiB | 45s |
| vLLM NVFP4 | 11.3s / 1.9s cached | 30.3 tok/s | 112 GiB | 14 min |

vLLM is 1.44x faster only on cold prompts; llama.cpp wins on cache hits and costs
24 GiB less. Break-even is 12% cold calls.

### The shipped profile, measured

What `./stack.sh start llamacpp` runs today, and what it measured on this machine. Every
number links to the file it came from; none is typed from memory.

| | Value | Source |
|---|---|---|
| Config | Unsloth fork `b10715`, UD-Q4_K_XL, MTP head Q8_0 `--nmax 3` `--pmin 0.50`, KV q8_0, context 131k, `-ub 256`, vision projector | `stack.sh`, `serve.sh` |
| TTFT, 10.9k-token prompt | 26.2 s cold, 1.8 s on a prefix-cache hit | `runs/bench-stream2.jsonl` |
| Decode, benchmark | 36.7 tok/s at p-min 0.75 (median of 2, `-ub 256`); p-min 0.50, the default since 2026-09-06, measured +10% on matched prompts | `runs/bench-stream2.jsonl` |
| Decode, real coding sessions | 33.0 tok/s at ≤40k context; 27.0 tok/s at 30–70k | `runs/cache-ratio-2026-09-05.jsonl`, `runs/cache-ratio-2026-09-06-marked.jsonl` |
| Decode vs depth, same build | 43/35 tok/s at 8k, 33/26 at 64k (MTP); the optional `QSA_GATHER=1` build below holds 33–36 at 100k | `runs/longctx-2026-09-06.jsonl` |
| Prefix-cache hit, real sessions | 95.9% (23 requests); 99.1%, 0 cold (94 requests) | same two files |
| MTP draft acceptance, real sessions | 0.86 | same two files |
| Accuracy, thinking off, greedy | GSM8K 97.3%, HumanEval+ 93.9% | `runs/accuracy-2026-09-04.jsonl` |
| Images | two test images answered correctly; MTP stays active, 42 tok/s | `runs/vision-2026-09-05.jsonl` |
| Memory | 88 GiB resident + ~1 GiB for vision, ~28 GiB left | `stack.sh status` |
| Start to ready | ~45 s | `stack.sh` |

The cold TTFT is 4 s worse than the 21.6 s in the table above: that is the price of
`-ub 256`, the workaround for the cuBLAS crash described below. Cached TTFT and decode are
unchanged by it. One known cost not in the table: about 2% of requests miss the prefix
cache because the generated tokens are not the canonical tokenisation of their text, and
on this hybrid model such a miss re-prefills the whole previous response — 7 s after a
2.6k-token answer, 37 s after a 16k one (`OPTIMIZATION.md`, 2026-09-06).

### `p-min 0.50`: the speculative threshold, and why it is the default

The MTP drafter proposes up to 3 tokens per step and stops early when its confidence for
the next token drops below `--spec-draft-p-min`. The recipe shipped 0.75 until
2026-09-06; it now ships 0.50, measured as follows (`runs/bench-stream2.jsonl`, labels
`sweepA2-*`, three fresh-cache restarts per side, identical prompts):

| | p-min 0.75 | p-min 0.50 (default) |
|---|---|---|
| decode, same code prompt, 3 restarts | 34.0 / 36.0 / 35.0 tok/s | 39.0 / 39.1 / 38.1 tok/s |
| draft acceptance | 0.93 | 0.84 |
| accepted draft tokens per step | 3.21 | 3.30 |
| TTFT, RAM | unchanged | unchanged |

**What it does not change:** the answer. The target model verifies every drafted token and
keeps only what it would have produced itself, so p-min sets how much draft work is
wasted, not what comes out. **What it costs:** more rejected drafts, so the gain shrinks on
text the drafter predicts badly. Coding-agent traffic is easy for it — real sessions here
run at 0.84–0.91 acceptance even at 0.75, far above the ~0.4 where a lower p-min would
start to lose. `PMIN=0.75 ./stack.sh start llamacpp` restores the old setting; `nmax 4`
was also tried (+2.5%, within noise, not adopted). With MTP on, greedy output is not
bit-reproducible at either setting (`OPTIMIZATION.md`, 2026-09-06); `MTP=0` restores exact
reproducibility at ~24 tok/s.

### Optional: build it yourself to drop `-ub 256`

The prefill crash that forces `-ub 256` is upstream issue ggml-org/llama.cpp#27792: the
MMQ `mul_mat_id` path under-sizes a buffer, and whether the out-of-bounds read is visible
depends on the allocator, hence only some ubatch sizes die. Its fix (PR #27044) is one
line and unmerged as of 2026-09-06, so no prebuilt carries it yet. `build-fork.sh`
reproduces the Unsloth prebuilt's exact source composition (upstream tag + the PR set the
fork pins) on this machine, applies the line, and builds it:

```bash
./build-fork.sh                                   # ~15 min, ~1.3 GB of source clones
FORK=unsloth-mixfix UBATCH=512 ./stack.sh start llamacpp
```

Measured 2026-09-06 on the default tag (`b10798-mix-659e406`, `runs/bench-stream2.jsonl`
labels `b10798fix-*`): cold TTFT on the 10.9k prompt 26.2 → 21.1 s, decode 36.7 → 39.0
tok/s (the newer upstream base adds MoE and attention fusions), and the 1..600 ubatch
sweep that crashes the prebuilt at 367 and 512 runs clean. Everything else (MTP, images,
`PMIN`) is identical because only the binary directory changes. `TAG=b10715-mix-86bd2d3`
rebuilds the shipped prebuilt's base instead (23.4 s / 36.2 tok/s, also clean).

Two warnings from getting there, in `OPTIMIZATION.md`: the fork's release tags are
manifests, not source trees, so cloning a tag and building it does not give you the
prebuilt; and building the fork's MTP *branch* instead fixes the crash but is 30–50%
slower at prefill for reasons that are not the compiler. The shipped profile stays on
the prebuilt with `-ub 256` because it needs no build; this path becomes moot once
#27044 lands upstream and in a prebuilt.

Long context is where decode really drops (39 → 27 tok/s over a 70k coding session),
and two open Unsloth PRs target exactly that: #150 (QSA inputs once per ubatch) and #165
(attend only over the top-k cells instead of masking the whole cache). Measured
2026-09-06 (`runs/longctx-2026-09-06.jsonl`, `bench-longctx.py`): #150's idea is already
in the b10798 base in upstream form. #165 as written only engages for single-token
ubatches, so it never runs under MTP verification; `patches/unsloth-pr165-qsa-gather.diff`
is a port of it onto this tree extended to the 1 + draft verification batch, gated to
contexts above 24k cells where it pays off. Greedy output stays byte-identical, and the
gain grows with depth: the verification step gets 9–13% shorter at 64k and 16% at 100k,
where decode goes 27 → 33–36 tok/s; about 3% at 32k, nothing below. GSM8K behind a 30k
prefix (`bench-accuracy-longctx.py`, 100 items): 98 with the patch, 97 without, same two
misses. Recommended for long sessions; the prebuilt stays the default for a first install:

```bash
QSA_GATHER=1 ./build-fork.sh                      # ~15 min, on top of the crash fix above
FORK=unsloth-qsa UBATCH=512 ./stack.sh start llamacpp
```

Not the default because it needs a build, the patch is pinned to the b10798 mix and
maintained only here, and it only helps deep into a session; the measurements and the
port's caveats are in `OPTIMIZATION.md`.

## Quick start

First-time setup (downloads ~110 GB) is in [`SETUP.md`](SETUP.md). Once that is done:

```bash
./stack.sh start llamacpp   # port 18080, OpenAI-compatible, ~45s
./stack.sh status
./stack.sh stop
```

`stack.sh` refuses to start when another backend is up, when Ollama holds a model,
or when memory is short. Do not bypass it: the three of them share 121.7 GiB of
unified memory, and overcommitting it caused an `NVRM NV_ERR_NO_MEMORY` stall and a
watchdog kernel panic on 2026-09-01.

It also starts llama-server with `-ub 256`. The CUDA backend (Unsloth fork and
upstream alike) dies on certain prefill batch sizes — one HumanEval prompt reproduced it
every time — and capping the physical batch at 256 avoided every case we could reproduce.
The error first surfaced in `cublasGemmEx` and was reported as ggml-org/llama.cpp#28377,
but `CUDA_LAUNCH_BLOCKING=1` later pinned the real culprit: the MMQ `MUL_MAT_ID` kernel,
upstream issue #27792, whose one-line fix (#27044) is still unmerged. Until it lands, the
workaround costs ~4 s of cold TTFT; the optional build above removes the crash without
that cost. Details and the sweeps in `OPTIMIZATION.md`.

## Images

The model is a vision-language model and the llama.cpp path serves images too: the
0.85 GiB vision projector (`mmproj-BF16.gguf`, 27-layer tower, `qwen3vl_merger`) loads
next to the text weights. `./download-mmproj.sh` fetches it, and `stack.sh` enables it
automatically whenever the file is present (`VISION=0` for text-only). Send images the
OpenAI way, as `image_url` parts with an `http(s)://` URL or a `data:` URI.

Measured 2026-09-05 (`runs/vision-2026-09-05.jsonl`), 336x336 PNGs:

| Test | Result |
|---|---|
| Three colour stripes, "which colours, top to bottom?" | correct; 196 prompt tokens, 8.9 s wall incl. 318 tokens of reasoning + answer |
| A drawn digit, "which character?" | correct; 190 prompt tokens, 2.1 s wall |
| MTP on image requests | still active, draft acceptance 0.97, decode 42 tok/s |
| Extra memory | ~1 GiB (`MemAvailable` 29 → 27.9 GiB) |

Two notes. A small image costs ~140 prompt tokens; llama.cpp warns that Qwen-VL grounding
tasks (bounding boxes, precise localisation) want at least 1024 image tokens — add
`--image-min-tokens 1024` to `serve.sh` if you do that kind of work, at the price of a
longer prefill per image. And this is images only: the model card lists image and video,
not audio, and llama-server's multimodal layer takes still images, so video means
sampling frames yourself. Pi gets `"input": ["text", "image"]` in `pi-models.json`.

## Accuracy of the two quants

Measured on this machine, same harness and decoding settings on both backends, thinking
off, greedy (`runs/accuracy-2026-09-04.jsonl`, details in `OPTIMIZATION.md`):

| | llama.cpp UD-Q4_K_XL | vLLM NVFP4 |
|---|---|---|
| GSM8K, 300 items | 97.3% | 97.0% |
| HumanEval+ pass@1 (plus tests) | 93.9% | 95.7% |

Equivalent within noise; the problems that fail are mostly the same ones on both. BF16 does
not fit on the GB10, so this is a relative measurement, not an absolute retention figure.

## Using it from a coding agent

The server speaks the OpenAI API, so any harness that accepts a custom base URL
works. Verified against the running server: `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, SSE streaming and tool calling.

```
base URL : http://127.0.0.1:18080/v1      # vLLM: 18300
API key  : anything (not checked)
model    : qwen3.8-flash-next
```

### Reaching the server from another machine

The server binds to loopback by default, so nothing outside the box can see it.
`HOST=tailscale` binds it to this node's tailnet address instead:

```bash
HOST=tailscale API_KEY=<your-key> ./stack.sh start llamacpp
```

That is reachable from every device on the tailnet and from nowhere else.
Verified on 2026-09-04 with the server bound this way:

| From | Result |
|---|---|
| tailnet IP / MagicDNS name, correct key | 200 |
| no key, or a wrong key | **401** |
| `127.0.0.1` | connection refused |
| the machine's own LAN IP | connection refused |

The listening socket is the tailnet address only — not the WiFi subnet, not the
docker or kube bridges. When both machines sit on the same WiFi, Tailscale routes
peer-to-peer over the LAN automatically, so being at home costs nothing in latency
and needs no second configuration.

`API_KEY` is optional but recommended once the port leaves loopback; the server is
unauthenticated without it.

**If your laptop runs NVIDIA Sync instead of the Tailscale app**, `HOST=tailscale`
will not work for that laptop. Sync bundles its own Tailscale node inside the
application (it shows up in the tailnet as `nvsync-<machine>` with a `-dev` build
number), and that node is private to Sync: the laptop's OS gets no `100.x`
interface, so `ping 100.x.y.z` times out and no other program can reach the tailnet
address. What Sync does provide is per-application port forwarding through that
node — the same mechanism that puts the DGX Dashboard on `localhost:11000`. Use it
for the model server too (verified 2026-09-04):

1. In Sync, connect to the DGX, then **Settings → Custom → Add New**.
2. Name it anything, set the port to `18080` (Sync binds local 18080 to remote 18080).
3. Launch script:

   ```bash
   cd <path-to-this-repo> && API_KEY=<your-key> ./stack.sh start llamacpp
   ```

   Leave `HOST` unset. Sync's forward targets `127.0.0.1` on the DGX, so the default
   loopback bind is exactly right — and the server never listens on WiFi or the
   tailnet at all. `stack.sh start` returns once the server is up (~45 s) and leaves
   it running in the background.

   Alternatively, keep the model server out of Sync's hands: start it yourself over
   SSH and give the Custom Application a launch script that only reports status, so a
   Sync reconnect never triggers a start (`stack.sh` refuses to start twice, but the
   entry would still show a failed script):

   ```bash
   # on the DGX, once
   LLAMA_SERVER_SLOTS_DEBUG=1 LLAMA_SERVER_SLOTS_N_DIFF=12 API_KEY=<your-key> ./stack.sh start llamacpp
   # Sync launch script
   cd <path-to-this-repo> && ./stack.sh status
   ```

   The two `LLAMA_SERVER_SLOTS_*` variables are recommended either way. They cost
   nothing while the prefix cache hits, and when a turn misses the cache the server
   logs the tokens around the mismatch (`old: ... | ...` / `new: ... | ...` in
   `runs/serve-current.log`). On this hybrid model a miss anywhere in a turn re-prefills
   the whole previous response, so that one line is the evidence you want when a
   follow-up turn suddenly takes 30 s (see `OPTIMIZATION.md`, 2026-09-05).
4. **Click the new entry** in the Custom list. Adding it only saves the definition;
   the forward is not active until the entry is clicked and its status dot turns
   green. Until then `127.0.0.1:18080` on the laptop refuses connections even though
   the server is up on the DGX.
5. On the laptop the base URL is `http://127.0.0.1:18080/v1`.

Two simpler alternatives when both machines are on the same WiFi: bind to the DGX's
LAN address directly (`HOST=192.168.x.y API_KEY=<key> ./stack.sh start llamacpp`;
DHCP leases move, so reserve the address in the router), or install the real
Tailscale app on the laptop alongside Sync — it becomes a second node and
`HOST=tailscale` works as described above.

For local benchmarking leave `HOST` unset — it defaults to `127.0.0.1`, which is
what `bench-code.sh` and `bench.sh` expect.

### Pi (https://pi.dev)

Pi is a minimal agent harness that takes custom providers from a JSON file.
[`pi-models.json`](pi-models.json) in this directory is ready to use — every field
in it was set by probing the running server, not copied from documentation.

**1. Install Node 22.19+ and Pi.** Neither Node nor npm is on this machine yet, so
install Node first (nvm, apt, or your preference), then:

```bash
npm install -g @earendil-works/pi-coding-agent   # provides the `pi` binary
```

**2. Install the provider config.** It defines two providers, `qwen38-llamacpp`
(port 18080) and `qwen38-vllm` (port 18300):

```bash
mkdir -p ~/.pi/agent
cp pi-models.json ~/.pi/agent/models.json    # merge by hand if you already have one
```

**3. Start the model on the DGX, then Pi on the laptop:**

```bash
# on the DGX
HOST=tailscale API_KEY=<your-key> ./stack.sh start llamacpp

# on the laptop
export QWEN38_API_KEY=<the same key>
pi
```

`pi-models.json` ships with a placeholder host and reads the key from
`QWEN38_API_KEY`, so neither your tailnet address nor the secret lives in the file.
Export the variable in the same shell you start `pi` from — `/model` re-reads the
file, but not the environment.

If you go through NVIDIA Sync's port forward instead (see *Reaching the server from
another machine*), the base URL on the laptop is simply `http://127.0.0.1:18080/v1`
and you can skip the address lookup below. Otherwise find your own address on the DGX:

```bash
tailscale ip -4                                   # 100.x.y.z
tailscale status --json | jq -r .Self.DNSName     # host.tailnet-name.ts.net
```

Put either one in `baseUrl` in your local copy of `~/.pi/agent/models.json`. The
MagicDNS name needs MagicDNS enabled in the tailnet's DNS settings; the raw
100.x address always works. Your tailnet name identifies your account, so keep it
out of anything you publish.

**4. Pick the model** with `/model` (or `Ctrl+L`) and choose
*Qwen3.8-Flash-Next (llama.cpp fork + MTP, GB10)*. The file is re-read every time
you open `/model`, so edits apply without restarting.

Notes:

- `apiKey` is `"$QWEN38_API_KEY"`, which is Pi's syntax for reading an environment
  variable (`$VAR`, `${VAR}`, or `!command`). An earlier revision of this file used
  `{env:QWEN38_API_KEY}`; Pi does not know that form and sends the placeholder
  literally, which the server answers with 401. If you run the server without
  `API_KEY`, set the variable to any non-empty string anyway: Pi hides models that
  have no key configured.
- This is a **reasoning model**: it returns `reasoning_content` alongside `content`,
  which is why the config sets `"thinkingFormat": "deepseek"`. Give it a generous
  `maxTokens` — with a small budget the thinking consumes it all and `content` comes
  back empty. To turn thinking off entirely, add `--reasoning off` in `serve.sh`.
- `contextWindow` is 131072 to match the server's `-c`. Raise both together
  (`CTX=262144 ./stack.sh start llamacpp`) if you need the model's full native
  context; that costs about 6 GiB more.
- Responses report `usage.prompt_tokens_details.cached_tokens`, so you can measure
  your real prefix-cache hit rate — the number that decides whether llama.cpp or
  vLLM suits your usage. `./cache-ratio.py -v` computes the same thing server-side
  from `runs/serve-current.log` after a real session, so no client changes are needed.
  Measured on real sessions: 95.9% cached over 23 requests (2026-09-05), and 99.1% cached,
  0 cold calls, over a 94-request session fixing seven bugs in `marked` with context
  growing to 70k (2026-09-06; decode 27 tok/s at that length). About 2% of requests
  miss because the generated tokens are not the canonical tokenisation of their text, and
  on this hybrid model such a miss re-prefills the whole previous response. Details in
  `OPTIMIZATION.md`.

### Optional: the maintainer's Pi rules and settings

[`pi/`](pi/) holds the global `AGENTS.md` and `settings.json` I use with this model:
rules that keep a 131k context from filling up (read files with `grep`/`sed`, `tail` the
logs), a session tracker for continuity across sessions, thinking level `medium` by
default, and a larger compaction reserve. Personal preferences, not part of the recipe —
copy what you like, ignore the rest. [`pi/README.md`](pi/README.md) explains each value.

## Documents

| File | Contents |
|---|---|
| [`SETUP.md`](SETUP.md) | Reproduce from a fresh clone: downloads, builds, operational rules |
| [`OPTIMIZATION.md`](OPTIMIZATION.md) | Benchmarks, the roofline analysis, what failed, three benchmark defects found |
| [`FINDINGS.md`](FINDINGS.md) | Initial survey of candidate quants and stacks |
| [`RESULTS.md`](RESULTS.md) | First benchmark run (superseded; carries a correction banner) |

## Scripts

| Script | Purpose |
|---|---|
| `stack.sh` | start / stop / status for either backend, with a memory preflight |
| `serve.sh` | launches llama-server directly (`--mtp`, `--vision`, `--ngram-mod`, `--mmap`, `CTX=`, `BUILD=`, `UBATCH=`) |
| `download-parts.sh` | main GGUF, 104 GiB |
| `download-mtp.sh` | MTP draft head, 3.85 GiB |
| `download-mmproj.sh` | vision projector, 0.85 GiB — enables images |
| `download-fork.sh` | Unsloth llama.cpp prebuilt — required for MTP (`TAG=` for another release) |
| `build-fork.sh` | rebuild that prebuilt's exact composition from source with the one-line MMQ crash fix; `QSA_GATHER=1` adds the long-context gather patch (`patches/`) |
| `bench-code.sh` | coding-shaped benchmark against llama.cpp |
| `bench-stream.py` | streaming benchmark; the only one valid for comparing across backends |
| `bench-longctx.py` | decode speed against context depth (8k–64k) from llama-server's own timings; llama.cpp A/B only, `--fixed` for byte-identity checks |
| `cache-ratio.py` | prefix-cache hit rate of real usage, reconstructed from the llama-server log |
| `cache-probe.py` | does resending a turn hit the prefix cache? Generates each turn shape (text, tool call, streaming...), resends it as a client would, reads `cached_tokens`, and diffs the re-rendered history against the generated tokens |
| `bench-accuracy.py` | GSM8K + HumanEval+ through the API, same settings on either backend |
| `bench-accuracy-longctx.py` | GSM8K behind a fixed long prefix, for attention changes that only engage deep in the context |

## Credits

- [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
  — the vLLM recipe for the same model on the same hardware. Its "patch the runtime and
  build it yourself" approach is what `build-fork.sh` copies to llama.cpp, its
  reduced-vocabulary drafting is the decode idea noted in `OPTIMIZATION.md`, and the
  greedy-non-determinism report in its issue tracker prompted the same test here.
- [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) — the fork, its prebuilts,
  the MTP draft head and the GGUF quants this recipe runs on; `patches/compose-mix.py`
  reproduces its release composition with its own `additive_merge.py`.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the server itself, and
  PR #27044 for the one-line fix carried in `patches/`.
- [pi.dev](https://pi.dev), [evalplus](https://github.com/evalplus/evalplus) and
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the agent
  harness and the accuracy harnesses used for the measurements.
