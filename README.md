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
upstream alike) aborts in `cublasGemmEx` on certain prefill batches — one HumanEval
prompt reproduced it every time — and capping the physical batch at 256 avoided every
case we could reproduce. Details and the sweep in `OPTIMIZATION.md`.

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
Find your own address on the DGX:

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

- `apiKey` is the dummy string `"local"` on purpose. Pi hides models that have no
  auth configured, so a keyless local server still needs a placeholder.
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
| `serve.sh` | launches llama-server directly (`--mtp`, `--ngram-mod`, `--mmap`, `CTX=`, `BUILD=`, `UBATCH=`) |
| `download-parts.sh` | main GGUF, 104 GiB |
| `download-mtp.sh` | MTP draft head, 3.85 GiB |
| `download-fork.sh` | Unsloth llama.cpp prebuilt — required for MTP |
| `bench-code.sh` | coding-shaped benchmark against llama.cpp |
| `bench-stream.py` | streaming benchmark; the only one valid for comparing across backends |
| `cache-ratio.py` | prefix-cache hit rate of real usage, reconstructed from the llama-server log |
| `bench-accuracy.py` | GSM8K + HumanEval+ through the API, same settings on either backend |
