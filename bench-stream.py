#!/usr/bin/env python3
"""Client-side streaming benchmark that works against both llama.cpp and vLLM.

Both servers expose OpenAI-compatible streaming endpoints, so timing is measured
the same way for both: TTFT is the wall time until the first token arrives, decode
rate is the remaining tokens over the remaining time. Server-reported `timings`
differ between the two and cannot be compared.

Since 2026-11-12 the prompt goes through /v1/chat/completions (thinking off) and
every request of a run uses a different source file. The raw /v1/completions form
used before made the model answer some prompts with an immediate end token, so a
start yielded two valid prompts out of five, and requests 4-6 repeated 1-3 and hit
the prompt cache. API=completions restores the old request; records carry "api",
and numbers from the two forms are not comparable with each other.
"""
import json, os, sys, time, urllib.request, statistics as st

# Run from the script's own directory so the relative prompt sources and the
# runs/ output path resolve no matter where this is invoked from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT, LABEL, RUNS = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
HOST = os.environ.get("HOST", "127.0.0.1")          # match serve.sh / stack.sh
API_KEY = os.environ.get("API_KEY", "")
HDRS = {"Content-Type": "application/json"}
if API_KEY:
    HDRS["Authorization"] = "Bearer " + API_KEY

API = os.environ.get("API", "chat")                 # chat | completions (pre-2026-11-12 form)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))   # longer outputs average over more draft steps
# Seven sources: the warm-up plus six runs never repeat a prompt (a repeat hits the prompt cache).
SRCS = ["llama.cpp/common/speculative.cpp",
        "llama.cpp/src/llama-kv-cache.cpp",
        "llama.cpp/common/arg.cpp",
        "llama.cpp/src/llama-sampler.cpp",
        "llama.cpp/src/llama-context.cpp",
        "llama.cpp/src/llama-graph.cpp",
        "llama.cpp/src/llama-model.cpp"]
if API == "completions": SRCS = SRCS[:3]
TASKS = ["Explain what this file does, then write a unit test in C++ for the most complex function in it.",
         "Review this code for correctness bugs and write a patch in diff format for anything you find.",
         "Write a Python script that parses the command-line flags defined in this file and emits them as JSON."]

def model_id():
    """vLLM 404s on an unknown model name; llama.cpp ignores the field. Ask the
    server which model it serves instead of guessing."""
    try:
        req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/models", headers=HDRS)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:
        print(f"!! could not read /v1/models ({e}), falling back to 'default'", file=sys.stderr)
        return "default"

MODEL = model_id()


def run(i):
    k = len(SRCS)
    src, task = SRCS[i % k], TASKS[(i + (i // k if k > 3 else 0)) % 3]
    if not os.path.exists(src):
        sys.exit(f"missing prompt source: {src}\n"
                 "These benchmarks read real llama.cpp sources so the prompt stays\n"
                 "identical across runs and machines. Clone it first (see SETUP.md):\n"
                 "  git clone https://github.com/ggml-org/llama.cpp.git")
    ctx = open(src, encoding="utf-8", errors="replace").read()[:40000]
    prompt = f"{task}\n\n```cpp\n{ctx}\n```\n"
    # include_usage makes the server report completion_tokens, so we never have
    # to infer token count from the number of SSE chunks -- vLLM can pack several
    # tokens into one chunk, which silently understated its decode rate by 2.6x.
    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if API == "chat":
        body.update({"messages": [{"role": "user", "content": prompt}],
                     "chat_template_kwargs": {"enable_thinking": False}})
        url = f"http://{HOST}:{PORT}/v1/chat/completions"
    else:
        body["prompt"] = prompt
        url = f"http://{HOST}:{PORT}/v1/completions"
    req = urllib.request.Request(url, json.dumps(body).encode(), HDRS)
    t0 = time.perf_counter(); ttft = None; n = 0; chars = 0
    usage_tok = None; finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            chunk = raw[6:].strip()
            if chunk == b"[DONE]": break
            try: d = json.loads(chunk)
            except Exception: continue
            if d.get("usage"):
                usage_tok = d["usage"].get("completion_tokens")
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"): finish = ch["finish_reason"]
            dl = ch.get("delta") or {}
            txt = ch.get("text", "") or (dl.get("content") or "") + (dl.get("reasoning_content") or "")
            if txt:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1; chars += len(txt)
    total = time.perf_counter() - t0
    ntok = usage_tok if usage_tok else n          # prefer the server's own count
    dec = (ntok - 1) / (total - ttft) if ttft and ntok > 1 and total > ttft else 0.0
    rec = {"label": LABEL, "run": i + 1, "src": src, "ttft_s": round(ttft or 0, 2),
           "total_s": round(total, 2), "chunks": n, "tokens": ntok, "chars": chars,
           "finish": finish, "decode_tps": round(dec, 2), "api": API}
    if MAX_TOKENS != 256: rec["max_tokens"] = MAX_TOKENS
    if not ttft or not ntok:
        rec["FAILED"] = True
    return rec

out = []
for i in range(RUNS + 1):                 # run 0 is a discarded warm-up
    r = run(i)
    tag = "warmup" if i == 0 else "run"
    print(f"[{tag}] " + json.dumps(r), flush=True)
    if i == 0: continue
    if r.get("FAILED"):
        print("!! run gagal, tidak dihitung", flush=True); continue
    out.append(r)
    with open("runs/bench-stream2.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
if out:
    print("MEDIAN %s: ttft %.2fs  decode %.2f tok/s  (%d run valid, %d token)" % (
        LABEL, st.median([x["ttft_s"] for x in out]),
        st.median([x["decode_tps"] for x in out]), len(out),
        st.median([x["tokens"] for x in out])))
else:
    print("MEDIAN %s: TIDAK ADA RUN VALID" % LABEL)
