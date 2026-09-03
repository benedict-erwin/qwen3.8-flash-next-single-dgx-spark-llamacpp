#!/usr/bin/env python3
"""Client-side streaming benchmark that works against both llama.cpp and vLLM.

Both servers expose an OpenAI-compatible /v1/completions with stream=true, so
timing is measured the same way for both: TTFT is the wall time until the first
token arrives, decode rate is the remaining tokens over the remaining time.
Server-reported `timings` differ between the two and cannot be compared.
"""
import json, sys, time, urllib.request, statistics as st

PORT, LABEL, RUNS = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3

SRCS = ["llama.cpp/common/speculative.cpp",
        "llama.cpp/src/llama-kv-cache.cpp",
        "llama.cpp/common/arg.cpp"]
TASKS = ["Explain what this file does, then write a unit test in C++ for the most complex function in it.",
         "Review this code for correctness bugs and write a patch in diff format for anything you find.",
         "Write a Python script that parses the command-line flags defined in this file and emits them as JSON."]

def model_id():
    """vLLM 404s on an unknown model name; llama.cpp ignores the field. Ask the
    server which model it serves instead of guessing."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:
        print(f"!! could not read /v1/models ({e}), falling back to 'default'", file=sys.stderr)
        return "default"

MODEL = model_id()


def run(i):
    src, task = SRCS[i % 3], TASKS[i % 3]
    ctx = open(src, encoding="utf-8", errors="replace").read()[:40000]
    prompt = f"{task}\n\n```cpp\n{ctx}\n```\n"
    # include_usage makes the server report completion_tokens, so we never have
    # to infer token count from the number of SSE chunks -- vLLM can pack several
    # tokens into one chunk, which silently understated its decode rate by 2.6x.
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 256,
                       "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", body,
                                 {"Content-Type": "application/json"})
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
            txt = ch.get("text", "")
            if txt:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1; chars += len(txt)
    total = time.perf_counter() - t0
    ntok = usage_tok if usage_tok else n          # prefer the server's own count
    dec = (ntok - 1) / (total - ttft) if ttft and ntok > 1 and total > ttft else 0.0
    rec = {"label": LABEL, "run": i + 1, "src": src, "ttft_s": round(ttft or 0, 2),
           "total_s": round(total, 2), "chunks": n, "tokens": ntok, "chars": chars,
           "finish": finish, "decode_tps": round(dec, 2)}
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
