#!/usr/bin/env python3
"""Decode speed against context depth, llama.cpp only.

Timing comes from llama-server's own `timings` (prompt_ms, predicted_ms), which is
fine for A/B between two llama.cpp builds on the same machine but not comparable
with vLLM: cross-backend numbers come from bench-stream.py only. Every request gets
a distinct prompt (real source files, shuffled, cut to exactly N tokens through
/tokenize + /detokenize), so the prefix cache never hits; --fixed makes the prompt
independent of LABEL for byte-identity checks between builds (same prompt, greedy).

    API_KEY=<key> ./bench-longctx.py LABEL [--sizes 8000,16000,32000,64000] [--reps 2]
                  [--temp 0.7] [--max 256] [--fixed] [--src llama.cpp]

Output: one JSON line per request appended to runs/longctx-<date>.jsonl with the
server timings, usage.completion_tokens, draft acceptance and a hash of the output.
"""
import argparse, datetime, hashlib, json, os, random, sys, time, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--sizes", default="8000,16000,32000,64000")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max", type=int, default=256)
ap.add_argument("--fixed", action="store_true", help="prompt text independent of LABEL")
ap.add_argument("--src", default="llama.cpp", help="source tree the prompts are cut from")
ap.add_argument("--port", default="18080")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=f"runs/longctx-{datetime.date.today()}.jsonl")
a = ap.parse_args()

HOST = os.environ.get("HOST", "127.0.0.1")
HDRS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDRS["Authorization"] = "Bearer " + os.environ["API_KEY"]

def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://{HOST}:{a.port}{path}", json.dumps(body).encode(), HDRS)
    return json.load(urllib.request.urlopen(req, timeout=timeout))

files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(a.src, "src")) for f in fs if f.endswith((".cpp", ".h")))
files += sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(a.src, "ggml/src/ggml-cuda")) for f in fs if f.endswith((".cu", ".cuh")))
if not files:
    sys.exit(f"no sources under {a.src}; pass --src <llama.cpp checkout>")

for size in [int(s) for s in a.sizes.split(",")]:
    for rep in range(a.reps):
        key = "fixed" if a.fixed else a.label
        rng = random.Random(f"{key}/{size}/{rep}/{a.seed}")
        order = files[:]
        rng.shuffle(order)
        text = f"run {key} {size} {rep} {rng.random():.6f}\n"
        i = 0
        while len(text) < size * 5 and i < len(order):
            text += f"\n// ===== {os.path.relpath(order[i], a.src)} =====\n" + open(order[i], errors="replace").read()
            i += 1
        toks = post("/tokenize", {"content": text})["tokens"]
        head = post("/detokenize", {"tokens": toks[:size - 120]})["content"]
        prompt = head + "\n\nQuestion: in about 200 words, list which source files appear above and explain what the flash-attention related code does."
        t0 = time.time()
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": a.max,
                                          "temperature": a.temp, "seed": rng.randrange(1 << 30)})
        wall = time.time() - t0
        tm = r["timings"]
        msg = r["choices"][0]["message"]
        out = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
        row = {"label": a.label, "target": size, "rep": rep, "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n"),
               "prompt_ms": tm["prompt_ms"], "prefill_tok_s": round(tm["prompt_n"] / tm["prompt_ms"] * 1000, 1),
               "predicted_n": tm["predicted_n"], "completion_tokens": r["usage"]["completion_tokens"],
               "predicted_ms": tm["predicted_ms"], "decode_tok_s": round(tm["predicted_n"] / tm["predicted_ms"] * 1000, 2),
               "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"), "wall_s": round(wall, 1),
               "temp": a.temp, "finish": r["choices"][0].get("finish_reason"),
               "out_sha": hashlib.sha256(out.encode()).hexdigest()[:16], "out_head": out.replace("\x00", "")[:100]}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
