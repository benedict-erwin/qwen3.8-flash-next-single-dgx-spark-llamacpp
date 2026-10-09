#!/usr/bin/env python3
"""Plan C: is greedy decoding (MTP off) reproducible request to request, and where does it diverge?

  planc.py <short|NCHARS> <n_predict> <reps> [src_i]

NCHARS slices prompt source src_i (bench-stream's files) to that many characters; 'short' is
the 47-token prose prompt. Every request: temperature 0, cache_prompt false, n_probs 5. Prints
per run the sha of the content, the first differing character against run 1, and the top-5
first-token candidates with probabilities (prefill determinism check).
"""
import hashlib, json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
HOST = os.environ.get("HOST", "127.0.0.1"); PORT = os.environ.get("PORT", "18080")
HDRS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDRS["Authorization"] = "Bearer " + os.environ["API_KEY"]

SRCS = ["llama.cpp/common/speculative.cpp", "llama.cpp/src/llama-kv-cache.cpp", "llama.cpp/common/arg.cpp"]
TASKS = ["Explain what this file does, then write a unit test in C++ for the most complex function in it.",
         "Review this code for correctness bugs and write a patch in diff format for anything you find.",
         "Write a Python script that parses the command-line flags defined in this file and emits them as JSON."]
SHORT = ("Write a detailed, well-structured explanation of how a Linux page cache works, "
         "covering readahead, dirty page writeback, the LRU lists, and how O_DIRECT bypasses it. "
         "Use section headings and give concrete examples.\n")

what, n_predict, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src_i = int(sys.argv[4]) if len(sys.argv) > 4 else 1
if what == "short":
    prompt = SHORT
else:
    ctx = open(SRCS[src_i % 3], encoding="utf-8", errors="replace").read()[:int(what)]
    prompt = f"{TASKS[src_i % 3]}\n\n```cpp\n{ctx}\n```\n"

outs = []
for r in range(reps):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                       "cache_prompt": False, "stream": False, "n_probs": 5}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"http://{HOST}:{PORT}/completion", body, HDRS), timeout=900) as resp:
        d = json.load(resp)
    c = d.get("content", "")
    sha = hashlib.sha256(c.encode()).hexdigest()[:12]
    probs = d.get("completion_probabilities") or []
    top = [(p.get("token"), round(p.get("prob", 0.0), 6)) for p in (probs[0].get("top_probs", []) if probs else [])]
    div = None
    if outs:
        a = outs[0]
        div = next((k for k in range(min(len(a), len(c))) if a[k] != c[k]), None if len(a) == len(c) else min(len(a), len(c)))
    outs.append(c)
    t = d.get("timings", {})
    print(json.dumps({"run": r + 1, "prompt_n": t.get("prompt_n"), "sha": sha, "same_as_run1": c == outs[0],
                      "first_diff_char": div, "decode_tps": round(t.get("predicted_per_second") or 0, 2),
                      "prefill_tps": round(t.get("prompt_per_second") or 0, 1), "top_probs_tok0": top}), flush=True)
