#!/usr/bin/env python3
"""Decode-only request for an nsys capture: prof2.py cache|run prose|code
'cache' evaluates the prompt (n_predict=1, cache_prompt) so the captured 'run' is decode only."""
import json, os, sys, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)  # repo root
H = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"): H["Authorization"] = "Bearer " + os.environ["API_KEY"]
step, kind = sys.argv[1], sys.argv[2]
if kind == "prose":
    prompt = ("Write a detailed, well-structured explanation of how a Linux page cache works, "
              "covering readahead, dirty page writeback, the LRU lists, and how O_DIRECT bypasses it. "
              "Use section headings and give concrete examples.\n")
else:
    src = open("llama.cpp/src/llama-sampler.cpp", errors="replace").read()[3000:11000]
    prompt = ("Write a Python script that parses the C++ below and emits every function signature as JSON. "
              "Output only the code.\n\n```cpp\n" + src + "\n```\n\n```python\n")
body = {"prompt": prompt, "n_predict": 1 if step == "cache" else 384, "temperature": 0, "cache_prompt": True, "stream": False}
d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18080/completion", json.dumps(body).encode(), H), timeout=900))
t = d["timings"]
print(json.dumps({"label": os.environ.get("LABEL"), "kind": kind, "step": step, "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                  "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"), "predicted_tps": t.get("predicted_per_second"),
                  "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"), "stop": d.get("stop_type")}))
