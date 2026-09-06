#!/usr/bin/env python3
"""GSM8K behind a fixed long prefix: does an attention change hold up deep in the context?

First N test items with lm-eval's gsm8k_cot_zeroshot prompt, each preceded by the same
--prefix tokens of unrelated source code, so paths that only engage past a context depth
(the QSA gather patch: 24k cells) are active for every answer. Settings match
bench-accuracy.py: thinking off, temperature 0, seed 1234, max_tokens 1024, acc_final_para
scoring. The prefix is served from the prompt cache after the first item. Needs
tmp/eval-venv (datasets) and the GSM8K cache bench-accuracy.py leaves in tmp/hf-cache.

    API_KEY=<key> tmp/eval-venv/bin/python bench-accuracy-longctx.py LABEL [--n 100] [--prefix 30000] [--src llama.cpp]

Appends one summary line to runs/accuracy-<date>.jsonl; samples under tmp/accuracy/<label>/.
"""
import argparse, datetime as dt, json, os, random, re, sys, time, urllib.request
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.abspath("tmp/hf-cache")); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--prefix", type=int, default=30000); ap.add_argument("--src", default="llama.cpp"); ap.add_argument("--port", default="18080")
a = ap.parse_args()
BASE = f"http://{os.environ.get('HOST', '127.0.0.1')}:{a.port}"
HDRS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HDRS["Authorization"] = "Bearer " + os.environ["API_KEY"]
def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), HDRS)
    return json.load(urllib.request.urlopen(req, timeout=timeout))
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def final_answer(resp):
    for para in reversed(re.split(r"\n\s*\n", resp.strip())):
        if not NUM.search(para): continue
        bolds = [b for b in re.findall(r"\*\*(.+?)\*\*", para) if NUM.search(b)]
        return float(NUM.findall(bolds[-1] if bolds else para)[-1].replace(",", ""))
    return None
from datasets import load_dataset
docs = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
SRC = a.src
files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(SRC + "/src") for f in fs if f.endswith((".cpp", ".h")))
rng = random.Random("gsm8k-prefix"); rng.shuffle(files)
text = "Reference material (unrelated to the question that follows):\n"; i = 0
while len(text) < a.prefix * 5: text += f"\n// ===== {os.path.relpath(files[i], SRC)} =====\n" + open(files[i], errors="replace").read(); i += 1
toks = post("/tokenize", {"content": text})["tokens"]
prefix = post("/detokenize", {"tokens": toks[:a.prefix]})["content"] + "\n\nEnd of reference material.\n\n"
out_dir = f"tmp/accuracy/{a.label}"; os.makedirs(out_dir, exist_ok=True); samples = f"{out_dir}/gsm8k_longctx.jsonl"
t0 = time.time(); ok = 0; cache = []
with open(samples, "w") as f:
    for k, d in enumerate(docs):
        q = f"Q: {d['question']}\nA: Let's think step by step."
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prefix + q}], "max_tokens": 1024, "temperature": 0, "seed": 1234,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        resp = r["choices"][0]["message"].get("content") or ""; tm = r["timings"]
        target = float(d["answer"].split("####")[-1].strip().replace(",", "")); got = final_answer(resp)
        hit = got is not None and abs(got - target) < 1e-6; ok += hit; cache.append(tm.get("cache_n", 0))
        f.write(json.dumps({"doc_id": k, "target": target, "got": got, "ok": hit, "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n"),
                            "predicted_n": tm["predicted_n"], "decode_tok_s": round(tm["predicted_n"] / tm["predicted_ms"] * 1000, 1), "resp": resp}, ensure_ascii=False) + "\n"); f.flush()
        if k % 10 == 9: print(f"[{a.label}] {k+1}/{a.n} acc {ok/(k+1):.3f} cached {sum(cache)/len(cache):.0f} tok {time.time()-t0:.0f}s", flush=True)
row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "label": a.label, "port": int(a.port), "task": "gsm8k_cot_zeroshot+prefix", "n": a.n,
       "prefix_tokens": a.prefix, "thinking": False, "temperature": 0, "concurrent": 1, "acc_final_para": round(ok / a.n, 4),
       "cache_n_mean": round(sum(cache) / len(cache)), "seconds": round(time.time() - t0, 1), "samples": samples}
open(f"runs/accuracy-{dt.date.today()}.jsonl", "a").write(json.dumps(row) + "\n"); print(json.dumps(row))
