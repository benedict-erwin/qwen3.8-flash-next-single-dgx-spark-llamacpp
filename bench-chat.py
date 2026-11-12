#!/usr/bin/env python3
"""Decode rate on prose and thinking workloads through /v1/chat/completions (streaming).
usage: API_KEY=... ./bench-chat.py <port> <label> <out.jsonl>     (llama.cpp only: it sends chat_template_kwargs)
Every request in one run has a different prompt; request 0 is a warm-up and is not recorded.
Tokens come from usage.completion_tokens; decode rate = (tokens-1) / (time after the first delta)."""
import json, os, sys, time, urllib.request, statistics as st
PORT, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HOST = os.environ.get("HOST", "127.0.0.1")          # match serve.sh / stack.sh
H = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"): H["Authorization"] = "Bearer " + os.environ["API_KEY"]
PROSE = ["Write a reflective essay about why people keep handwritten letters long after the sender is gone.",
         "Describe a small fishing town at dawn in autumn, in the style of literary fiction. No dialogue.",
         "Write an opinion column arguing that cities should plant fruit trees along public streets.",
         "Tell the story of a clockmaker who receives a watch that runs backwards. About five paragraphs.",
         "Write a long, warm letter from a grandmother to her grandchild who is moving abroad to study."]
THINK = ["A tank is filled by pipe A in 6 hours and pipe B in 9 hours, and drained by pipe C in 12 hours. A and C are opened at 8:00, B joins at 9:30. When is the tank full?",
         "Five people (Ana, Budi, Chen, Dewi, Eko) sit in a row. Ana is not at either end. Budi is left of Chen but not adjacent. Dewi is next to Eko. Chen is not next to Ana. List every valid arrangement.",
         "How many integers between 1 and 2000 inclusive are divisible by 3 or 7 but not by 21, and have a digit sum that is even? Explain how you would count them.",
         "A farmer has 240 m of fence for a rectangular field along a river (no fence on the river side), split into three equal pens by fences perpendicular to the river. What dimensions maximise the area?"]
WARM = "Explain in two paragraphs how a bicycle stays upright while moving."
def ask(prompt, think, max_tokens=400):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": think}}).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", body, H)
    t0 = time.perf_counter(); t1 = None; usage = None; tim = None; fin = None; nc = nr = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if d.get("timings"): tim = d["timings"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"):
                    if t1 is None: t1 = time.perf_counter()
                    nc += len(dl.get("content") or ""); nr += len(dl.get("reasoning_content") or "")
                if c.get("finish_reason"): fin = c["finish_reason"]
    t2 = time.perf_counter(); n = (usage or {}).get("completion_tokens", 0)
    rec = {"tokens": n, "finish": fin, "ttft_s": round((t1 or t2) - t0, 2), "chars_content": nc, "chars_reasoning": nr,
           "decode_tps": round((n - 1) / (t2 - t1), 2) if t1 and n > 1 else 0.0}
    if tim: rec.update({k: tim[k] for k in ("draft_n", "draft_n_accepted") if k in tim})
    return rec
ask(WARM, False, 64)
res = {"prose": [], "thinking": []}
for cat, prompts, think in (("prose", PROSE, False), ("thinking", THINK, True)):
    for i, p in enumerate(prompts):
        r = ask(p, think); r.update({"label": LABEL, "cat": cat, "prompt": i})
        ok = r["finish"] == "length" and r["tokens"] >= 300
        if not ok: r["FAILED"] = True
        else: res[cat].append(r["decode_tps"])
        print(json.dumps(r), flush=True)
        open(OUT, "a").write(json.dumps(r) + "\n")
for cat, v in res.items():
    print(f"MEDIAN {LABEL} {cat}: {st.median(v):.2f} tok/s  (n={len(v)}, min {min(v):.2f}, max {max(v):.2f})" if v else f"MEDIAN {LABEL} {cat}: no valid run")
