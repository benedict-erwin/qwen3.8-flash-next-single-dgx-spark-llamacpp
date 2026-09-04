#!/usr/bin/env python3
"""Prefix-cache hit ratio of real usage, reconstructed from llama-server logs.

llama-server does not export a cached-token counter, but its default INFO log
carries enough per-request lines to rebuild one:

    prompt eval time = ... / P tokens      tokens actually prefilled (cache misses)
           eval time = ... / G tokens      tokens generated
    stop processing: n_tokens = N          context at release = prompt + G - 1

so prompt_total = N - G + 1 and cached = prompt_total - P. Together they give
the number OPTIMIZATION.md leaves open: what share of calls is cold. The
break-even there is 12% cold calls -- above it vLLM's prefill lead pays off,
below it llama.cpp with MTP is the better backend.

Usage:
    ./cache-ratio.py [--cold 0.5] [-v] [LOG ...]     default LOG: runs/serve-current.log

A call is "cold" when less than --cold of its prompt came from the cache.
Warm-up probes (1 generated token) are excluded, as they are in the benchmarks.
"""
import os, re, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RE_TASK = re.compile(r"slot\s+\S+: id\s+(\d+) \| task (\d+) \| (.*)$")
RE_PROMPT = re.compile(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_EVAL = re.compile(r"^\s*eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_STOP = re.compile(r"stop processing: n_tokens =\s+(\d+)")


def parse(path):
    """Yield one dict per finished request, in log order."""
    open_tasks = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_TASK.search(line)
            if not m:
                continue
            key, rest = (m.group(1), m.group(2)), m.group(3)
            t = open_tasks.setdefault(key, {})
            if (p := RE_PROMPT.search(rest)):
                t["prefill_ms"], t["prefilled"] = float(p.group(1)), int(p.group(2))
            elif (e := RE_EVAL.search(rest)):
                t["gen_ms"], t["generated"] = float(e.group(1)), int(e.group(2))
            elif (s := RE_STOP.search(rest)):
                t["n_ctx"] = int(s.group(1))
                open_tasks.pop(key)
                if {"prefilled", "generated", "n_ctx"} <= t.keys():
                    t["prompt"] = t["n_ctx"] - t["generated"] + 1
                    t["cached"] = max(t["prompt"] - t["prefilled"], 0)
                    t["task"] = int(key[1])
                    yield t


def main(argv):
    cold_thr, verbose, logs = 0.5, False, []
    it = iter(argv)
    for a in it:
        if a == "--cold":
            cold_thr = float(next(it))
        elif a == "-v":
            verbose = True
        else:
            logs.append(a)
    logs = logs or ["runs/serve-current.log"]

    calls = [c for p in logs for c in parse(p) if c["generated"] > 1]
    if not calls:
        sys.exit(f"no finished requests found in {' '.join(logs)}")

    if verbose:
        print(f"{'task':>6} {'prompt':>7} {'cached':>7} {'hit':>6} {'prefill s':>10} {'gen':>5}")
        for c in calls:
            hit = c["cached"] / c["prompt"] if c["prompt"] else 0.0
            print(f"{c['task']:>6} {c['prompt']:>7} {c['cached']:>7} {hit:>6.1%} "
                  f"{c['prefill_ms'] / 1000:>10.2f} {c['generated']:>5}")
        print()

    n = len(calls)
    prompt = sum(c["prompt"] for c in calls)
    cached = sum(c["cached"] for c in calls)
    cold = [c for c in calls if c["prompt"] and c["cached"] / c["prompt"] < cold_thr]
    prefill_s = sum(c["prefill_ms"] for c in calls) / 1000

    print(f"requests          {n}   (warm-up probes excluded)")
    print(f"prompt tokens     {prompt}   cached {cached} ({cached / prompt:.1%})")
    print(f"cold calls        {len(cold)} / {n} = {len(cold) / n:.1%}   (cached share < {cold_thr:.0%})")
    print(f"prefill wall      {prefill_s:.1f} s total, {prefill_s / n:.2f} s per call")
    verdict = "vLLM territory" if len(cold) / n > 0.12 else "llama.cpp + MTP territory"
    print(f"break-even 12%    -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1:])
