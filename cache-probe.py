#!/usr/bin/env python3
"""Why does the second turn of a conversation miss the prefix cache?

Observed 2026-09-05 (runs/cache-ratio-2026-09-05.jsonl, task 6875): turn 2 of a Pi
session re-prefilled 16k tokens -- the whole of turn 1's response -- even though the
client resent it verbatim. This probe reproduces a two-turn exchange against the
running server and reports, for each way of resending the assistant turn, how many
prompt tokens came from the cache (usage.prompt_tokens_details.cached_tokens):

    same        turn 1 resent unchanged            -> sanity: should be ~all cached
    no-reason   assistant = {content}              -> what a client that drops thinking sends
    reason      assistant = {content, reasoning}   -> what a client that keeps thinking sends

It then locates the first token where the re-rendered history diverges from what the
model actually generated, using /apply-template, /tokenize and the token ids returned
by the native /completion endpoint. That separates the two server-side hypotheses:

    (a) generated token ids != re-tokenised text   (boundary drift, e.g. after MTP)
    (b) the chat template re-renders the thinking block differently

If both 'reason' and 'same' are ~fully cached here but a Pi session is still cold on
turn 2, the client is not resending what it received (hypothesis c) -- capture its
traffic to confirm.

Usage:
    API_KEY=<key> ./cache-probe.py [PORT] [--max-tokens N]     default 18080, 1200
Only one probe should run at a time; it takes ~one response worth of decode.
"""
import json
import os
import sys
import urllib.request

SYSTEM = "You are a terse assistant. Answer directly."
USER1 = ("Write a Python function that parses 'HH:MM:SS' into total seconds, with input "
         "validation, then show three example calls. Keep the whole answer under 200 words.")
USER2 = "Now add a function that does the reverse. Same constraints."


def post(base, key, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          **({"authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def chat(base, key, messages, max_tokens):
    d = post(base, key, "/v1/chat/completions",
             {"model": "probe", "messages": messages, "max_tokens": max_tokens, "temperature": 0})
    u = d["usage"]
    return d["choices"][0]["message"], u["prompt_tokens"], u.get("prompt_tokens_details", {}).get("cached_tokens", 0)


def render(base, key, messages):
    return post(base, key, "/apply-template", {"messages": messages})["prompt"]


def tokenize(base, key, text):
    return post(base, key, "/tokenize", {"content": text, "add_special": False, "with_pieces": False})["tokens"]


def detok(base, key, toks):
    return post(base, key, "/detokenize", {"tokens": toks})["content"]


def main(argv):
    port, max_tokens = 18080, 1200
    it = iter(argv)
    for a in it:
        if a == "--max-tokens":
            max_tokens = int(next(it))
        else:
            port = int(a)
    base, key = f"http://127.0.0.1:{port}", os.environ.get("API_KEY", "")

    t1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER1}]

    # --- turn 1 through the native endpoint so we get the generated token ids -----------
    prompt1 = render(base, key, t1)
    g = post(base, key, "/completion", {"prompt": prompt1, "n_predict": max_tokens, "temperature": 0,
                                        "cache_prompt": True, "return_tokens": True})
    gen_text, gen_toks = g["content"], g["tokens"]
    if "</think>" in gen_text:
        reasoning, content = gen_text.split("</think>", 1)
        reasoning, content = reasoning.strip("\n"), content.lstrip("\n")
    else:
        reasoning, content = "", gen_text
    print(f"turn 1: {len(gen_toks)} tokens generated, reasoning {len(reasoning)} chars, "
          f"content {len(content)} chars, stop={g.get('stop_type', g.get('stopped_eos'))}")
    if not g.get("stopped_eos", g.get("stop_type") == "eos"):
        print("!! turn 1 hit the token limit; raise --max-tokens so the history is a complete turn")

    # --- turn 2 variants, prefill only (max_tokens=1) -------------------------------------
    variants = {
        "same":      t1,
        "no-reason": t1 + [{"role": "assistant", "content": content},
                           {"role": "user", "content": USER2}],
        "reason":    t1 + [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                           {"role": "user", "content": USER2}],
    }
    print(f"\n{'variant':<10} {'prompt':>7} {'cached':>7} {'prefilled':>9}  hit")
    for name, msgs in variants.items():
        _, ptoks, cached = chat(base, key, msgs, 1)
        print(f"{name:<10} {ptoks:>7} {cached:>7} {ptoks - cached:>9}  {cached / ptoks:6.1%}")

    # --- where does the re-rendered history diverge from what was generated? -------------
    expected = tokenize(base, key, prompt1) + gen_toks
    rendered = tokenize(base, key, render(base, key, variants["reason"]))
    n = next((i for i, (a, b) in enumerate(zip(expected, rendered)) if a != b), len(expected))
    if n >= len(expected):
        print(f"\nno divergence: the re-rendered history reproduces all {len(expected)} generated-path "
              f"tokens exactly. Server side is cache-friendly; a cold turn 2 means the client did not "
              f"resend reasoning_content as received (hypothesis c) -- capture its traffic.")
        return
    print(f"\nfirst divergence at token {n} of {len(expected)} generated-path tokens "
          f"(prompt1 = {len(expected) - len(gen_toks)} tokens)")
    if n < len(expected) - len(gen_toks):
        print("  -> inside the turn-1 prompt itself: template renders the *history* header differently")
    lo, hi = max(0, n - 6), n + 8
    print("  generated : " + json.dumps(detok(base, key, expected[lo:hi])))
    print("  re-render : " + json.dumps(detok(base, key, rendered[lo:hi])))
    print("  generated ids: ", expected[lo:hi])
    print("  re-render ids: ", rendered[lo:hi])
    same_text = detok(base, key, expected[n:n + 40]) == detok(base, key, rendered[n:n + 40])
    print("  same text, different token ids -> hypothesis (a) boundary drift" if same_text
          else "  text differs -> hypothesis (b) template/whitespace re-rendering")


if __name__ == "__main__":
    main(sys.argv[1:])
