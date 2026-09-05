#!/usr/bin/env python3
"""Does re-sending a conversation turn hit the prefix cache on this server?

Background (OPTIMIZATION.md 2026-09-05): turn 2 of a Pi session re-prefilled 16k tokens,
i.e. the whole of turn 1's response. This model (qwen4exp, hybrid SSM + attention) cannot
rewind its recurrent state, so a mismatch ANYWHERE in the resent turn makes llama-server
fall back to the last context checkpoint -- the end of the previous prompt -- and the
cost is the whole response, not the tail after the mismatch.

For each turn shape in SCENARIOS this probe:

  1. generates turn 1 through /completion (to get the token ids), splits it the way the
     server's parser does, resends it with a follow-up and reports cached_tokens for
       as-received   {content, reasoning_content, tool_calls} exactly as returned
       no-reason     reasoning_content dropped   (what a client that discards thinking sends)
       same          turn 1 resent unchanged      (sanity; also resets the slot, so it runs last)
     and diffs the re-rendered history (/apply-template + /tokenize) against the generated
     tokens to print the first divergence, if any;
  2. repeats the client path end to end through /v1/chat/completions -- non-streaming and
     streaming (deltas assembled and arguments re-serialised like an agent client does) --
     and reports HIT/MISS from cached_tokens alone.

Measured 2026-09-05: every shape (text, tool call, text+tool call, 3.6k-token tool call,
long thinking + tool call, streaming or not) is a HIT when the turn is resent as received.
What breaks the cache: dropping or renaming reasoning_content, changing the key order of
tool-call arguments, stripping the trailing newline of a parameter value. None of these
is done by Pi (packages/ai/src/api/openai-completions.ts), so the 16k miss is still
unreproduced from the server side; see OPTIMIZATION.md for the next step
(LLAMA_SERVER_SLOTS_DEBUG during a real session).

Usage:
    API_KEY=<key> ./cache-probe.py [PORT] [--max-tokens N] [--only <scenario>] [--real-flow-only]
    default 18080, 8000, all scenarios (the 'sheet' scenario alone is ~10 min and may hit
    max_tokens; run it with --only sheet --max-tokens 30000 when you mean it)
Only one probe should run at a time.
"""
import json
import os
import sys
import urllib.request

SYSTEM = "You are a terse assistant. Answer directly."
USER1 = ("Write a Python function that parses 'HH:MM:SS' into total seconds, with input "
         "validation, then show three example calls. Keep the whole answer under 200 words.")
USER2 = "Now add a function that does the reverse. Same constraints."

TOOLS = [{"type": "function", "function": {
    "name": "write_file",
    "description": "Create or overwrite a text file.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]
USER_T = ("Use the write_file tool to create hms.py containing a Python function that parses "
          "'HH:MM:SS' into total seconds with input validation. About 20 lines. Call the tool "
          "directly; do not print the code in your reply.")
USER_TT = ("Use the write_file tool to create hms.py containing a Python function that parses "
           "'HH:MM:SS' into total seconds with input validation. About 20 lines. First write one "
           "sentence saying what you are about to do, then call the tool. Do not print the code.")
TOOL_RESULT = "Wrote hms.py (22 lines)."

USER_TL = ("Use the write_file tool to create index.html: a self-contained page (inline CSS and JS, no "
           "libraries) implementing a todo list with add, toggle-done, delete, filter all/active/done, "
           "and localStorage persistence. Aim for roughly 250 lines with comments. First write one "
           "sentence saying what you are about to do, then call the tool. Do not print the code.")

USER_SHEET = """Build a minimal spreadsheet as a single self-contained index.html (inline CSS and JS, no external libraries, no build step, must work when opened via file://). Use the write_file tool to create it.

Requirements:
1. A 10x10 grid, columns A-J, rows 1-10. Clicking a cell selects it; typing edits it; Enter commits and moves down; Tab moves right; Escape cancels the edit; arrow keys move the selection when not editing.
2. A cell value starting with "=" is a formula. Support: numbers, cell references (A1), the operators + - * / with correct precedence and parentheses, and the functions SUM(range) and AVG(range) where range is like A1:B3.
3. Formulas recompute automatically when any cell they depend on changes, including transitive dependencies (A1 -> B1 -> C1). Do not recompute the whole sheet on every keystroke; track dependencies.
4. Circular references must not hang the page. Every cell in the cycle shows #CYCLE.
5. Errors: division by zero shows #DIV/0, a malformed formula shows #ERR, a reference to an empty cell counts as 0.
6. The selected cell's raw content (the formula text, not the computed value) is shown in a formula bar above the grid.
7. Persist the sheet to localStorage and restore it on reload.
8. No frameworks, no eval(), no new Function().

When you are done, list which requirements you are confident are fully met, which you are not sure about, and anything you deliberately left out."""

USER_TK = ("Puzzle: 100 prisoners are numbered 1-100. A room has 100 boxes, each containing one prisoner's "
           "number in a random permutation. Each prisoner may open 50 boxes and must find their own number; "
           "they cannot communicate once the first enters. Work out the cycle-following strategy and derive the "
           "probability that ALL prisoners succeed, as an exact expression and to three decimals. Think it "
           "through carefully. Then use write_file to save the final answer and a 5-line justification to "
           "answer.txt. Reply with one sentence only.")

SCENARIOS = {
    "plain":      (USER1,   None),   # thinking + text, no tool
    "tools":      (USER_T,  TOOLS),  # thinking + tool call, no text
    "tools+text": (USER_TT, TOOLS),  # thinking + text + tool call -- the shape of the 2026-09-05 cold turn
    "tools+long": (USER_TL, TOOLS),  # same shape, several thousand generated tokens (MTP, long argument)
    "sheet":      (USER_SHEET, TOOLS),  # the actual 2026-09-05 one-shot prompt: ~16k tokens, long thinking (~8 min)
    "think+tool": (USER_TK, TOOLS),     # long thinking (a puzzle) followed by a small tool call
}


def post(base, key, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          **({"authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def chat(base, key, messages, max_tokens, tools=None):
    body = {"model": "probe", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools:
        body["tools"] = tools
    d = post(base, key, "/v1/chat/completions", body)
    u = d["usage"]
    return d["choices"][0]["message"], u["prompt_tokens"], u.get("prompt_tokens_details", {}).get("cached_tokens", 0)


def render(base, key, messages, tools=None):
    body = {"messages": messages}
    if tools:
        body["tools"] = tools
    return post(base, key, "/apply-template", body)["prompt"]


def parse_tool_calls(text):
    """Qwen3-Coder XML tool calls -> (content_before, [openai tool_call dicts])."""
    import re
    calls = []
    for i, m in enumerate(re.finditer(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", text, re.S)):
        args = {k: v.strip("\n") for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S)}
        calls.append({"id": f"call_{i}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    before = text.split("<tool_call>", 1)[0].strip("\n")
    return before, calls


def tokenize(base, key, text):
    return post(base, key, "/tokenize", {"content": text, "add_special": False, "with_pieces": False})["tokens"]


def detok(base, key, toks):
    return post(base, key, "/detokenize", {"tokens": toks})["content"]


def parse_like_server(gen_text):
    """Split one generated turn the way llama-server's Qwen3-Coder parser does:
    <think>...</think> -> reasoning_content; XML <tool_call> blocks -> tool_calls
    (exactly one newline stripped on each side of a parameter value); the rest -> content."""
    import re
    reasoning, rest = "", gen_text
    if "</think>" in gen_text:
        reasoning, rest = gen_text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1).strip("\n")
    calls = []
    for i, m in enumerate(re.finditer(r"<tool_call>\n?<function=([^>\n]+)>\n?(.*?)\n?</function>\n?</tool_call>", rest, re.S)):
        args = dict(re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S))
        calls.append({"id": f"call_{i}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    content = rest.split("<tool_call>", 1)[0].strip("\n")
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def run_scenario(base, key, max_tokens, name):
    user, tools = SCENARIOS[name]
    print(f"\n=== scenario: {name} ===")
    t1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    # --- turn 1 via the native endpoint: gives the generated token ids ------------------
    prompt1 = render(base, key, t1, tools)
    g = post(base, key, "/completion", {"prompt": prompt1, "n_predict": max_tokens, "temperature": 0,
                                        "cache_prompt": True, "return_tokens": True})
    gen_text, gen_toks = g["content"], g["tokens"]
    m = parse_like_server(gen_text)
    tool_calls = m.get("tool_calls", [])
    print(f"turn 1: {len(gen_toks)} tokens generated: reasoning {len(m.get('reasoning_content') or '')} chars, "
          f"content {len(m['content'])} chars, tool_calls {len(tool_calls)}, stop={g.get('stop_type', g.get('stopped_eos'))}")
    if not g.get("stopped_eos", g.get("stop_type") == "eos"):
        print("!! turn 1 hit the token limit; raise --max-tokens so the history is a complete turn")
    if tools and not tool_calls:
        print("!! the model did not call the tool; scenario inconclusive")
        return

    follow = ([{"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": TOOL_RESULT}] if tool_calls
              else [{"role": "user", "content": USER2}])
    # order matters: the slot holds the generation state only until a shorter prompt lands on
    # it, so "as-received" must go first; "same" last (it truncates the slot to turn 1).
    variants = {
        "as-received": t1 + [m] + follow,
        "no-reason":   t1 + [{k: v for k, v in m.items() if k != "reasoning_content"}] + follow,
        "same":        t1,
    }
    print(f"\n{'variant':<12} {'prompt':>7} {'cached':>7} {'prefilled':>9}  hit")
    for label, msgs in variants.items():
        _, ptoks, cached = chat(base, key, msgs, 1, tools)
        print(f"{label:<12} {ptoks:>7} {cached:>7} {ptoks - cached:>9}  {cached / ptoks:6.1%}")
    print("(cached = rollback point: on this hybrid model a mismatch anywhere in the turn falls back "
          "to the last context checkpoint, not to the mismatch itself)")

    # --- where does the re-rendered history diverge from what was generated? -------------
    expected = tokenize(base, key, prompt1) + gen_toks
    rendered = tokenize(base, key, render(base, key, variants["as-received"], tools))
    n = next((i for i, (a, b) in enumerate(zip(expected, rendered)) if a != b), len(expected))
    if n >= len(expected):
        print(f"\nno divergence: the re-rendered history reproduces all {len(expected)} generated-path tokens exactly.")
        return
    print(f"\nfirst divergence at token {n} of {len(expected)} generated-path tokens "
          f"(prompt1 = {len(expected) - len(gen_toks)} tokens)")
    lo, hi = max(0, n - 6), n + 8
    print("  generated : " + json.dumps(detok(base, key, expected[lo:hi])))
    print("  re-render : " + json.dumps(detok(base, key, rendered[lo:hi])))
    same_text = detok(base, key, expected[n:n + 40]) == detok(base, key, rendered[n:n + 40])
    print("  same text, different token ids -> tokenizer boundary drift" if same_text
          else "  text differs -> template re-rendering / parser normalisation")


def chat_stream(base, key, messages, max_tokens, tools=None):
    """Consume the SSE stream the way an agent client does and assemble the assistant
    message: reasoning_content / content by concatenating deltas, tool call arguments by
    concatenating the argument deltas and re-serialising the parsed JSON (Pi does
    parseStreamingJson + JSON.stringify)."""
    body = {"model": "probe", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          **({"authorization": f"Bearer {key}"} if key else {})})
    reasoning, content, calls, usage, finish = [], [], {}, {}, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta", {})
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": []})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] = fn.get("name") or c["name"]
                    if fn.get("arguments"):
                        c["args"].append(fn["arguments"])
    m = {"role": "assistant"}
    text = "".join(content)
    if text.strip():
        m["content"] = text
    if "".join(reasoning).strip():
        m["reasoning_content"] = "".join(reasoning)
    if calls:
        def reserialise(raw):
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False)   # what a client does
            except ValueError:
                return raw                                                # truncated stream
        m["tool_calls"] = [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": reserialise("".join(c["args"]))}}
                           for _, c in sorted(calls.items())]
    m["_finish"] = finish
    return m, usage.get("prompt_tokens", 0)


def real_flow(base, key, max_tokens, name, stream=False):
    """The client path end to end: generate through /v1/chat/completions, resend exactly what
    came back plus the follow-up, read cached_tokens. No token ids, so no divergence position,
    but no parser of ours in the loop either."""
    user, tools = SCENARIOS[name]
    t1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if stream:
        m, p1 = chat_stream(base, key, t1, max_tokens, tools)
        finish = m.pop("_finish", None)
        if finish == "length":
            print(f"real-flow stream {name:<11} turn 1 hit max_tokens={max_tokens} (finish=length); "
                  f"history would be a truncated turn -- rerun with a larger --max-tokens")
            return
    else:
        m, p1, _ = chat(base, key, t1, max_tokens, tools)
        m = {k: v for k, v in m.items() if k in ("role", "content", "reasoning_content", "tool_calls") and v not in (None, [])}
    tc = m.get("tool_calls", [])
    follow = ([{"role": "tool", "tool_call_id": tc[0]["id"], "content": TOOL_RESULT}] if tc
              else [{"role": "user", "content": USER2}])
    _, p2, c2 = chat(base, key, t1 + [m] + follow, 1, tools)
    verdict = "HIT (turn kept)" if p2 - c2 <= 64 else "MISS (rolled back to <= turn-1 prompt)"
    print(f"real-flow {'stream' if stream else 'plain ':<6} {name:<11} turn1 prompt {p1:>5}  turn2 prompt {p2:>5} cached {c2:>5}  -> {verdict}"
          f"   [content {len(m.get('content') or '')} chars, tool_calls {len(tc)}]")
def main(argv):
    port, max_tokens, only, real_only = 18080, 8000, None, False
    it = iter(argv)
    for a in it:
        if a == "--max-tokens":
            max_tokens = int(next(it))
        elif a == "--only":
            only = next(it)
        elif a == "--real-flow-only":
            real_only = True
        else:
            port = int(a)
    base, key = f"http://127.0.0.1:{port}", os.environ.get("API_KEY", "")
    for name in SCENARIOS:
        if only in (None, name) and not real_only:
            run_scenario(base, key, max_tokens, name)
    print("\n=== real client flow (chat endpoint both turns, nothing parsed by us) ===")
    for name in SCENARIOS:
        if only in (None, name):
            real_flow(base, key, max_tokens, name)
            real_flow(base, key, max_tokens, name, stream=True)


if __name__ == "__main__":
    main(sys.argv[1:])
