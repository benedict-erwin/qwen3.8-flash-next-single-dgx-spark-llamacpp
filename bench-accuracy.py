#!/usr/bin/env python3
"""Accuracy of a served quant on GSM8K and HumanEval+, through the OpenAI API.

Runs the same two task sets against whichever backend is up (llama.cpp
UD-Q4_K_XL on 18080, vLLM NVFP4 on 18300) so the two quants can be compared on
identical prompts, harness, and decoding settings. This is a relative
measurement: the BF16 original (360 GB) does not fit on the GB10, so absolute
retention can only be read against published BF16 numbers.

Settings, fixed on purpose so runs stay comparable:
  - thinking off via chat_template_kwargs.enable_thinking=false (both servers
    honour it), temperature 0, seed 1234
  - GSM8K: lm-eval `gsm8k_cot_zeroshot`, first N test items. Two scores are kept:
    lm-eval's flexible-extract (last number anywhere) and `acc_final_para`, the
    last number of the final paragraph with a bold number preferred. The model
    writes "**$64** for the 16 glasses" style conclusions, so flexible-extract
    fails ~12% of correct answers; the second score is the one to read.
  - HumanEval+: evalplus `humaneval`, greedy, base + plus tests, max_tokens 2048
    (evalplus defaults to 768, which truncated answers that reason in prose first)

Usage:
    ./bench-accuracy.py <port> <label> [--gsm8k N] [--concurrent N] [--note TEXT] [--skip-gsm8k] [--skip-humaneval]
    ./bench-accuracy.py <port> <label> --rescore-gsm8k     # re-score saved samples only

HumanEval+ always runs all 164 problems: evalplus refuses to score a subset.
`--concurrent` (default 1) is the number of GSM8K requests in flight. Keep it at
1 on llama.cpp with MTP: upstream #28286 reports cross-slot content contamination
with draft-mtp and --parallel > 1. HumanEval+ is always sequential (evalplus).

Needs tmp/eval-venv (see SETUP.md). Summary lines are appended to
runs/accuracy-YYYY-MM-DD.jsonl; per-sample logs stay under tmp/accuracy/<label>/.
"""
import datetime as dt, json, os, re, sys, time, urllib.request

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Keep dataset downloads (GSM8K, HumanEval+) out of the shared HF cache that
# the vLLM container owns; tmp/ is disposable.
os.environ.setdefault("HF_HOME", os.path.abspath("tmp/hf-cache"))
os.environ.setdefault("EVALPLUS_CACHE_DIR", os.path.abspath("tmp/evalplus-cache"))

PORT, LABEL = sys.argv[1], sys.argv[2]
HOST = os.environ.get("HOST", "127.0.0.1")
API_KEY = os.environ.get("API_KEY", "")
BASE = f"http://{HOST}:{PORT}/v1"
N_GSM, N_CONC, DO_GSM, DO_HE, RESCORE, NOTE = 300, 1, True, True, False, ""
args = iter(sys.argv[3:])
for a in args:
    if a == "--gsm8k": N_GSM = int(next(args))
    elif a == "--concurrent": N_CONC = int(next(args))
    elif a == "--note": NOTE = next(args)
    elif a == "--skip-gsm8k": DO_GSM = False
    elif a == "--skip-humaneval": DO_HE = False
    elif a == "--rescore-gsm8k": RESCORE = True
    else: sys.exit(f"unknown option {a}")

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
HE_MAX_TOKENS = 2048
OUT_DIR = f"tmp/accuracy/{LABEL}"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY = f"runs/accuracy-{dt.date.today()}.jsonl"
# Both harnesses read the key from the environment; the servers ignore it
# unless one was set at start time.
os.environ["OPENAI_API_KEY"] = API_KEY or "local"


def model_id():
    """vLLM 404s on an unknown model name; llama.cpp ignores the field."""
    hdrs = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    req = urllib.request.Request(f"{BASE}/models", headers=hdrs)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)["data"][0]["id"]


MODEL = model_id()


def record(task, n, metrics, seconds, samples_path):
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "label": LABEL,
           "port": int(PORT), "model": MODEL, "task": task, "n": n,
           "thinking": False, "temperature": 0, "concurrent": N_CONC, "note": NOTE, **metrics,
           "seconds": round(seconds, 1), "samples": samples_path}
    with open(SUMMARY, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{LABEL}] {task} n={n} {metrics} in {seconds:.0f}s -> {SUMMARY}")


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def final_answer(resp):
    """Last number of the final paragraph that has one; a bold number there wins."""
    for para in reversed(re.split(r"\n\s*\n", resp.strip())):
        if not NUM.search(para):
            continue
        bolds = [b for b in re.findall(r"\*\*(.+?)\*\*", para) if NUM.search(b)]
        return float(NUM.findall(bolds[-1] if bolds else para)[-1].replace(",", ""))
    return None


def score_gsm8k(path):
    """Both scores from a saved samples file: lm-eval's own verdicts + final_answer()."""
    rows = [json.loads(l) for l in open(path)]
    flex = [r for r in rows if r["filter"] == "flexible-extract"]
    n = len(flex)
    lm = sum(bool(r["exact_match"]) for r in flex)
    ours = 0
    for r in flex:
        got = final_answer(r["resp"])
        ours += got is not None and abs(got - float(r["target"].replace(",", ""))) < 1e-6
    return n, {"acc_flexible_extract": round(lm / n, 4), "acc_final_para": round(ours / n, 4)}


def run_gsm8k():
    import lm_eval
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args={"base_url": f"{BASE}/chat/completions", "model": MODEL,
                    "num_concurrent": N_CONC, "max_retries": 3, "timeout": 600,
                    "tokenized_requests": False, "tokenizer_backend": None},
        tasks=["gsm8k_cot_zeroshot"], limit=N_GSM, apply_chat_template=True,
        gen_kwargs={"temperature": 0, "max_gen_toks": 1024, **NO_THINK},
        log_samples=True)
    r = res["results"]["gsm8k_cot_zeroshot"]
    metrics = {k: round(v, 4) for k, v in r.items() if k.startswith("exact_match")}
    # lm-eval emits one sample row per filter (strict-match, flexible-extract);
    # keep both, count documents once.
    samples = res["samples"]["gsm8k_cot_zeroshot"]
    path = f"{OUT_DIR}/gsm8k_cot_zeroshot.jsonl"
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps({"doc_id": s["doc_id"], "filter": s["filter"],
                                "target": s["target"].split("####")[-1].strip(),
                                "filtered": s["filtered_resps"], "exact_match": s["exact_match"],
                                "resp": s["resps"][0][0]}, ensure_ascii=False) + "\n")
    n, scores = score_gsm8k(path)
    record("gsm8k_cot_zeroshot", n, {**scores, "stderr": metrics.get("exact_match_stderr,flexible-extract")},
           time.time() - t0, path)


def run_humaneval():
    import openai.resources.chat.completions as occ
    from evalplus.evaluate import evaluate
    # evalplus' OpenAI backend has no hook for extra body fields, so thread the
    # no-thinking flag through the SDK call itself.
    _create = occ.Completions.create
    def create(self, *a, **kw):
        kw["extra_body"] = {**kw.get("extra_body", {}), **NO_THINK}
        kw["seed"] = 1234
        kw["max_tokens"] = HE_MAX_TOKENS
        return _create(self, *a, **kw)
    occ.Completions.create = create

    t0 = time.time()
    root = f"{OUT_DIR}/evalplus"
    evaluate(dataset="humaneval", model=MODEL, backend="openai", base_url=BASE,
             greedy=True, root=root, parallel=8)
    ident = MODEL.strip("./").replace("/", "--") + "_openai_temp_0.0"
    samples = f"{root}/humaneval/{ident}.jsonl"
    with open(samples.replace(".jsonl", "_eval_results.json")) as f:
        ev = f.read()
    ev = json.loads(ev)["eval"]
    n = len(ev)
    # evalplus' own definition: "+" counts a task only when base AND plus pass.
    base = sum(r[0]["base_status"] == "pass" for r in ev.values())
    plus = sum(r[0]["base_status"] == r[0]["plus_status"] == "pass" for r in ev.values())
    record("humaneval", n, {"pass@1_base": round(base / n, 4), "pass@1_plus": round(plus / n, 4),
                            "max_tokens": HE_MAX_TOKENS},
           time.time() - t0, samples)


if __name__ == "__main__":
    print(f"[{LABEL}] {BASE} model={MODEL} gsm8k={N_GSM if DO_GSM else '-'} humaneval={'164' if DO_HE else '-'}")
    if RESCORE:
        path = f"{OUT_DIR}/gsm8k_cot_zeroshot.jsonl"
        n, scores = score_gsm8k(path)
        record("gsm8k_cot_zeroshot", n, {**scores, "rescored": True}, 0.0, path)
        sys.exit(0)
    if DO_GSM: run_gsm8k()
    if DO_HE: run_humaneval()
