#!/usr/bin/env bash
# Benchmark shaped like a coding-agent workload: a long real-code prompt, then a
# code-generation task. Reports prefill and decode separately -- for long prompts
# prefill dominates time-to-first-token, and the short-prompt numbers from
# bench.sh are pure noise for that.
# Usage: ./bench-code.sh <label> [runs]
set -uo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: bench-code.sh <label> [runs]}"
RUNS="${2:-3}"
PORT=18080
OUT="runs/bench-code-$(date +%F).jsonl"
mkdir -p runs

# Real source as context, the way a coding agent would paste a file in.
# Three different files so no run can redraft the previous run's output.
SRCS=(
  "llama.cpp/common/speculative.cpp"
  "llama.cpp/src/llama-kv-cache.cpp"
  "llama.cpp/common/arg.cpp"
)
TASKS=(
  "Explain what this file does, then write a unit test in C++ for the most complex function in it."
  "Review this code for correctness bugs and write a patch in diff format for anything you find."
  "Write a Python script that parses the command-line flags defined in this file and emits them as JSON."
)

for i in $(seq 1 "$RUNS"); do
  idx=$(( (i - 1) % ${#SRCS[@]} ))
  src="${SRCS[$idx]}"
  # Prompts are real llama.cpp sources so they stay identical across runs and
  # machines; llama.cpp/ is gitignored, so a fresh clone must fetch it first.
  [[ -f "$src" ]] || { echo "missing prompt source: $src -- git clone https://github.com/ggml-org/llama.cpp.git (see SETUP.md)" >&2; exit 1; }
  # ~6000 lines is far more than needed; cap to keep the prompt near 8-16k tokens.
  ctx=$(head -c 40000 "$src")
  prompt=$(printf '%s\n\n```cpp\n%s\n```\n' "${TASKS[$idx]}" "$ctx")

  curl -sf "http://127.0.0.1:$PORT/completion" \
    -d "$(jq -n --arg p "$prompt" '{prompt: $p, n_predict: 256, temperature: 0}')" \
  | jq -c --arg label "$LABEL" --argjson run "$i" --arg src "$src" \
      '{label: $label, run: $run, src: $src,
        prompt_tokens: .timings.prompt_n,
        prefill_tps: .timings.prompt_per_second,
        ttft_ms: .timings.prompt_ms,
        decode_tps: .timings.predicted_per_second,
        n_decoded: .timings.predicted_n,
        draft_n: (.timings.draft_n // null),
        draft_accepted: (.timings.draft_n_accepted // null)}' \
  | tee -a "$OUT"
done
