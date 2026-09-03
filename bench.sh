#!/usr/bin/env bash
# Benchmark decode/prefill tok/s against a running serve.sh instance.
# Usage: ./bench.sh <label> [runs]   -> appends JSON lines to runs/bench-<date>.jsonl
set -euo pipefail
cd "$(dirname "$0")"

LABEL="${1:?usage: bench.sh <label> [runs]}"
RUNS="${2:-3}"
PORT=18080
OUT="runs/bench-$(date +%F).jsonl"
mkdir -p runs

# Distinct prompt per run. Reusing one prompt at temperature 0 makes every run
# after the first regenerate identical text, which the n-gram cache then drafts
# almost perfectly -- inflating speculative-decoding results into a measurement
# of self-repetition rather than of real work. Measured 2026-09-03: same prompt
# gave ngram-mod 98.9 tok/s at 82% acceptance; that number is an artifact.
PROMPTS=(
  "Write a detailed technical explanation of how mixture-of-experts routing works in modern transformer language models, covering load balancing, expert capacity, and token dropping."
  "Explain the tradeoffs between optimistic and pessimistic concurrency control in distributed databases, and describe when each one degrades badly under contention."
  "Describe how a modern operating system reclaims memory under pressure: page cache eviction, swap, the OOM killer, and why unified-memory systems complicate this."
  "Discuss the design of columnar storage formats for analytical workloads, covering encoding schemes, predicate pushdown, and the cost of wide schemas."
  "Explain how TLS 1.3 shortened the handshake compared to TLS 1.2, and what security properties the removed round trip used to provide."
)

for i in $(seq 1 "$RUNS"); do
  PROMPT="${PROMPTS[$(( (i - 1) % ${#PROMPTS[@]} ))]}"
  curl -sf "http://127.0.0.1:$PORT/completion" \
    -d "$(jq -n --arg p "$PROMPT" '{prompt: $p, n_predict: 256, temperature: 0}')" \
  | jq -c --arg label "$LABEL" --argjson run "$i" \
      '{label: $label, run: $run,
        decode_tps: .timings.predicted_per_second,
        prefill_tps: .timings.prompt_per_second,
        n_decoded: .timings.predicted_n,
        draft_n: (.timings.draft_n // null),
        draft_accepted: (.timings.draft_n_accepted // null)}' \
  | tee -a "$OUT"
done
