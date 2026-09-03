#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next UD-Q4_K_XL on DGX Spark GB10.
# Default = the measured baseline (21.8 tok/s on 2026-09-01): GPU experts,
# PLE n-gram table pinned to CPU, KV cache q8_0, no speculative decoding.
#
# Usage: ./serve.sh [--ngram-mod] [--mtp] [--mmap] [--nmax N]
#   --ngram-mod  n-gram speculative decoding (free, no extra model)
#   --mtp        MTP draft head sidecar   (needs ./download-mtp.sh first)
#   --mmap       PLE served from page cache instead of resident (~26.8 GiB less RSS)
#   --nmax N     max draft tokens per step (default 2)
#   BUILD=build  use the old 5d4a3be binary instead of build-new
set -euo pipefail
cd "$(dirname "$0")"

# BUILD=fork uses the Unsloth prebuilt (b10715-mix-86bd2d3), which is the only
# build that can load their MTP draft heads -- upstream rejects them for missing
# output_hc_norm.weight / token_embd.weight. See OPTIMIZATION.md.
BUILD="${BUILD:-build-new}"
if [[ "$BUILD" == "fork" ]]; then
  FORKDIR=forks/unsloth-b10715
  BIN="$FORKDIR/llama-server"
  export LD_LIBRARY_PATH="$PWD/$FORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  BIN="llama.cpp/$BUILD/bin/llama-server"
fi
MODEL=models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
MTP=models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PORT=18080
NMAX=2
# Bind address. Default stays loopback so local benchmarking is unchanged.
#   HOST=tailscale  -> bind to this node's tailnet IP only: reachable from every
#                      device on the tailnet (and directly over LAN when they are
#                      on the same network), never from the WiFi subnet at large
#                      and never from the docker/kube bridges.
#   HOST=0.0.0.0    -> every interface; only do this behind a firewall.
HOST="${HOST:-127.0.0.1}"
if [[ "$HOST" == "tailscale" ]]; then
  HOST=$(tailscale ip -4 2>/dev/null | head -1)
  [[ -n "$HOST" ]] || { echo "cannot resolve the tailscale IP -- is tailscaled up?" >&2; exit 1; }
fi
# The server is unauthenticated by default; set one when binding off-loopback.
API_KEY="${API_KEY:-}"
# Context defaults to 128k, not the model's 262k native: KV at q8_0 costs ~6.4 GiB
# here vs ~12.8 GiB at full native, and 32k was too tight for a coding agent
# pasting real files. Override with CTX=262144 ./serve.sh
CTX="${CTX:-131072}"
# --alias gives the API a clean model name; without it clients see the full
# gguf path as the model id, which some harnesses reject or mangle.
ALIAS="${ALIAS:-qwen3.8-flash-next}"

EXTRA=()
LABEL="baseline"
USE_MMAP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ngram-mod) EXTRA+=(--spec-type ngram-mod); LABEL="ngram-mod" ;;
    --mtp)       EXTRA+=(--spec-type draft-mtp -md "$MTP" -ngld 999 --spec-draft-p-min 0.75); LABEL="mtp" ;;
    --mmap)      USE_MMAP=1 ;;
    --nmax)      shift; NMAX="$1" ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --spec-draft-n-max only means anything once a spec type is active.
[[ ${#EXTRA[@]} -gt 0 ]] && EXTRA+=(--spec-draft-n-max "$NMAX")

# mmap keeps the 26.82 GiB PLE table in reclaimable page cache rather than
# resident, so memory pressure degrades to "slow" instead of the NVRM stall ->
# watchdog panic hit on 2026-09-01. See OPTIMIZATION.md.
LOADMODE=(-lm mmap); (( USE_MMAP )) || LOADMODE=()

[[ -x "$BIN" ]] || { echo "missing binary: $BIN (run the build first)" >&2; exit 1; }
if [[ " ${EXTRA[*]} " == *draft-mtp* && ! -f "$MTP" ]]; then
  echo "missing MTP head: $MTP -- run ./download-mtp.sh" >&2; exit 1
fi

echo "[serve] build=$BUILD config=$LABEL mmap=$USE_MMAP nmax=$NMAX ctx=$CTX alias=$ALIAS host=$HOST:$PORT auth=$([ -n "$API_KEY" ] && echo yes || echo no)"
exec "$BIN" \
  -m "$MODEL" \
  -ngl 999 \
  -ot "per_layer_token_embd.weight=CPU" \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c "$CTX" \
  -a "$ALIAS" \
  "${LOADMODE[@]}" \
  "${EXTRA[@]}" \
  ${API_KEY:+--api-key "$API_KEY"} \
  --host "$HOST" --port "$PORT"
