#!/usr/bin/env bash
# Start/stop either inference stack for Qwen3.8-Flash-Next, one at a time.
#
#   ./stack.sh start llamacpp   # fork Unsloth + MTP, port 18080  (recommended)
#   VISION=0 ./stack.sh start llamacpp   # text-only; by default images work once ./download-mmproj.sh ran
#   ./stack.sh start vllm       # vLLM NVFP4 container, port 18300
#   ./stack.sh stop
#   ./stack.sh status
#
# Refuses to start when memory is short or Ollama is holding a model: the two
# stacks need 88-112 GiB of the 121.7 GiB unified pool, and loading one next to
# a resident Ollama model is what caused the NVRM stall -> watchdog panic on
# 2026-09-01 (Decision #4).
set -uo pipefail
cd "$(dirname "$0")"

# Same HOST contract as serve.sh, so health checks probe whatever we bound to.
HOST="${HOST:-127.0.0.1}"
if [ "$HOST" = "tailscale" ]; then
  HOST=$(tailscale ip -4 2>/dev/null | head -1)
  [ -n "$HOST" ] || { echo "cannot resolve the tailscale IP" >&2; exit 1; }
fi
export HOST
API_KEY="${API_KEY:-}"
export API_KEY
# Health checks must present the key too, or they 401 and we wait out the timeout
# on a server that is actually up.
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
LC_PORT=18080
VL_PORT=18300
VL_NAME=qwen38-flash
NEED_LC=95      # GiB available required (88 resident + margin)
NEED_VL=118     # vLLM claims ~112 GiB

avail() { awk '/^MemAvailable:/{printf "%.0f",$2/1048576}' /proc/meminfo; }
used_gib() { awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^Buffers:/{b=$2}/^Cached:/{c=$2}/^SReclaimable:/{s=$2}END{printf "%.0f",(t-f-b-c-s)/1048576}' /proc/meminfo; }
lc_pid() { pgrep -x llama-server | head -1; }
vl_up()  { docker ps -q --filter "name=$VL_NAME" | grep -q .; }

ollama_models() {
  # `grep -c` already prints 0 and exits 1, so `|| echo 0` would emit a second
  # line and break the arithmetic. Count with wc -l on non-blank lines instead.
  local n=0 h c
  for h in 127.0.0.1:11434 127.0.0.1:11435; do
    c=$(OLLAMA_HOST=$h ollama ps 2>/dev/null | tail -n +2 | grep -c '[^[:space:]]')
    n=$(( n + ${c:-0} ))
  done
  echo "$n"
}

preflight() {
  local need=$1 a m
  if [ -n "$(lc_pid)" ] || vl_up; then
    echo "!! stack lain masih jalan -- jalankan ./stack.sh stop dulu"; return 1
  fi
  m=$(ollama_models)
  if [ "$m" -gt 0 ]; then
    echo "!! Ollama sedang memegang $m model. Lepaskan dulu:"
    echo "     ollama stop <nama>    atau    sudo systemctl stop ollama"
    return 1
  fi
  a=$(avail)
  if [ "$a" -lt "$need" ]; then
    echo "!! available cuma ${a} GiB, butuh >= ${need} GiB"; return 1
  fi
  echo ">> preflight ok (available ${a} GiB, Ollama tidak memegang model)"
}

wait_ready() {
  local port=$1 max=$2 t=0
  while [ "$t" -lt "$max" ]; do
    curl -sf "${AUTH[@]}" "http://$HOST:$port/v1/models" >/dev/null 2>&1 && return 0
    sleep 5; t=$((t+5))
  done
  return 1
}

case "${1:-status}" in
start)
  case "${2:-llamacpp}" in
  llamacpp)
    preflight "$NEED_LC" || exit 1
    echo ">> starting llama.cpp fork + MTP (port $LC_PORT), ~45s"
    # -ub 256: the CUDA backend aborts in cublasGemmEx on some prefill batch
    # sizes (367 and 512 seen in a 300-600 sweep; upstream is affected too).
    # Capping the physical batch at 256 avoided every size in 1-600. See
    # OPTIMIZATION.md "Crash CUDA pada ukuran batch tertentu".
    # Vision projector: on whenever models/mmproj is present (./download-mmproj.sh),
    # VISION=0 forces text-only. Costs ~1 GiB; MTP is unaffected (OPTIMIZATION.md 2026-09-05).
    VIS=(); [ "${VISION:-auto}" != 0 ] && [ -f models/mmproj/mmproj-BF16.gguf ] && VIS=(--vision)
    # NMAX / PMIN override the measured defaults (3 / 0.75) for speculative sweeps.
    UBATCH="${UBATCH:-256}" BUILD=fork nohup ./serve.sh --mtp --nmax "${NMAX:-3}" --pmin "${PMIN:-0.75}" "${VIS[@]}" > runs/serve-current.log 2>&1 &
    if wait_ready "$LC_PORT" 300; then
      echo ">> READY  http://$HOST:$LC_PORT/v1  (used $(used_gib) GiB, available $(avail) GiB)"
    else
      echo "!! gagal siap; lihat runs/serve-current.log"; tail -12 runs/serve-current.log; exit 1
    fi ;;
  vllm)
    preflight "$NEED_VL" || exit 1
    echo ">> starting vLLM NVFP4 (port $VL_PORT), ~14 menit"
    nohup vllm-dgx/scripts/serve.sh > runs/serve-current.log 2>&1 &
    if wait_ready "$VL_PORT" 1500; then
      echo ">> READY  http://$HOST:$VL_PORT/v1  (used $(used_gib) GiB, available $(avail) GiB)"
    else
      echo "!! gagal siap"; docker logs "$VL_NAME" 2>&1 | tail -15; exit 1
    fi ;;
  *) echo "usage: $0 start [llamacpp|vllm]"; exit 2 ;;
  esac ;;
stop)
  p=$(lc_pid); [ -n "$p" ] && { kill "$p"; echo ">> llama-server ($p) dihentikan"; }
  vl_up && { docker rm -f "$VL_NAME" >/dev/null; echo ">> container $VL_NAME dihapus"; }
  for _ in $(seq 1 40); do [ -z "$(lc_pid)" ] && ! vl_up && break; sleep 3; done
  sleep 2; echo ">> available sekarang $(avail) GiB" ;;
status)
  p=$(lc_pid)
  if [ -n "$p" ]; then echo "llama.cpp : JALAN (pid $p, port $LC_PORT)"; else echo "llama.cpp : mati"; fi
  if vl_up; then echo "vLLM      : JALAN (port $VL_PORT)"; else echo "vLLM      : mati"; fi
  echo "Ollama    : $(ollama_models) model ter-load"
  echo "memori    : available $(avail) GiB dari 121 GiB" ;;
*) echo "usage: $0 {start [llamacpp|vllm]|stop|status}"; exit 2 ;;
esac
