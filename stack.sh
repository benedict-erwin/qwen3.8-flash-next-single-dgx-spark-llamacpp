#!/usr/bin/env bash
# Start/stop either inference stack for Qwen3.8-Flash-Next, one at a time.
#
#   ./stack.sh start llamacpp   # fork Unsloth + MTP, port 18080  (recommended)
#   VISION=0 ./stack.sh start llamacpp   # text-only; by default images work once ./download-mmproj.sh ran
#   ./stack.sh start vllm       # vLLM NVFP4 container, port 18300
#   ./stack.sh start llamacpp --clock-cap   # same, with the GPU clock capped at 2200 MHz (sudo)
#   ./stack.sh stop
#   ./stack.sh status
#   ./stack.sh help             # every command, flag and env var, with what each needs
#
# Refuses to start when memory is short or Ollama is holding a model: the two
# stacks need 88-112 GiB of the 121.7 GiB unified pool, and loading one next to
# a resident Ollama model is what caused the NVRM stall -> watchdog panic on
# 2026-09-01 (Decision #4).
set -uo pipefail
cd "$(dirname "$0")"

# Personal defaults: plain KEY=value lines in ./stack.local (gitignored). A variable
# already set in the environment wins, so `CTX=131072 ./stack.sh start` still
# overrides the file. Anything serve.sh or llama-server reads from the environment
# can go here (FORK, CTX, UBATCH, LLAMA_ARG_N_PARALLEL, API_KEY, HOST, ...).
LOCAL_VARS=()
if [ -f stack.local ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in *[!A-Za-z0-9_]*|'') echo "!! stack.local: cannot parse '$line'" >&2; exit 2 ;; esac
    v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
    if [ -z "${!k+x}" ]; then export "$k=$v"; LOCAL_VARS+=("$k=$v"); fi
  done < stack.local
fi

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
CAP_DEFAULT=2200  # MHz; the knee measured by agjs/gb10-clock-cap and tonyd2wild (see help)

avail() { awk '/^MemAvailable:/{printf "%.0f",$2/1048576}' /proc/meminfo; }
used_gib() { awk '/^MemTotal:/{t=$2}/^MemFree:/{f=$2}/^Buffers:/{b=$2}/^Cached:/{c=$2}/^SReclaimable:/{s=$2}END{printf "%.0f",(t-f-b-c-s)/1048576}' /proc/meminfo; }
# Only OUR llama-server: the binary name at the start of the command line, bound to
# LC_PORT. A bare `pgrep -x llama-server` also matched the llama-server processes
# Ollama spawns for its own models (other ports), so status reported "running" and
# stop killed an Ollama worker while our server was not even up.
lc_pid() { pgrep -f -- "^[^ ]*/llama-server .*--port $LC_PORT( |$)" | head -1; }
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

# One line of GPU state for status: current clock, power on the GPU rail, temperature,
# and whether a clock cap is currently limiting it. nvidia-smi only flags the cap
# ("Applications Clocks Setting", event-reason bit 0x2) while the GPU is under load
# and the cap is what limits it; at idle a cap is invisible from userspace, so
# status says "idle, cap not observable" instead of claiming either way.
gpu_line() {
  command -v nvidia-smi >/dev/null || { echo "n/a (nvidia-smi missing)"; return; }
  local q cur max pw tc reason
  q=$(nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null) || { echo "n/a"; return; }
  IFS=, read -r cur max pw tc reason <<<"$q"
  cur=${cur// /}; max=${max// /}; pw=${pw// /}; tc=${tc// /}; reason=${reason// /}
  # bit 0x2 = applications clocks setting (what -lgc / -ac report while limiting)
  if [ $(( reason & 2 )) -ne 0 ]; then
    echo "${cur} MHz (max ${max}), ${pw} W, ${tc} C -- clock cap active"
  elif [ $(( reason & 1 )) -ne 0 ]; then
    echo "${cur} MHz (max ${max}), ${pw} W, ${tc} C -- idle, cap not observable"
  else
    echo "${cur} MHz (max ${max}), ${pw} W, ${tc} C"
  fi
}

check_clock_cap() {
  local mhz=$1
  case "$mhz" in
    ''|*[!0-9]*) echo "!! --clock-cap wants a whole number of MHz, got '$mhz'"; return 1 ;;
  esac
  if [ "$mhz" -lt 1000 ] || [ "$mhz" -gt 3003 ]; then
    echo "!! --clock-cap=$mhz is outside 1000-3003 MHz (GB10 max clock is 3003)"; return 1
  fi
}

apply_clock_cap() {
  local mhz=$1
  echo ">> capping the GPU clock at ${mhz} MHz (sudo nvidia-smi -lgc 0,${mhz}; password prompt follows)"
  sudo nvidia-smi -lgc 0,"$mhz" >/dev/null || { echo "!! clock cap failed; nothing started"; return 1; }
  echo ">> cap set; it does not survive a reboot. Undo: ./stack.sh stop --clock-reset"
}

reset_clock_cap() {
  echo ">> restoring the default GPU clocks (sudo nvidia-smi -rgc; password prompt follows)"
  sudo nvidia-smi -rgc >/dev/null && echo ">> clocks restored" || echo "!! clock reset failed"
}

preflight() {
  local need=$1 a m
  if [ -n "$(lc_pid)" ] || vl_up; then
    echo "!! another stack is still running -- run ./stack.sh stop first"; return 1
  fi
  m=$(ollama_models)
  if [ "$m" -gt 0 ]; then
    echo "!! Ollama is holding $m model(s). Release them first:"
    echo "     ollama stop <name>    or    sudo systemctl stop ollama"
    return 1
  fi
  a=$(avail)
  if [ "$a" -lt "$need" ]; then
    echo "!! only ${a} GiB available, need >= ${need} GiB"; return 1
  fi
  echo ">> preflight ok (${a} GiB available, Ollama holds no model)"
}

wait_ready() {
  local port=$1 max=$2 t=0
  while [ "$t" -lt "$max" ]; do
    curl -sf "${AUTH[@]}" "http://$HOST:$port/v1/models" >/dev/null 2>&1 && return 0
    sleep 5; t=$((t+5))
  done
  return 1
}

usage() { echo "usage: $0 {start [llamacpp|vllm] [--clock-cap[=MHZ]] | stop [--clock-reset] | status | help}"; }

help() {
  cat <<HELP
$(usage)

Commands
  start [llamacpp|vllm]  Start one backend (default llamacpp). Runs a preflight first and
                         refuses when the other backend is up, when Ollama holds a model, or
                         when less than 95 GiB (llamacpp) / 118 GiB (vllm) is available.
                         Needs: the model files from download-parts.sh (+ download-mtp.sh),
                         the fork from download-fork.sh; for vllm the vllm-dgx container.
  stop                   Stop whichever backend is running and wait until memory is back.
  status                 Which backend is up, Ollama's loaded models, free memory, GPU clock.
  help                   This text.

Flags
  --clock-cap[=MHZ]      (start) Cap the GPU clock before launching, default ${CAP_DEFAULT} MHz.
                         Decode on GB10 is memory-bandwidth bound, so the cap costs about
                         1% decode and 4-8% cold prefill while cutting GPU-rail power by
                         roughly a third and peak temperature by ~12 C (community
                         measurements, vLLM; not yet re-measured on this recipe).
                         Needs: sudo (you will be asked for your password). Does not survive
                         a reboot. Undo with 'stop --clock-reset' or 'sudo nvidia-smi -rgc'.
  --clock-reset          (stop) Restore the default GPU clocks after stopping. Needs: sudo.
  -h, --help             Same as 'help'.

Environment (start llamacpp; all optional, defaults are the measured profile)
  VISION=0               Text-only. Default: the vision projector loads whenever
                         models/mmproj/mmproj-BF16.gguf exists (./download-mmproj.sh).
  MTP=0                  Disable speculative decoding (diagnostics only, decode ~24 tok/s).
  NMAX=3  PMIN=0.50      Speculative draft length and confidence threshold.
  UBATCH=256             Physical batch; 256 avoids the CUDA prefill crash on the prebuilt fork.
  FORK=<name>            Which build under forks/ to run (unsloth-mixfix, unsloth-qsa, ...).
  CTX=131072             Context length passed to serve.sh (262144 = the model's native max).
  HOST=127.0.0.1         Bind address; 'tailscale' resolves the tailnet IP. API_KEY=<key>
                         turns on bearer auth; stack.sh presents it in its own health checks.
  LLAMA_ARG_N_PARALLEL=1 One slot: a strict one-request queue (see README).

Personal defaults      Put KEY=value lines in ./stack.local (gitignored, never committed)
                       and they apply to every start; an explicit environment variable
                       still wins. 'status' lists what the file contributed. Example:
                         FORK=unsloth-qsa
                         CTX=262144
                         UBATCH=512
                         LLAMA_ARG_N_PARALLEL=1
                         API_KEY=...        # chmod 600 stack.local if you keep the key here
HELP
}

# ---- argument parsing: subcommand first, then its positional + flags in any order ----
cmd="${1:-status}"; shift 2>/dev/null || true
BACKEND=""; CAP=""; RESET=0
for arg in "$@"; do
  case "$arg" in
    llamacpp|vllm)   BACKEND=$arg ;;
    --clock-cap)     CAP=$CAP_DEFAULT ;;
    --clock-cap=*)   CAP=${arg#--clock-cap=}; check_clock_cap "$CAP" || exit 2 ;;
    --clock-reset)   RESET=1 ;;
    -h|--help)       cmd=help ;;
    *) echo "!! unknown argument '$arg'"; usage; exit 2 ;;
  esac
done

case "$cmd" in
help|-h|--help) help ;;
start)
  [ -n "$CAP" ] && [ "$RESET" = 1 ] && { echo "!! --clock-reset belongs to stop"; exit 2; }
  case "${BACKEND:-llamacpp}" in
  llamacpp)
    preflight "$NEED_LC" || exit 1
    [ -z "$CAP" ] || apply_clock_cap "$CAP" || exit 1
    echo ">> starting llama.cpp fork + MTP (port $LC_PORT), ~45s"
    # -ub 256: the CUDA backend aborts in cublasGemmEx on some prefill batch
    # sizes (367 and 512 seen in a 300-600 sweep; upstream is affected too).
    # Capping the physical batch at 256 avoided every size in 1-600.
    # See OPTIMIZATION.md, 2026-09-04 (the CUDA crash during prefill).
    # Vision projector: on whenever models/mmproj is present (./download-mmproj.sh),
    # VISION=0 forces text-only. Costs ~1 GiB; MTP is unaffected (OPTIMIZATION.md 2026-09-05).
    VIS=(); [ "${VISION:-auto}" != 0 ] && [ -f models/mmproj/mmproj-BF16.gguf ] && VIS=(--vision)
    # NMAX / PMIN override the measured defaults (3 / 0.50) for speculative sweeps; MTP=0 disables
    # speculative decoding entirely (diagnostics only -- decode drops to ~24 tok/s).
    # p-min 0.50 has been the default since 2026-09-06: +10% decode over 0.75 on matched prompts,
    # same answers (the target verifies every draft). PMIN=0.75 restores the old behaviour.
    SPEC=(--mtp --nmax "${NMAX:-3}" --pmin "${PMIN:-0.50}"); [ "${MTP:-1}" = 0 ] && SPEC=()
    UBATCH="${UBATCH:-256}" BUILD=fork nohup ./serve.sh "${SPEC[@]}" "${VIS[@]}" > runs/serve-current.log 2>&1 &
    if wait_ready "$LC_PORT" 300; then
      echo ">> READY  http://$HOST:$LC_PORT/v1  (used $(used_gib) GiB, available $(avail) GiB)"
    else
      echo "!! not ready in time; see runs/serve-current.log"; tail -12 runs/serve-current.log; exit 1
    fi ;;
  vllm)
    preflight "$NEED_VL" || exit 1
    [ -z "$CAP" ] || apply_clock_cap "$CAP" || exit 1
    echo ">> starting vLLM NVFP4 (port $VL_PORT), ~14 min"
    nohup vllm-dgx/scripts/serve.sh > runs/serve-current.log 2>&1 &
    if wait_ready "$VL_PORT" 1500; then
      echo ">> READY  http://$HOST:$VL_PORT/v1  (used $(used_gib) GiB, available $(avail) GiB)"
    else
      echo "!! not ready in time"; docker logs "$VL_NAME" 2>&1 | tail -15; exit 1
    fi ;;
  esac ;;
stop)
  [ -n "$CAP" ] && { echo "!! --clock-cap belongs to start"; exit 2; }
  p=$(lc_pid); [ -n "$p" ] && { kill "$p"; echo ">> llama-server ($p) stopped"; }
  vl_up && { docker rm -f "$VL_NAME" >/dev/null; echo ">> container $VL_NAME removed"; }
  for _ in $(seq 1 40); do [ -z "$(lc_pid)" ] && ! vl_up && break; sleep 3; done
  sleep 2; echo ">> $(avail) GiB available now"
  [ "$RESET" = 1 ] && reset_clock_cap ;;
status)
  if [ -n "$CAP" ] || [ "$RESET" = 1 ]; then echo "!! status takes no flags"; exit 2; fi
  p=$(lc_pid)
  if [ -n "$p" ]; then echo "llama.cpp : running (pid $p, port $LC_PORT)"; else echo "llama.cpp : stopped"; fi
  if vl_up; then echo "vLLM      : running (port $VL_PORT)"; else echo "vLLM      : stopped"; fi
  echo "Ollama    : $(ollama_models) model(s) loaded"
  echo "memory    : $(avail) GiB available of 121"
  echo "GPU       : $(gpu_line)"
  if [ -f stack.local ]; then
    shown=(); for kv in "${LOCAL_VARS[@]}"; do case "$kv" in API_KEY=*) shown+=("API_KEY=<set>") ;; *) shown+=("$kv") ;; esac; done
    echo "local     : ${shown[*]:-(all overridden by the environment)}"
  fi ;;
*) usage; exit 2 ;;
esac
