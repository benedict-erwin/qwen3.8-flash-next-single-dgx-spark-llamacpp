#!/usr/bin/env bash
# nsys decode capture on the current local profile (stock model), then the same two requests without nsys.
set -uo pipefail
cd "$(dirname "$0")/../.."; O=runs/profile-2026-11-14; S=qwenprof
export API_KEY=$(sed -n 's/^API_KEY=//p' stack.local | tail -1)
./stack.sh stop >/dev/null 2>&1; sleep 5
env "LAUNCHER=nsys launch --session=$S -t cuda --cuda-graph-trace=node" ./stack.sh start llamacpp 2>&1 | tail -2
grep -oE "nmax=[0-9]+ pmin=[0-9.]+ ctx=[0-9]+" runs/serve-current.log | head -1
for k in prose code; do
  LABEL=nov14-nsys python3 $O/prof2.py cache $k >/dev/null
  nsys start --session=$S -o $O/nov14-$k --force-overwrite=true
  LABEL=nov14-nsys python3 $O/prof2.py run $k | tee -a $O/timings.jsonl
  nsys stop --session=$S
done
./stack.sh stop >/dev/null 2>&1; sleep 8
for k in prose code; do nsys export --type sqlite --force-overwrite=true -o $O/nov14-$k.sqlite $O/nov14-$k.nsys-rep >/dev/null 2>&1; done
ls -la $O/nov14-*
./stack.sh start llamacpp 2>&1 | tail -1
for k in prose code; do LABEL=nov14-plain python3 $O/prof2.py cache $k >/dev/null; LABEL=nov14-plain python3 $O/prof2.py run $k | tee -a $O/timings.jsonl; done
./stack.sh stop >/dev/null 2>&1
echo "=== done $(date +%H:%M:%S)"
