#!/bin/bash
# Quick screen (~10 min): ABA decode at one depth (default ~32K cached), 256 tokens, MTP3, f16 draft KV,
# 192K deployment config. Ideas that move ms/step here get the full ABBA + parity run; the rest are dropped.
# Usage: quick-ab.sh <base llama-server> <new llama-server> <out-root> [new-arm env, e.g. BIGCHERRY_X=1]
#   Same binary for both arms + an env string screens env-gated options.
set -u
base=$1 new=$2 root=$3 newenv=${4:-}
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export CTKD=f16 CTVD=f16 DECODE_N=256 DEPTH=${QUICK_DEPTH:-24576}
for arm in base-a new base-b; do
  bin=$base; envs=; [ $arm = new ] && { bin=$new; envs=$newenv; }
  out=$(env $envs bash "$s" "$bin" "$root/$arm" timing 2>&1 | grep -E "^timing: prompt|SERVER_FAILED")
  echo "$arm ${envs:-} $out" | python3 -c "
import re, sys
l = sys.stdin.read().strip()
m = re.search(r'decode ([\d.]+) t/s, accepted (\d+)/(\d+)', l)
if m:
    tps, acc = float(m.group(1)), int(m.group(2))
    print(f'{l.split()[0]}: {tps} t/s, accepted {acc}/{m.group(3)}, {256/tps/(256-acc)*1e3:.1f} ms/step')
else:
    print(l)"
done
