#!/bin/bash
# Rerun of the adaptive max-context search with the free-VRAM rebalance heuristic, aimed at keeping prefill
# (ub256 costs ~28% prefill vs ub512 and is not a RAM spill: prefill is flat across context at ub256).
# Usage: maxctx-search-2.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
py=$(cd "$(dirname "$0")" && pwd)/maxctx-search.py
python3 "$py" "$bin" "$root/k16-v8-ub512" f16 q8_0 512 131072 196608
python3 "$py" "$bin" "$root/k16-v8-ub384" f16 q8_0 384 139264 196608
