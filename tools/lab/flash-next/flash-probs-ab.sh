#!/bin/bash
# Output-distribution check of a prefill/attention change on ONE binary, without MTP (single-token greedy decode,
# so batch-shape near-ties from speculative verify cannot interfere): A1 = caller env, B = caller env + the given
# extra env, A2 = caller env again. Each arm fills the same prompt at DEPTH uncached-then-cached and decodes N
# tokens with n_probs; probs-compare.py then reports A1 vs A2 (the run-to-run floor) and A1 vs B.
# Pass: B agrees with A1 over the whole prefix with max |dp| in the range of the A1-vs-A2 floor, or diverges only
# where both runs show a near-tie (small top-1/top-2 margin).
# Usage: flash-probs-ab.sh <depth> <llama-server> <out-root> <VAR=value>...   (N from PROBS_N, default 32)
set -u
depth=$1 bin=$2 root=$3; shift 3
here="$(cd "$(dirname "$0")" && pwd)"
n=${PROBS_N:-32}
run() {  # <arm dir> [extra env...]
  local dir=$1; shift
  env "$@" NO_MTP=1 REPEAT=1 PROBS_KEEP=$n DECODE_N=$n UB=${UB:-512} B=${UB:-512} DEPTH=$depth \
      bash "$here/long-ctx-profile.sh" "$bin" "$root/$dir" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
}
echo "A1: $(run A1 | tr '\n' ' ')"
echo "B:  $(run B "$@" | tr '\n' ' ')"
echo "A2: $(run A2 | tr '\n' ' ')"
p="timing.$depth.probs.json"
if [ -f "$root/A1/$p" ] && [ -f "$root/A2/$p" ] && [ -f "$root/B/$p" ]; then
  python3 "$here/probs-compare.py" "d$depth floor A1-vs-A2" "$root/A1/$p" "$root/A2/$p"
  python3 "$here/probs-compare.py" "d$depth change A1-vs-B" "$root/A1/$p" "$root/B/$p"
  echo "text md5: A1 $(md5sum < "$root/A1/timing.$depth.greedy.txt" | cut -c1-12) A2 $(md5sum < "$root/A2/timing.$depth.greedy.txt" | cut -c1-12) B $(md5sum < "$root/B/timing.$depth.greedy.txt" | cut -c1-12)"
else
  echo "d$depth: PROBS_MISSING"
fi
