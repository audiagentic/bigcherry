#!/bin/bash
# Fidelity gate for a change that is not bit-exact (attention path, mask construction, wire format): next-token
# distributions of natural-text continuations over one cached fill at DEPTH, without MTP, for
#   D  = caller env (the current GPU path)
#   S  = caller env + the given extra env (the changed path)
#   D2 = caller env again (run-to-run floor)
#   C  = CPU f32 reference (only with `ref`; slow, use a depth the CPU can prefill, e.g. 24576)
# probes-compare.py reports every arm against the reference (C when present, else D). The GPU path is itself an
# approximation of f32 (f16 KV, Q8_1 activations, BF16 collective wire), so the bar is: S is no further from the
# reference than D is, and S-vs-D is of the same order as D-vs-C - not bit identity.
# Usage: flash-fidelity.sh <depth> <llama-server> <out-root> <ref|noref> <VAR=value>...    (PROBES from the environment)
set -u
depth=$1 bin=$2 root=$3 ref=$4; shift 4
here="$(cd "$(dirname "$0")" && pwd)"
run() {  # <arm dir> [extra env...]
  local dir=$1; shift
  env "$@" PROBES=${PROBES:-24} UB=${UB:-512} B=${UB:-512} DEPTH=$depth \
      bash "$here/long-ctx-profile.sh" "$bin" "$root/$dir" probes 2>&1 | grep -E "^probes:|SERVER_FAILED"
}
echo "D:  $(run D NO_MTP=1 | tr '\n' ' ')"
echo "S:  $(run S NO_MTP=1 "$@" | tr '\n' ' ')"
echo "D2: $(run D2 NO_MTP=1 | tr '\n' ' ')"
p="probes.$depth.probes.json"
set -- "D=$root/D/$p" "D2=$root/D2/$p" "S=$root/S/$p"
if [ "$ref" = ref ]; then
  echo "C:  $(run C CPU_REF=1 CTX=${REF_CTX:-49152} | tr '\n' ' ')"
  [ -f "$root/C/$p" ] && set -- "C=$root/C/$p" "$@"
fi
python3 "$here/probes-compare.py" "$@"
