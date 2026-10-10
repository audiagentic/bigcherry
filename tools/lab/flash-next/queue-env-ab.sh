#!/bin/bash
# Generic Flash-Next production ABBA on an existing build: A = production environment, B = the same plus AB_ENV
# (flash-prefill-env-ab.sh per depth: prefill t/s, decode t/s, acceptance, greedy md5 per arm). For flags that must
# not change the result (memory layout, diagnostics) the two md5 lines must be equal.
# FIDELITY=1 adds the probe comparison (flash-fidelity.sh) at the first depth. FIDELITY_REF=ref also compares both arms
# with the CPU f32 reference.
# Usage: AB_ENV="VAR=value ..." queue-env-ab.sh <name> <build run id> <depth>...     env: CTX (245760), BC_MODEL, FIDELITY
set -u
NAME=${1:?name}; RUN=${2:?build run id}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=$(bash tools/lab/plan-qualification/work-root.sh "$PWD")/runs
mkdir -p "$R"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=${CTX:-245760} TS=${TS:-0.31,0.27,0.42} UB=512 B=512 DECODE_N=${DECODE_N:-512}
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
for d in "$@"; do
  echo "VIS=0,1,2,3 SCRIPT $NAME-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/$NAME-d$d ${AB_ENV:?AB_ENV}" >> "$jobs"
done
[ "${FIDELITY:-0}" = 1 ] && echo "VIS=0,1,2,3 SCRIPT $NAME-fid tools/lab/flash-next/flash-fidelity.sh $1 @$RUN $R/$NAME-fid ${FIDELITY_REF:-noref} $AB_ENV" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "$@"; do
  echo "== depth $d: A = production, B = $AB_ENV"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/$NAME-d$d.log
done
[ "${FIDELITY:-0}" = 1 ] && { echo "== fidelity at depth $1: D = production, S = with $AB_ENV, D2 = production again"; grep -E " vs |SERVER_FAILED" $R/$NAME-fid.log; }
echo ALL_RUNS_DONE
