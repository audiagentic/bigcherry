#!/bin/bash
# PRBE52 / 1268 adaptive MTP depth on the b11474 pin: build the production set + 1210 + 1255 + 1268 (experiment
# adaptive-mtp), then on Flash-Next production (2x XTX + R9700 tensor split, MTP draft on the 6900 XT), per depth:
#   1. adaptive OFF equals the production build: ABBA of an existing production build run vs the new binary with the
#      adaptive floor unset (greedy text identity, decode ms/step, acceptance) - also the cost of 1210 + 1255 alone;
#   2. adaptive ON vs OFF on the one binary: ABBA with LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE=<floor>, both arms at
#      --spec-draft-n-max <max> (greedy identity is the correctness gate; effective decode t/s is the result).
# wait= waits for ALL_RUNS_DONE (the end of a whole chain): a chain log also carries one ALL_JOBS_DONE per sub-run.
# Usage: queue-adaptive-mtp.sh <tag> <production build run> <depth>... [wait=<chain log with ALL_RUNS_DONE>]
#        env: SPEC_MAX (4), FLOOR (1)
set -u
TAG=${1:?tag}; PROD=${2:?production build run}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_RUNS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext SPEC_N=${SPEC_MAX:-4} BIGCHERRY_PATCH_TRACE=1
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-adaptmtp-$TAG
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi adaptive-mtp gfx1100,gfx1201,gfx1030" > "$jobs"
for d in "${depths[@]}"; do
  echo "VIS=0,1,2,3 SCRIPT adaptmtp-$TAG-off-d$d tools/lab/flash-next/flash-prefill-ab.sh $d @$PROD @$RUN $R/adaptmtp-$TAG-off-d$d" >> "$jobs"
  echo "VIS=0,1,2,3 SCRIPT adaptmtp-$TAG-on-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/adaptmtp-$TAG-on-d$d LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE=${FLOOR:-1}" >> "$jobs"
done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "${depths[@]}"; do
  echo "== depth $d: adaptive off, A = $PROD (production), B = $RUN"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/adaptmtp-$TAG-off-d$d.log
  echo "== depth $d: A = fixed depth ${SPEC_MAX:-4}, B = adaptive floor ${FLOOR:-1} (same binary)"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/adaptmtp-$TAG-on-d$d.log
  echo "   activation: $(grep -rhc "BIGCHERRY_PATCH_HIT patch=1268" $R/adaptmtp-$TAG-on-d$d/*-B/*.server.log 2>/dev/null | paste -sd+ | bc) marker line(s) in B, $(grep -rhc "BIGCHERRY_PATCH_HIT patch=1268" $R/adaptmtp-$TAG-on-d$d/*-A/*.server.log 2>/dev/null | paste -sd+ | bc) in A; depth changes in B: $(grep -rh "event=depth_change" $R/adaptmtp-$TAG-on-d$d/*-B/*.server.log 2>/dev/null | wc -l)"
done
echo ALL_JOBS_DONE
