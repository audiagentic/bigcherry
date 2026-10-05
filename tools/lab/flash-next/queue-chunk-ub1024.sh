#!/bin/bash
# QFP17 / 1332 on top of sparse flash attention (1334): build production + 1332 + 1334, then per depth a prefill ABBA
# of ub512 (A) vs ub1024 with BIGCHERRY_QSA_CHUNK=256 (B), both with BIGCHERRY_FA_SPARSE as given, and a fidelity
# gate (flash-fidelity.sh) of ub1024 + chunk 256 against ub512 at the first depth.
# Usage: queue-chunk-ub1024.sh <tag> <sparse 0|1> <depth>... [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; SPARSE=${2:?sparse 0|1}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext BIGCHERRY_FA_SPARSE=$SPARSE
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-chunk-$TAG
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi sparse-fa-chunk gfx1100,gfx1201,gfx1030" > "$jobs"
for d in "${depths[@]}"; do
  echo "VIS=0,1,2,3 SCRIPT chunk-$TAG-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/chunk-$TAG-d$d UB=1024 B=1024 BIGCHERRY_QSA_CHUNK=256" >> "$jobs"
done
echo "VIS=0,1,2,3 SCRIPT chunk-$TAG-fid tools/lab/flash-next/flash-fidelity.sh ${depths[0]} @$RUN $R/chunk-$TAG-fid noref UB=1024 B=1024 BIGCHERRY_QSA_CHUNK=256" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "${depths[@]}"; do grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/chunk-$TAG-d$d.log; done
echo "== fidelity: ub1024 + chunk 256 (S) vs ub512 (D)"; grep -E "^D:|^S:|^D2:| vs |SERVER_FAILED" $R/chunk-$TAG-fid.log
echo ALL_JOBS_DONE
