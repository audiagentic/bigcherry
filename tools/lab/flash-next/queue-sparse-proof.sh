#!/bin/bash
# QFP25 / 1334 proof run: build the production set + 1334, then
#   1. backend correctness vs CPU with the activation marker (fa-sparse-backend-test.sh);
#   2. fidelity at 24K depth with the CPU f32 reference (flash-fidelity.sh ref): is sparse as close to f32 as dense?
#   3. fidelity at the long depth, dense vs sparse vs the dense floor (no CPU reference possible there).
# Usage: queue-sparse-proof.sh <tag> <long depth> [wait=<log with ALL_JOBS_DONE>]
set -u
TAG=${1:?tag}; LONG=${2:?long depth}; shift 2
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_JOBS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 UB=512 B=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-sparsefa-$TAG
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi sparse-fa gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT sparseproof-$TAG-backend tools/lab/flash-next/fa-sparse-backend-test.sh @$RUN $R/sparseproof-$TAG-backend
VIS=0,1,2,3 SCRIPT sparseproof-$TAG-fid24k tools/lab/flash-next/flash-fidelity.sh 24576 @$RUN $R/sparseproof-$TAG-fid24k ref BIGCHERRY_FA_SPARSE=1
VIS=0,1,2,3 SCRIPT sparseproof-$TAG-fid$LONG tools/lab/flash-next/flash-fidelity.sh $LONG @$RUN $R/sparseproof-$TAG-fid$LONG noref BIGCHERRY_FA_SPARSE=1
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== backend test"; grep -E "^sparse=|tests passed|FAIL|BUILD_FAILED" $R/sparseproof-$TAG-backend.log
for d in 24k $LONG; do echo "== fidelity $d"; grep -E "^D:|^S:|^D2:|^C:| vs |SERVER_FAILED" $R/sparseproof-$TAG-fid$d.log; done
echo ALL_JOBS_DONE
