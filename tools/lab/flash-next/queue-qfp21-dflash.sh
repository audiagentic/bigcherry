#!/bin/bash
# QFP21 DFlash tuning: Qwen3.8-27B dual-XTX tensor split + 1286, DFlash2 Q8/Q4_K_M n_max 4-10 on the R9700 and 6900 vs MTP5, 3 timed requests per arm.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-qfp21.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-r1286 bigcherry:stock:linux-multi retest-1286 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT q21-dflash tools/lab/dflash/probe-27b.sh @b-r1286 $R/qfp21-dflash/out (plain|mtp5|q21-dflash-.*) 3
JOBS
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^[a-z0-9-]+: (prompt|SERVER_FAILED)" $R/q21-dflash.log
md5sum $R/qfp21-dflash/out/*.greedy.txt | awk '{print $1}' | sort | uniq -c
echo ALL_JOBS_DONE
