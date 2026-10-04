#!/bin/bash
# QFP21: DFlash vs MTP5 at 16K/64K on the 27B (R9700 drafter) + 27B default-on re-run with devices pinned (after chunk-prof).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-chunk-prof.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-r1286c bigcherry:stock:linux-multi retest-1286 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-release3 bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT dflash-depth tools/lab/dflash/dflash-depth.sh @b-r1286c $R/qfp21-dflash-depth
VIS=0,1,2,3 SCRIPT don3-27b tools/lab/default-on/xmodel-ab.sh @b-release3 $R/default-on3-27b hip-q81,sched-async /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -ub 512 -b 2048 --spec-type draft-mtp --spec-draft-n-max 4
JOBS
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== dflash depth"; grep -E "^d[0-9]+ [0-9]|^d[0-9]+ +[0-9]+ " $R/dflash-depth.log
echo "== default-on 27B"; grep -E "^d[0-9]|^ +[0-9]" $R/don3-27b.log
echo ALL_JOBS_DONE
