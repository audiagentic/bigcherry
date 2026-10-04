#!/bin/bash
# Default-on check for hip-q81 + sched-async: Qwen3.8-27B Q8_0 dual-XTX (MTP4), Qwen3.6-35B-A3B IQ3_S single XTX, Qwen3.5-4B Q6_K single XTX; flags off vs on, same release binary.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-qfp21-dflash.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-release2 bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT don-27b tools/lab/default-on/xmodel-ab.sh @b-release2 $R/default-on-27b hip-q81,sched-async /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf -sm tensor -ts 1,1 -ub 512 -b 2048 --spec-type draft-mtp --spec-draft-n-max 4
VIS=0 SCRIPT don-35b tools/lab/default-on/xmodel-ab.sh @b-release2 $R/default-on-35b hip-q81,sched-async /mnt/data/llm-models/qwen3.6-35B-A3B/gguf/Qwen3.6-35B-A3B-UD-IQ3_S.gguf -ub 512 -b 2048
VIS=0 SCRIPT don-4b tools/lab/default-on/xmodel-ab.sh @b-release2 $R/default-on-4b hip-q81,sched-async /mnt/data/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf -ub 512 -b 2048
JOBS
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for m in 27b 35b 4b; do echo "== $m"; grep -E "^d[0-9]|^ +[0-9]" $R/don-$m.log; done
echo ALL_JOBS_DONE
