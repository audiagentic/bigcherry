#!/bin/bash
# QFP17: which ops own the compute-buffer growth from -ub 512 to 1024 at 240K f16 (v6 + 1329 BIGCHERRY_ALLOC_TOP=25).
# ub1024 OOMs after the reserve logs are written. Waits for the CPU expert sweep.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 DECODE_N=64
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1
export BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1 BIGCHERRY_QSA_HOST_REMAP=1 BIGCHERRY_ALLOC_TOP=25
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-expert-cpu.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-alloc-top bigcherry:stock:linux-multi deploy-v6-alloc-trace gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT alloc-top tools/lab/flash-next/ub-sweep.sh @b-alloc-top $R/flashnext-alloc-top 512 1024
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in $R/flashnext-alloc-top/ub512-1 $R/flashnext-alloc-top/ub1024-2; do
  echo "== $d"
  # the target graph is the reserved graph with the most nodes; print its block
  awk '/BIGCHERRY_ALLOC_TOP graph n_nodes=/{n=$0; sub(/.*n_nodes=/,"",n); n=n+0; blk=(n>best)?1:0; if(blk){best=n; out=""}} blk && /BIGCHERRY_ALLOC_TOP/{out=out $0 "\n"} END{printf "%s", out}' $d/timing.server.log | sed -E 's/^[0-9.]+ W //' | head -45
done
echo ALL_JOBS_DONE
