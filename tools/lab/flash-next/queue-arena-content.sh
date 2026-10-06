#!/bin/bash
# MSM03 step 2: what fills the tensor split's shared compute arena, and how much of it scales with the context.
# Builds the production set + 1329 (largest tensors of a reserved graph) + 1331 (live set at the arena's peak) and
# loads Flash-Next production at each context in CTX_LIST with one short request, then prints the trace lines.
# ARENA_ENV adds variables to the load (e.g. BIGCHERRY_QSA_CHUNK=128), SUFFIX names that variant.
# Usage: queue-arena-content.sh <tag>      env: CTX_LIST ("49152 245760"), RUN_OVERRIDE, ARENA_ENV, SUFFIX
set -u
TAG=${1:?tag}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=32
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
RUN=${RUN_OVERRIDE:-b-arenac-$TAG}
N=arenac-$TAG${SUFFIX:-}
cat > "$R/$N.arms.sh" <<ARMS
#!/bin/bash
# written by queue-arena-content.sh: <llama-server> <out-root>
bash tools/lab/flash-next/wait-gpus-free.sh   # a production model may have loaded while the build ran
for ctx in ${CTX_LIST:-49152 245760}; do
  env CTX=\$ctx DEPTH=2048 BIGCHERRY_ALLOC_PEAK=24 BIGCHERRY_ALLOC_TOP=24 ${ARENA_ENV:-} bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/c\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
done
ARMS
jobs=$(mktemp)
: > "$jobs"
[ -n "${RUN_OVERRIDE:-}" ] || echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi deploy-v6-alloc-peak gfx1100,gfx1201,gfx1030" > "$jobs"
echo "VIS=0,1,2,3 SCRIPT $N $R/$N.arms.sh @$RUN $R/$N" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "$R/$N"/*/; do
  echo "== $(basename "$d"): compute buffer sizes"
  grep -hE "compute buffer size" "$d"*.server.log | sed -E 's/^[0-9.]+ I //' | sort -u | head -6
  echo "== $(basename "$d"): largest graph by peak (1331), then its largest tensors (1329)"
  grep -hE "ALLOC_PEAK buf=0 +op |ALLOC_PEAK buf=0 +[0-9]+ +[0-9.]+ MiB" "$d"*.server.log | sed -E 's/^[0-9.]+ [IWE] //' | awk '$5 + 0 >= 4 || $3 == "op"' | awk '!seen[$0]++' | head -40
done
echo ALL_RUNS_DONE
