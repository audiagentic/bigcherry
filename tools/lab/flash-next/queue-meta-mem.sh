#!/bin/bash
# MET04 / 1339: what each card of the tensor split really holds. Builds the production set + 1339 (experiment
# meta-mem) and, for every context in CTX_LIST, loads Flash-Next with BIGCHERRY_META_MEM=1 in two layouts and sums the
# report per device: P = production row split, O = owner's layout (dense + attention + KV on the XTXs, usage-placed
# experts, expert-parallel; OWNER_ENV). One short request per load (depth 2048) so the compute arenas exist.
# FLAG_ARMS=1 with EXPERIMENT=meta-memory adds the MSM02 / MSM03 arms (per-device arena, subset-mirrored indexer cache)
# on both layouts; their greedy text must equal their base arm.
# Usage: queue-meta-mem.sh <tag>      env: CTX_LIST ("49152 245760"), BC_MODEL, OWNER_ENV, RUN_OVERRIDE, EXPERIMENT, FLAG_ARMS
set -u
TAG=${1:?tag}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=${BC_MODEL:-/mnt/data/llm-models/qwen3.8-flash-next/gguf/placed-128-128-256/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=64
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 BIGCHERRY_FEATURES=flashnext
OWNER_ENV=${OWNER_ENV:-BIGCHERRY_MOE_EP=1 BIGCHERRY_MOE_EP_TS=132,132,248 TS=1,1,0}
docker stop radiance-vllm >/dev/null 2>&1
RUN=${RUN_OVERRIDE:-b-metamem-$TAG}
N=metamem-$TAG
cat > "$R/$N.arms.sh" <<ARMS
#!/bin/bash
# written by queue-meta-mem.sh: <llama-server> <out-root>
for ctx in ${CTX_LIST:-49152 245760}; do
  env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/P-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
  env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 $OWNER_ENV bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/O-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
  if [ "${FLAG_ARMS:-0}" = 1 ]; then   # MSM02 / MSM03: a = per-device arena (1340), m = subset-mirrored indexer cache (1341)
    env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 BIGCHERRY_META_PER_DEVICE_ARENA=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/Pa-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
    env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 BIGCHERRY_META_SUBSET_MIRROR=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/Pm-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
    env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 $OWNER_ENV BIGCHERRY_META_PER_DEVICE_ARENA=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/Oa-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
    env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 $OWNER_ENV BIGCHERRY_META_SUBSET_MIRROR=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/Om-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
    env CTX=\$ctx DEPTH=2048 BIGCHERRY_META_MEM=1 $OWNER_ENV BIGCHERRY_META_PER_DEVICE_ARENA=1 BIGCHERRY_META_SUBSET_MIRROR=1 bash tools/lab/flash-next/long-ctx-profile.sh "\$1" "\$2/Oam-\$ctx" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
  fi
done
ARMS
jobs=$(mktemp)
: > "$jobs"
[ -n "${RUN_OVERRIDE:-}" ] || echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi ${EXPERIMENT:-meta-mem} gfx1100,gfx1201,gfx1030" > "$jobs"
echo "VIS=0,1,2,3 SCRIPT $N $R/$N.arms.sh @$RUN $R/$N" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
grep -E "^timing:|SERVER_FAILED" "$R/$N.log" | cut -c1-200
echo "greedy text per arm (flag arms must match their base: Pa, Pm = P; Oa, Om, Oam = O):"
for d in "$R/$N"/*/; do echo "  $(basename "$d") $(md5sum "$d"*.greedy.txt 2>/dev/null | awk '{print substr($1,1,12)}' | sort -u | tr "
" " ") $(grep -hcE " E .*(out of memory|illegal|abort|assert)" "$d"*.server.log | head -1) error lines"; done
for d in "$R/$N"/*/; do
  echo "== $(basename "$d")  (MiB per device: compute arenas | static buffers by first tensor)"
  grep -h "BIGCHERRY_META_MEM" "$d"*.server.log | grep -vE "ROCm3|buft=CPU" | awk '
    { dev = ""; size = 0; first = ""; kind = ($0 ~ /BIGCHERRY_META_MEM (compute|arena) /) ? "compute" : "static"
      for (i = 1; i <= NF; i++) { if ($i ~ /^dev=/) dev = substr($i, 5); if ($i ~ /^size_mib=/) size = substr($i, 10); if ($i ~ /^first=/) first = substr($i, 7) }
      if (kind == "compute") cls = "compute arena"
      else if (first ~ /^cache_idx/) cls = "indexer cache"
      else if (first ~ /^cache_/) cls = "KV cache"
      else if (first ~ /^blk\.|^token|^output|^per_layer/) cls = "weights"
      else cls = "other static (" first ")"
      if (kind == "compute") { if (size > s[cls, dev]) s[cls, dev] = size } else s[cls, dev] += size   # arenas are re-reported per reserve: keep the largest
      seen[cls] = 1; if (dev > maxdev) maxdev = dev }
    END { for (c in seen) { printf "  %-34s", c; for (d = 0; d <= maxdev; d++) printf " dev%d %9.1f", d, s[c, d]; printf "\n" } }' | sort
done
echo ALL_RUNS_DONE
