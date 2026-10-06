#!/bin/bash
# MET04 / 1283 first end-to-end run: build the production set + 1281 + 1283 (experiment moe-expert-parallel), then on
# Flash-Next production (2x XTX + R9700 tensor split, MTP draft on the 6900 XT), per depth:
#   1. ABBA on the one binary, A = row split (default), B = BIGCHERRY_MOE_EP=1 (whole experts per device, one delayed
#      AllReduce per block): prefill t/s, decode t/s, acceptance, greedy text;
#   2. fidelity (flash-fidelity.sh) at the first depth: D = row split, S = expert split.
# The expert split is not bit-identical to the row split (different partial sums), so the fidelity gate decides
# correctness, not text identity. BIGCHERRY_PATCH_TRACE is on so a fallback to three AllReduces would show up as
# a slower arm, and the per-device expert counts are in the server log.
# CTX from the environment (default 245760). The expert split distributes expert bytes exactly by -ts, while the row
# split rounds rows to 128 (640 rows at 0.31 / 0.27 / 0.42 become 192 / 64 / 384), so a -ts tuned for the row split
# does not fit the expert split at the full context: use a smaller CTX or a re-tuned TS for the comparison.
# Usage: queue-moe-ep.sh <tag> <depth>...
set -u
TAG=${1:?tag}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
depths=()
for a in "$@"; do case "$a" in wait=*) until grep -q "^ALL_RUNS_DONE" "${a#wait=}" 2>/dev/null; do sleep 20; done ;; *) depths+=("$a") ;; esac; done
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=${CTX:-245760} TS=0.31,0.27,0.42 UB=512 B=512 DECODE_N=${DECODE_N:-512}
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
docker stop radiance-vllm >/dev/null 2>&1
RUN=b-moeep-$TAG
jobs=$(mktemp)
echo "VIS=0,1,2,3 BUILD $RUN bigcherry:stock:linux-multi moe-expert-parallel gfx1100,gfx1201,gfx1030" > "$jobs"
for d in "${depths[@]}"; do
  echo "VIS=0,1,2,3 SCRIPT moeep-$TAG-d$d tools/lab/flash-next/flash-prefill-env-ab.sh $d @$RUN $R/moeep-$TAG-d$d BIGCHERRY_MOE_EP=1" >> "$jobs"
done
echo "VIS=0,1,2,3 SCRIPT moeep-$TAG-fid tools/lab/flash-next/flash-fidelity.sh ${depths[0]} @$RUN $R/moeep-$TAG-fid noref BIGCHERRY_MOE_EP=1" >> "$jobs"
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for d in "${depths[@]}"; do
  echo "== depth $d: A = row split, B = BIGCHERRY_MOE_EP=1"; grep -E "^d[0-9]|^md5|SERVER_FAILED" $R/moeep-$TAG-d$d.log
  grep -hE " E |abort|assert|unsupported mul_mat split|illegal memory" $R/moeep-$TAG-d$d/*-B/*.server.log 2>/dev/null | head -5 | cut -c1-220
done
echo "== fidelity: D = row split, S = expert split"; grep -E "^D:|^S:|^D2:| vs |SERVER_FAILED" $R/moeep-$TAG-fid.log
echo ALL_RUNS_DONE
