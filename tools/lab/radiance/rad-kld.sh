#!/bin/bash
# RR06: radiance's KL mode (docs/TOOLS.md, "The KL mode") on one container: record a reference, or score a container
# against a recorded one. The mode starts as a serve would, with the same placement and kernels, runs the corpus
# through prefill and exits, so it measures a kernel library and a weight form together.
# The corpus is cut from the lab's KLD text (kld-docs.txt) if the JSONL does not exist yet: CHUNKS pieces of
# CHUNK_CHARS characters, each scored from its first position.
# Run as a queue SCRIPT job on the cards the container is served on:
#   MODE=record VIS=2 SCRIPT kld-ref tools/lab/radiance/rad-kld.sh @<any build run> <out-dir>
#   MODE=ref    VIS=0,1 SCRIPT kld-x tools/lab/radiance/rad-kld.sh @<any build run> <out-dir>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: rad-kld.sh <ignored> <out-dir>
# env: MODE (record | ref), REF (the reference directory; record writes it, ref reads it), MODEL (.rad container),
#      GPU (HIP index, 0), TARGET (gfx1100), TP (1), P2P (auto), KERNELS (libr3,libref), CTX (8192), KV (bf16),
#      CORPUS (JSONL), CHUNKS (24), CHUNK_CHARS (6000), EXTRA, KLD_TIMEOUT (3600 s), RADIANCE_SRC, WORK
set -u
out=$2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
model=${MODEL:?MODEL}
ref=${REF:?REF}
corpus=${CORPUS:-/mnt/data/bigcherry-work/corpus/rad-kld-corpus.jsonl}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
kernels=${KERNELS:-libr3,libref}
home=$src/build/radiance_home
case ",$kernels," in *,libr3,*)
    r3so=$(find "$work/libr3-$target-build" -name 'libr3.so' 2> /dev/null | head -1)
    [ -n "$r3so" ] || { echo "NO_LIBR3: run libr3-build.sh first"; exit 1; }
    home="$(dirname "$(dirname "$r3so")"):$home" ;;
esac
[ -s "$corpus" ] || python3 "$here/rad_kld_corpus.py" "$corpus" "${CHUNKS:-24}" "${CHUNK_CHARS:-6000}" || exit 1
case ${MODE:?MODE} in
    record) [ -e "$ref/kld.json" ] && { echo "REFERENCE_EXISTS $ref"; exit 0; }
            mode=(--kld-record "$ref" --kld-corpus "$corpus") ;;
    ref) [ -e "$ref/kld.json" ] || { echo "NO_REFERENCE $ref"; exit 1; }
         mode=(--kld-ref "$ref" --kld-out "$out/kld.json") ;;
    *) echo "unknown MODE $MODE"; exit 2 ;;
esac
echo "radiance $(git -C "$src" rev-parse --short HEAD); $MODE; model $(basename "$model"); HIP device ${GPU:-0}; tp ${TP:-1}; kernels $kernels; corpus $(wc -l < "$corpus") prompts"
t0=$(date +%s)
HIP_VISIBLE_DEVICES=${GPU:-0} timeout "${KLD_TIMEOUT:-3600}" "$src/build/bin/radiance" --model "$model" \
    --radiance-home "$home" --kernels "$kernels" --tp "${TP:-1}" --p2p "${P2P:-auto}" --max-model-len "${CTX:-8192}" \
    --max-num-seqs 1 --kv-cache-dtype "${KV:-bf16}" "${mode[@]}" ${EXTRA:-} > "$out/kld.log" 2>&1
rc=$?
echo "radiance: exit $rc in $(( $(date +%s) - t0 )) s"
grep -iE "kld|kl |mean|median|percentile|top-1|agreement|perplexity|positions|^E " "$out/kld.log" | tail -30 | cut -c1-220
exit $rc
