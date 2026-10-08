#!/bin/bash
# Which drafter FILE is best for Qwen3.8-27B Q8_0 on the dual-XTX tensor split, and how each compares with built-in MTP.
# Same measurement as probe-27b.sh (its probe() is reused): 1665-token prompt, <reps> x 128 greedy tokens, acceptance,
# 64-token greedy parity text. Drafters run on their own card (6900 XT = ROCm3, R9700 = ROCm2).
#   current   : the DFlash2 Q8_0 / Q4_K_M and DSpark Q8_0 files already on the lab disk
#   bf16      : incoai/Qwen3.8-27B-DFlash2-GGUF BF16 (full precision)
#   mag       : magnitudedev re-publications of 2026-10-06 ("verified draft attention and proposal layout metadata")
# Usage: drafter-files.sh <llama-server> <out-dir> [layout regex] [reps, default 5]
#   REPO=<lab tree> (default: the tree this script sits in) - set it when the script is run from outside the tree.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
bin=$1 out=$2 only=${3:-} reps=${4:-5}
mkdir -p "$out"
G=/mnt/data/llm-models/qwen3.8-27b/gguf
model=${PROBE_MODEL:-$G/mtp/Qwen3.8-27B-Q8_0.gguf}
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 8 -ub 2048 -b 2048 -lv 4)
# reuse probe() from probe-27b.sh: everything between its definition and its first layout line
eval "$(sed -n '/^probe() {$/,/^}$/p' "$REPO/tools/lab/dflash/probe-27b.sh")"

T="0,1,2,3 -dev ROCm0,ROCm1 -sm tensor"
probe plain $T
probe mtp5 $T --spec-type draft-mtp --spec-draft-n-max 5 -ctkd q8_0 -ctvd q8_0
probe mtp4 $T --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0

dflash() {  # <label> <file>: block 4 and 7, drafter on the 6900 XT; block 7 on the R9700
    [ -f "$2" ] || { echo "$1: FILE_MISSING $2"; return; }
    probe "dflash-$1-n4-d6900" $T -devd ROCm3 -md "$2" --spec-type draft-dflash --spec-draft-n-max 4
    probe "dflash-$1-n7-d6900" $T -devd ROCm3 -md "$2" --spec-type draft-dflash --spec-draft-n-max 7
    probe "dflash-$1-n7-dR9700" $T -devd ROCm2 -md "$2" --spec-type draft-dflash --spec-draft-n-max 7
}
dspark() {  # <label> <file>: block 6, drafter on the R9700 and on the 6900 XT
    [ -f "$2" ] || { echo "$1: FILE_MISSING $2"; return; }
    probe "dspark-$1-n6-dR9700" $T -devd ROCm2 -md "$2" --spec-type draft-dspark --spec-draft-n-max 6
    probe "dspark-$1-n6-d6900" $T -devd ROCm3 -md "$2" --spec-type draft-dspark --spec-draft-n-max 6
}
dflash cur-q8 $G/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf
dflash cur-q4 $G/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
dflash bf16 $G/dflash/Qwen3.8-27B-DFlash2-BF16.gguf
dflash mag-q8 $G/dflash-magnitudedev/Qwen3.8-27B-DFlash2-Q8_0.gguf
dspark cur-q8 $G/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf
dspark mag-q8 $G/dspark-magnitudedev/Qwen3.8-27B-DSpark-Q8_0.gguf

echo "== greedy parity against plain (64 tokens)"
ref=$(md5sum < "$out/plain.greedy.txt" 2>/dev/null | cut -c1-12)
for f in "$out"/*.greedy.txt; do
    m=$(md5sum < "$f" | cut -c1-12); n=$(basename "$f" .greedy.txt)
    [ "$m" = "$ref" ] && echo "  $n identical" || echo "  $n DIFFERS ($m vs $ref)"
done
