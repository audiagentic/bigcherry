#!/bin/bash
# Which production patches cost speed? Flash-Next production profile, ctx 245760, f16 KV.
#   1. Build the production set of the checked-out tree as b-<tag> (no experiment).
#   2. Two-build ABBA of BASE (default b-main2, the previous production build) against b-<tag> at 24K and 98K:
#      has the production set as a whole lost ground since BASE?
#   3. One-binary ablation on b-<tag>: for every runtime switch of a production patch, ABBA at DEPTH (24576) with
#      A = production, B = that one switch flipped off. A patch regresses when its B arm is the faster one.
#      Each line: switch, prefill A / B, decode A / B, B against A in percent, whether the text is the same.
# Patches without a runtime switch are not covered here; they need a leave-one-out build.
# A switch whose off state does not fit in memory or changes placement reports SERVER_FAILED and is skipped.
# Usage: queue-patch-ablation.sh [tag]     (default tag main3)     env: BASE, DEPTH, SWITCHES (override the list)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
tag=${1:-main3}
BASE=${BASE:-b-main2}
DEPTH=${DEPTH:-24576}
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
# patch -> the switch that turns it off under the flashnext profile
SWITCHES=${SWITCHES:-"1358:BIGCHERRY_META_SPLIT_CACHE_EVICT=0 1348:BIGCHERRY_MTP_DEFERRED_CATCHUP=0 1322:BIGCHERRY_MTP_AHEAD=0 1350:BIGCHERRY_MMQ_FEW_TILE_STREAMK=0 1347:BIGCHERRY_F32_THIN_MMVF=0 1345:BIGCHERRY_MOE_IDS_MULTIWARP=0 1344:BIGCHERRY_HC_GRID_INDEX=0 1334:BIGCHERRY_FA_SPARSE=0 1326:BIGCHERRY_SCHED_ASYNC_INPUTS=0 1295:BIGCHERRY_QSA_GATHER=0 1302:BIGCHERRY_GRAPH_OOM_EVICT=0 1341:BIGCHERRY_META_SUBSET_MIRROR_INPUTS=0"}

echo "== 1. build b-$tag at $(git log --oneline -1 | cut -c1-70)"
rm -f "$R/b-$tag.log"
j=$(mktemp)
echo "VIS=0,1,2,3 BUILD b-$tag bigcherry:stock:linux-multi - gfx1100,gfx1201,gfx1030" > "$j"
bash tools/lab/plan-qualification/queue.sh "$j" | grep -E "rc=|blocked"
rm -f "$j"
grep -hE "^BUILD_(EXIT|BINARY)" "$R/b-$tag.log" | cut -c1-160
grep -q "^BUILD_EXIT=0" "$R/b-$tag.log" || { echo "build failed; stopping"; echo PATCH_ABLATION_DONE; exit 1; }

echo "== 2. $BASE against b-$tag"
rm -rf $R/abl-$tag-base-d* $R/abl-$tag-base-d*.log
bash tools/lab/flash-next/queue-prefill-ab.sh "$BASE" "b-$tag" "abl-$tag-base" 24576 98304 | sed -E "s/timing: prompt 4 tok at [0-9.]+ t\/s, //; s/timing: SERVER_EXIT.*//" | grep -E "^==|^d[0-9]|^ +[0-9]|SERVER_FAILED" | cut -c1-130

echo "== 3. one switch off at a time on b-$tag, depth $DEPTH (B against A: positive = faster with the patch off)"
for item in $SWITCHES; do
    pid=${item%%:*} sw=${item#*:}
    name=abl-$tag-$pid
    rm -rf "$R/$name-d$DEPTH" "$R/$name-d$DEPTH.log"
    AB_ENV="$sw" bash tools/lab/flash-next/queue-env-ab.sh "$name" "b-$tag" "$DEPTH" > "$R/$name.out" 2>&1
    python3 - "$pid" "$sw" "$R/$name.out" <<'PY'
import re, sys
pid, sw, path = sys.argv[1:4]
text = open(path, errors="replace").read()
arms = {"A": [], "B": []}
for m in re.finditer(r"^d\d+ \d ([AB]): .*?prefill \d+ tok at ([0-9.]+) t/s.*?decode ([0-9.]+) t/s", text, re.M):
    arms[m.group(1)].append((float(m.group(2)), float(m.group(3))))
md5 = dict(re.findall(r"^md5 ([AB]): (\S+)", text, re.M))
if len(arms["A"]) < 2 or len(arms["B"]) < 2:
    tail = " ".join(text.split()[-18:])
    print(f"{pid} {sw}: INCOMPLETE ({len(arms['A'])} A, {len(arms['B'])} B runs) {tail[:150]}")
    sys.exit(0)
mean = lambda rows, i: sum(r[i] for r in rows) / len(rows)
show = lambda rows, i: " / ".join(f"{r[i]:.1f}" for r in rows)
def verdict(i):
    a, b = [r[i] for r in arms["A"]], [r[i] for r in arms["B"]]
    sep = "separated" if min(b) > max(a) or max(b) < min(a) else "overlap"
    return 100.0 * (mean(arms["B"], i) / mean(arms["A"], i) - 1.0), sep
dp, sp = verdict(0)
dd, sd = verdict(1)
same = "same text" if md5.get("A") and md5.get("A") == md5.get("B") else "TEXT DIFFERS"
print(f"{pid} {sw}: prefill A {show(arms['A'], 0)} B {show(arms['B'], 0)} = {dp:+.1f}% ({sp}); "
      f"decode A {show(arms['A'], 1)} B {show(arms['B'], 1)} = {dd:+.1f}% ({sd}); {same}")
PY
done
echo PATCH_ABLATION_DONE
