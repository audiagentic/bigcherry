#!/bin/bash
# Do the text-changing prefill patches cost decode through draft acceptance? In the one-prompt ablation
# (queue-patch-ablation.sh) every patch that changes the output text showed FASTER decode with the patch off, with
# higher draft acceptance: 1350, 1347 and 1334. One prompt cannot tell luck from an effect. This repeats the on/off
# ABBA at one depth over several different requests after the same filled context, and reports per request and
# pooled: decode t/s, accepted / drafted, and prefill t/s. A = production, B = the one switch off.
# The patches do not touch decode kernels (they apply to prefill-width batches), so a consistent decode gap here is
# an acceptance effect: the target's state after the prompt differs and the drafter agrees with it more or less.
# Usage: queue-decode-acceptance.sh [build]     (default b-main3)     env: DEPTH (24576), SWITCHES, DECODE_N (512)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
build=${1:-b-main3}
DEPTH=${DEPTH:-24576}
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
SWITCHES=${SWITCHES:-"1334:BIGCHERRY_FA_SPARSE=0 1347:BIGCHERRY_F32_THIN_MMVF=0 1350:BIGCHERRY_MMQ_FEW_TILE_STREAMK=0"}
asks=(
    "Summarise the above in detail:"
    "List the ten most important facts stated above, one per line:"
    "Write the next section of the text above, in the same style:"
    "Explain the main problem described above and how it could be solved:"
)
for item in $SWITCHES; do
    pid=${item%%:*} sw=${item#*:}
    for i in "${!asks[@]}"; do
        name=dacc-$pid-q$i
        rm -rf "$R/$name-d$DEPTH" "$R/$name-d$DEPTH.log"
        ASK="${asks[$i]}" AB_ENV="$sw" bash tools/lab/flash-next/queue-env-ab.sh "$name" "$build" "$DEPTH" > "$R/$name.out" 2>&1
    done
    python3 - "$pid" "$sw" "$R" "${#asks[@]}" <<'PY'
import re, sys
pid, sw, root, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
pool = {"A": [], "B": []}
print(f"== {pid} {sw} (A = production, B = switch off)")
for i in range(n):
    text = open(f"{root}/dacc-{pid}-q{i}.out", errors="replace").read()
    arms = {"A": [], "B": []}
    for m in re.finditer(r"^d\d+ \d ([AB]): .*?prefill \d+ tok at ([0-9.]+) t/s.*?decode ([0-9.]+) t/s, accepted (\d+)/(\d+)", text, re.M):
        arms[m.group(1)].append((float(m.group(2)), float(m.group(3)), int(m.group(4)), int(m.group(5))))
    md5 = dict(re.findall(r"^md5 ([AB]): (\S+)", text, re.M))
    if len(arms["A"]) < 2 or len(arms["B"]) < 2:
        print(f"  q{i}: INCOMPLETE {len(arms['A'])} A, {len(arms['B'])} B")
        continue
    def line(arm):
        rows = arms[arm]
        dec = sum(r[1] for r in rows) / len(rows)
        acc = 100.0 * sum(r[2] for r in rows) / sum(r[3] for r in rows)
        pre = sum(r[0] for r in rows) / len(rows)
        return dec, acc, pre
    (da, aa, pa), (db, ab, pb) = line("A"), line("B")
    pool["A"].append((da, aa, pa)); pool["B"].append((db, ab, pb))
    same = "same text" if md5.get("A") == md5.get("B") else "text differs"
    print(f"  q{i}: decode A {da:.1f} B {db:.1f} ({100*(db/da-1):+.1f}%); acceptance A {aa:.1f}% B {ab:.1f}%; "
          f"prefill A {pa:.1f} B {pb:.1f} ({100*(pb/pa-1):+.1f}%); {same}")
if pool["A"]:
    k = len(pool["A"])
    mean = lambda arm, j: sum(r[j] for r in pool[arm]) / k
    faster_off = sum(1 for a, b in zip(pool["A"], pool["B"]) if b[0] > a[0])
    print(f"  pooled over {k} request(s): decode A {mean('A',0):.1f} B {mean('B',0):.1f} "
          f"({100*(mean('B',0)/mean('A',0)-1):+.1f}%), acceptance A {mean('A',1):.1f}% B {mean('B',1):.1f}%, "
          f"prefill ({100*(mean('B',2)/mean('A',2)-1):+.1f}%); decode faster with the patch off in {faster_off} of {k}")
PY
done
echo DECODE_ACCEPTANCE_DONE
