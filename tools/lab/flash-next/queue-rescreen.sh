#!/bin/bash
# Re-screen of candidates whose verdict came from single quick-ab readings.
# quick-ab's 256-token readings swing between about 38.6 and 42.4 ms/step from one server start to the next for the
# same binary (2026-10-12), so a candidate that read "neutral" or a few percent off in one ABA was not measured. This
# repeats the comparison with eight server starts per arm on Flash-Next at one depth (24K, 512 decode tokens):
#   B:<name>:<experiment>              candidate build (production + experiment) against the base build, in
#                                      alternating blocks of four starts: cand, base, cand, base
#   E:<name>:<experiment>:<VAR=value>  one build (production + experiment), flag off / on: A B B A A B B A
#   F:<name>:<VAR=value>               the base build itself, flag off / on: A B B A A B B A
# Prints per arm: decode ms/step (each start, mean, min, max), prefill t/s (mean), accepted / drafted (sum), and how
# many distinct greedy texts each arm produced.
# Usage: queue-rescreen.sh <base build run> <entry>...        env: DEPTH (24576)
set -u
BASE=${1:?base build run}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=$(bash tools/lab/plan-qualification/work-root.sh "$PWD")/runs
DEPTH=${DEPTH:-24576}

build() {  # <build run> <experiment>
    # the queue reuses an existing build run of the same name
    local jobs; jobs=$(mktemp)
    echo "VIS=0,1,2,3 BUILD $1 bigcherry:stock:linux-multi $2 gfx1100,gfx1201,gfx1030" > "$jobs"
    bash tools/lab/plan-qualification/queue.sh "$jobs" > "$R/$1-queue.log" 2>&1
    local rc=$?; rm -f "$jobs"; return $rc
}

summ() {  # <label> <arm letter> <run log>...
    local label=$1 arm=$2; shift 2
    grep -hE "^d[0-9]+ [0-9]+ $arm:" "$@" 2>/dev/null |
        sed -E 's/.*fill prefill [0-9]+ tok at ([0-9.]+) t\/s.*decode ([0-9.]+) t\/s, accepted ([0-9]+)\/([0-9]+).*/\1 \2 \3 \4/' |
        awk -v label="$label" '
            NF == 4 { n++; ms = 512 / $2 / (512 - $3) * 1e3; s += ms; pf += $1; acc += $3; dr += $4; all = all sprintf("%.1f ", ms)
                      if (n == 1 || ms < lo) lo = ms; if (n == 1 || ms > hi) hi = ms }
            END { if (n) printf "  %-10s n=%d  decode ms/step mean %.2f (min %.1f max %.1f)  prefill %.0f t/s  accepted %d/%d (%.1f%%)  [%s]\n",
                              label, n, s / n, lo, hi, pf / n, acc, dr, 100.0 * acc / dr, all
                  else printf "  %-10s no completed runs\n", label }'
}

texts() {  # <label> <arm letter> <run dir>...
    local label=$1 arm=$2; shift 2
    local n; n=$(for d in "$@"; do md5sum "$d"/d*-"$arm"/*.greedy.txt 2>/dev/null; done | awk '{print $1}' | sort -u | wc -l)
    echo "  $label: $n distinct greedy text(s)"
}

for entry in "$@"; do
    IFS=: read -r kind name a b <<< "$entry"
    echo "######## $entry"
    case "$kind" in
        B)
            build "b-rs-$name" "$a" || { echo "  build of $a failed (see $R/b-rs-$name-queue.log)"; continue; }
            for i in 1 2; do
                ARMS="A A A A" AB_ENV=BC_NOOP=1 bash tools/lab/flash-next/queue-env-ab.sh "rs-$name-c$i" "b-rs-$name" "$DEPTH" > /dev/null 2>&1
                ARMS="A A A A" AB_ENV=BC_NOOP=1 bash tools/lab/flash-next/queue-env-ab.sh "rs-$name-b$i" "$BASE" "$DEPTH" > /dev/null 2>&1
            done
            summ "base" A "$R/rs-$name-b1-d$DEPTH.log" "$R/rs-$name-b2-d$DEPTH.log"
            summ "candidate" A "$R/rs-$name-c1-d$DEPTH.log" "$R/rs-$name-c2-d$DEPTH.log"
            texts "base" A "$R/rs-$name-b1-d$DEPTH" "$R/rs-$name-b2-d$DEPTH"
            texts "candidate" A "$R/rs-$name-c1-d$DEPTH" "$R/rs-$name-c2-d$DEPTH"
            echo "  base and candidate together: $(md5sum "$R"/rs-"$name"-[bc][12]-d"$DEPTH"/d*-A/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort -u | wc -l) distinct greedy text(s)"
            ;;
        E|F)
            if [ "$kind" = E ]; then
                run="b-rs-$name"; flag=$b
                build "$run" "$a" || { echo "  build of $a failed (see $R/$run-queue.log)"; continue; }
            else
                run=$BASE; flag=$a
            fi
            ARMS="A B B A A B B A" AB_ENV="$flag" bash tools/lab/flash-next/queue-env-ab.sh "rs-$name" "$run" "$DEPTH" > /dev/null 2>&1
            summ "off" A "$R/rs-$name-d$DEPTH.log"
            summ "$flag" B "$R/rs-$name-d$DEPTH.log"
            texts "off" A "$R/rs-$name-d$DEPTH"
            texts "on" B "$R/rs-$name-d$DEPTH"
            echo "  off and on together: $(md5sum "$R"/rs-"$name"-d"$DEPTH"/d*-[AB]/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort -u | wc -l) distinct greedy text(s)"
            ;;
        *) echo "  unknown entry kind '$kind'";;
    esac
done
echo RESCREEN_DONE
