#!/bin/bash
# RR10-C: validate/select native gfx11 paged-prefill attention and GDN chunk scan.
# Both forms are opt-in. Stop immediately on selftest failure; serve/profile runs compare
# baseline, each form alone, and both together on the two-XTX wht6 target.
# Usage: bash tools/lab/radiance/rr10-native-prefill.sh <tag>
# env: R, BUILD, STEPS ('build self serve profile'), DEPTH (6500)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}
b=${BUILD:-b-main2}
steps=${STEPS:-build self serve profile}
depth=${DEPTH:-6500}
jobs=$(mktemp)
one() {
    local vis=$1 run=$2 script=$3 args=$4; shift 4
    [ "$args" = - ] && args=
    rm -rf "$R/$run.log" "$R/$run"
    echo "VIS=$vis SCRIPT $run tools/lab/radiance/$script @$b $R/$run $args" > "$jobs"
    env "$@" bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
    echo "== $run [$*]"
}
has() { case " $steps " in *" $1 "*) return 0;; esac; return 1; }
show_self() { grep -E "pass|fail|PASS|FAIL|error|exit" "$1" | tail -8 | cut -c1-220; }
failed_self() { grep -qE "[1-9][0-9]* failed|verdict: FAIL" "$1" || ! grep -q "verdict: PASS" "$1"; }
show_serve() { grep -iE "tok/s|prefill|decode|exit [1-9]|^E " "$1" | tail -8 | cut -c1-220; }

if has build; then
    one 0 "$tag-build" libr3-build.sh - KBENCH=0 GPU=0
    tail -5 "$R/$tag-build.log" | cut -c1-220
fi

rows="attn_prefill_h256_gqa6_fp8kv attn_prefill_h256_gqa6_bf16kv gdn_chunk_scan_k128_v128_c64_bf16"
if has self; then
    one 0 "$tag-self-base" libr3-selftest.sh "$rows" GPU=0 R3_ATTN_NATIVE=0 R3_GDN_NATIVE=0
    show_self "$R/$tag-self-base.log"
    if failed_self "$R/$tag-self-base.log"; then
        echo "RR10C_STOP compatibility baseline selftest failed"; rm -f "$jobs"; exit 1
    fi
    one 0 "$tag-self-attn" libr3-selftest.sh "attn_prefill_h256_gqa6_fp8kv attn_prefill_h256_gqa6_bf16kv" GPU=0 R3_ATTN_NATIVE=1 R3_GDN_NATIVE=0
    show_self "$R/$tag-self-attn.log"
    if failed_self "$R/$tag-self-attn.log"; then
        echo "RR10C_STOP R3_ATTN_NATIVE selftest failed"; rm -f "$jobs"; exit 1
    fi
    one 0 "$tag-self-gdn" libr3-selftest.sh gdn_chunk_scan_k128_v128_c64_bf16 GPU=0 R3_ATTN_NATIVE=0 R3_GDN_NATIVE=1
    show_self "$R/$tag-self-gdn.log"
    if failed_self "$R/$tag-self-gdn.log"; then
        echo "RR10C_STOP R3_GDN_NATIVE selftest failed"; rm -f "$jobs"; exit 1
    fi
fi

common="GPU=0,1 TP=2 P2P=off CTX=8192 BENCH=1 BENCH_DEPTHS=$depth BENCH_TOKENS=4 N=4"
if has serve; then
    one 0,1 "$tag-serve-base" libr3-serve-smoke.sh - $common R3_ATTN_NATIVE=0 R3_GDN_NATIVE=0 "EXTRA=--tp-wire wht6"
    show_serve "$R/$tag-serve-base.log"
    one 0,1 "$tag-serve-attn" libr3-serve-smoke.sh - $common R3_ATTN_NATIVE=1 R3_GDN_NATIVE=0 "EXTRA=--tp-wire wht6"
    show_serve "$R/$tag-serve-attn.log"
    one 0,1 "$tag-serve-gdn" libr3-serve-smoke.sh - $common R3_ATTN_NATIVE=0 R3_GDN_NATIVE=1 "EXTRA=--tp-wire wht6"
    show_serve "$R/$tag-serve-gdn.log"
    one 0,1 "$tag-serve-both" libr3-serve-smoke.sh - $common R3_ATTN_NATIVE=1 R3_GDN_NATIVE=1 "EXTRA=--tp-wire wht6"
    show_serve "$R/$tag-serve-both.log"
fi

if has profile; then
    for cfg in base attn gdn both; do
        case $cfg in
            base) av=0; gv=0;;
            attn) av=1; gv=0;;
            gdn)  av=0; gv=1;;
            both) av=1; gv=1;;
        esac
        one 0,1 "$tag-prof-$cfg" libr3-serve-smoke.sh - $common R3_ATTN_NATIVE=$av R3_GDN_NATIVE=$gv "EXTRA=--profile-ops --tp-wire wht6"
        echo "-- $cfg op summary"
        python3 tools/lab/radiance/profile_ops_summary.py "$R/$tag-prof-$cfg/server.log" 2>&1 | grep -E "attn_paged|gdn_chunk_scan|TOTAL|total" | cut -c1-220
    done
fi

rm -f "$jobs"
echo "RR10_NATIVE_PREFILL_DONE $tag"
