#!/bin/bash
# QFP41: which other mechanism the 1356 output race needs. With the dispatch workers on, Flash-Next at 98K gives a
# second greedy text in about 3 of 8 runs (md5 10a864c9 against b301a49a; runs lk1, lk2r1-3, mdw4r1). Each variant
# here is one baseline pass (workers off) and RUNS passes with the workers on plus one switch that takes a
# candidate out, all on one binary. A variant whose workers-on runs all agree with each other names what the race
# needs; a variant that still shows two texts clears its candidate.
#   base       workers on, nothing else changed (the rate to compare with)
#   nographs   GGML_CUDA_DISABLE_GRAPHS=1          HIP graph capture and replay (the shared state found before)
#   nosparse   BIGCHERRY_FA_SPARSE=0               sparse flash attention (1334): pool allocations per call
#   nodefer    BIGCHERRY_MTP_DEFERRED_CATCHUP=0    the drafter's deferred catch-up (1348)
#   noasync    BIGCHERRY_SCHED_ASYNC_INPUTS=0      asynchronous host inputs (1326)
#   nofusion   GGML_CUDA_DISABLE_FUSION=1          the fusion pass
# The baseline text of a variant can differ from production's (the switch changes the arithmetic); what counts is
# whether the workers-on runs of that variant agree among themselves.
# Usage: queue-1356-race-bisect.sh <build run id> [variant...]     env: RUNS (6), DEPTH (98304)
set -u
RUN=${1:?build run id}; shift
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=$(bash tools/lab/plan-qualification/work-root.sh "$PWD")/runs
variants=("$@"); [ ${#variants[@]} -eq 0 ] && variants=(base nographs nosparse nodefer noasync nofusion)
arms="A"; for _ in $(seq "${RUNS:-6}"); do arms="$arms B"; done
depth=${DEPTH:-98304}
for v in "${variants[@]}"; do
    case $v in
        base) extra="" ;;
        nographs) extra="GGML_CUDA_DISABLE_GRAPHS=1" ;;
        nosparse) extra="BIGCHERRY_FA_SPARSE=0" ;;
        nodefer) extra="BIGCHERRY_MTP_DEFERRED_CATCHUP=0" ;;
        noasync) extra="BIGCHERRY_SCHED_ASYNC_INPUTS=0" ;;
        nofusion) extra="GGML_CUDA_DISABLE_FUSION=1" ;;
        *) echo "unknown variant $v"; exit 2 ;;
    esac
    rm -rf "$R/rb-$v-d$depth"
    # the switch goes to both arms through the caller's environment; only the workers differ between A and B
    out=$(env $extra ARMS="$arms" AB_ENV="BIGCHERRY_META_DISPATCH_THREADS=1" \
              bash tools/lab/flash-next/queue-env-ab.sh "rb-$v" "$RUN" "$depth" 2>&1)
    echo "== $v [${extra:-no switch}]: A $(echo "$out" | grep -a "^md5 A" | cut -c8-15); $(echo "$out" | grep -a "^B texts")$(echo "$out" | grep -ac "SERVER_FAILED" | sed 's/^0$//; s/^\([1-9].*\)$/; SERVER_FAILED x\1/')"
done
echo RACE_BISECT_DONE
