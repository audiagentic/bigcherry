# ENG05: v26 mailbox hand-off trace experiment

All measurements are pending GPU execution. This is a diagnostic build, not a speed fix.

## Existing and new counters

The existing helper TRACE record from worker_harness.h (sampled every Nth request by experiment.diagnostics.lanes.trace_level=N) reports layer, lane, seq, rows, routes, local_us, wait_us, helper_gpu_us, act_us, gemm_us, write_us, combine_wait_us and combine_us. local_us includes unrelated local engine work between publishing and waiting; wait_us includes real helper service. Existing sampled stage averages are not an engine mailbox decomposition.

With this change, each request has six engine GPU wall_clock64 stamps in the existing 48-byte v26 Request tail padding (sizeof Request stays 256). A TRACE line also reports:
- mailbox_stamp_ok: 1 only when all stamps were observed and ordered; invalid/missing fields print zero and MUST NOT be interpreted as zero latency
- publish_us: publish kernel entry to immediately before this helper request release store; includes the old system fence, histogram accounting and activation staging. For successive helpers in ONE publish kernel it is cumulative; use the maximum active-helper value per layer rather than summing the values.
- wait_bookkeeping_us: engine observation of helper completion to the end of engine wait bookkeeping
- finish_us: finish-kernel entry through consumed-record/count completion, per slot
- combine_engine_wait_us: engine polling for combine.done, zero on uncombined paths

These are all engine GPU clocks at 100 MHz and can be compared only to other stamps on the SAME engine. Existing helper GPU stamps are from separate clock domains. Do not sum parallel helper waits. With trace_level=0 there are no new GPU clock reads or trace stores; the same kernels, wait/fences and finish launches remain.

## Exact policy-on diagnostic run (hold GPU lease throughout)

Build a COMPLETE engine+helper+serving release from this branch, including the rebuilt rad0070 managed entry. Do not mix plugins built with old BcEngineSettings. The native owner admits only an exact lab-only trace experiment of shape {"diagnostics":{"lanes":{"trace_level":N}}}; shape, force and other diagnostics remain refused. For this pure diagnostic, a measured lab manifest derives its live policy from the SAME narrow calibration key and record as production. Check live_policy_state.configured is true in the committed measured manifest before measuring. Never bypass the manifest reader or owner guards.

~~~bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)  # run from the ENG05 task checkout
cd "$REPO"
source tools/env/bigcherry-env.sh
BUILD=/absolute/path/to/complete/eng05-mailbox-trace-build
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"
RUN="$BC_WORK_ROOT/eng05-mailbox-trace-single"
TRACE_JSON="$BC_WORK_ROOT/eng05-mailbox-trace-experiment.json"
CAL="$BC_WORK_ROOT/live-policy-calibration.toml"
HEAT="$REPO/tools/lab/radiance/heat/flashnext-mixed.hot-first"
test -x "$BC_SERVE"; test -x "$BUILD/bin/radiance"; test -s "$CAL"; test -f "$HEAT"
mkdir -p "$RUN"
printf '%s\n' '{"diagnostics":{"lanes":{"trace_level":1}}}' > "$TRACE_JSON"
"$BC_SERVE" serve flashnext-4gpu \
  --run-dir "$RUN" --model-root "$BC_MODEL_ROOT" \
  --engine-binary "$BUILD/bin/radiance" \
  --configs "$REPO/engines/radiance/configs" \
  --power-profile stock --allow-power-apply \
  --set "engine.radiance_home=$BUILD/radiance_home" \
  --set "handoff.heat=$HEAT" \
  --set "handoff.histogram=$RUN/heat.live" \
  --set handoff.histogram_period_s=1 \
  --calibration "$CAL" --lab-experiment "$TRACE_JSON" \
  > "$RUN/owner.log" 2>&1 &
OWNER_PID=$!
trap '"$BC_SERVE" stop --run-dir "$RUN" >/dev/null 2>&1 || true; wait "$OWNER_PID" || true' EXIT
ready=0
for _ in $(seq 1 2700); do
  if "$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null |
       jq -e '.state=="running"' >/dev/null; then ready=1; break; fi
  kill -0 "$OWNER_PID" 2>/dev/null || { cat "$RUN/owner.log"; exit 1; }
  sleep 1
done
test "$ready" -eq 1
MANIFEST=$(jq -r .manifest "$RUN/session.json")
jq -e '.spec.intent.service.profile=="lab" and
       .spec.experiment.diagnostics.lanes.trace_level==1 and
       .spec.handoff.placement.live_policy_state.configured==true' "$MANIFEST"

# Same workload/451s warmup as flashnext-parity.sh; no idle-only sleep.
PYTHONPATH=tools python3 tools/lab/radiance/flashnext_parity.py \
  doc-calibrate --url http://127.0.0.1:18450 --tokens 44000 \
  --measured 6 --timeout 600 --out "$RUN/doc-calibration.json"
start=$SECONDS; i=0
while (( SECONDS-start < 451 )); do
  PYTHONPATH=tools python3 -m bigcherry stream-bench \
    --url http://127.0.0.1:18450 --streams 1 \
    --prompt-tokens 27000 --max-tokens 8 --reps 1 \
    --temperature 0 --seed "$((900+i))" --timeout 600 \
    --output "$RUN/warm-stream-$i.json" >/dev/null
  (( SECONDS-start >= 451 )) && break
  PYTHONPATH=tools python3 tools/lab/radiance/flashnext_parity.py \
    doc-prefill --url http://127.0.0.1:18450 \
    --calibration "$RUN/doc-calibration.json" --slot "warmup:$i" \
    --timeout 600 --out "$RUN/warm-doc-$i.json"
  i=$((i+1))
done

PYTHONPATH=tools python3 -m bigcherry stream-bench \
  --url http://127.0.0.1:18450 --streams 1 \
  --prompt-tokens 900 --max-tokens 256 --reps 2 \
  --temperature 0 --seed 7 --timeout 600 --output "$RUN/decode1.json"
HELPER_LOG=$(jq -r '.processes[]|select(.role=="helper0")|.log' "$RUN/session.json")
rg '^TRACE,' "$HELPER_LOG" > "$RUN/mailbox-trace.csv"
"$BC_SERVE" stop --run-dir "$RUN"; wait "$OWNER_PID"; trap - EXIT
~~~

The lab trace adds instrumentation and must NOT be used for headline performance numbers. Group TRACE rows by layer and wall_ns interval corresponding to decode1, report per-layer p50/p95 for all new and old fields. A 0.5 ms decode-step loss divided by 49 routed layers needs around 10 us additional serial per layer; this is a criterion, NOT a measurement.

## v25 versus v26 (#1014) distinctions and source locations

- Publish: at moe_mailbox_ops.hip:160–375, v25 fixed kMaxSlots LDS step arrays, old Header::ring_offset and per-rank result-offset loop are now counted dynamic LDS, runtime Header::regions[Requests], runtime slot counts and a single result offset. Potential minor arithmetic/header load cost, less static work. No established cost until GPU timing (a plausible sub-us bookkeeping difference per layer, not a claimed observation).
- Mandatory engine release: the same system fence and release store before helper sees a request in v25 and v26, moe_mailbox_ops.hip:355–375. Not an extra v26 fence, though it can take multiple microseconds.
- Wait/add: fixed bounded arrays and helper waits in v25 versus runtime-sized dynamic scratch, lane topology and same acquire wait loops in v26, moe_mailbox_ops.hip:407–490 and 570–615. No newly unconditional extra wait kernel or remote round trip observed in diff; GPU scheduling effects unknown.
- Finish: same one finish launch but dynamic slot/consumed-count scratch and tracking in v26, moe_mailbox_ops.hip:815–866. Measure finish_us.
- Helper combine: same helper0 chunk CPU operation and engine polling, now runtime helpers/slots, worker_combine.h:145–240. Combine waits include genuine helper completion/CPU combination; not automatically ABI overhead.
- Setup-only: moe_remote_hook.cpp and moe_v26_gpu_contract.h replace static footprint/capacity with counted admission and device-attribute query at rank start; not per-routed-layer HIP queries.

v25 in d03287623^ names kVersion=25; v26 names kVersion=26 and has different Header regions/lane/offset and Request interpretation. A v25 engine cannot attach to v26 helpers or vice versa. The complete bc-radiance-1170.14.0 release can serve the same physical Flash-Next GPU layout as one **homogeneous v25 owner, engine, helpers and shm**, but that whole-build A/B includes other code, policy and power/placement changes; not an isolated #1014 result. A direct revert of #1014 on modern main is NOT an isolated supported build because later callers and contracts depend on v26 typed topology. If trace points at bookkeeping, do a within-v26 fixed-1engine/3helper specialization retaining all v26 guards and wire, compared against main with identical binaries except that specialization.

Compile/CTest, patch-lint, forbidden-term and hardware results must be reported separately; no GPU was used for preparing this instrumentation.
