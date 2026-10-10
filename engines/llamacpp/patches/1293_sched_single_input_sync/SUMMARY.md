# 1293_sched_single_input_sync

**Status:** superseded
**Plan item:** QFN01/RNX01

## What it does

`ggml_backend_sched` synchronizes an event-less split backend once per split before copying user inputs,
instead of before every input.

## Why

With `-sm tensor` the split backend is the meta backend, which has no events, so every user input copy began
with a synchronize of every rank. A `hipStreamSynchronize` call-site trace of Flash-Next MTP decode
(`tools/lab/flash-next/sync-tracer.c`) showed ~485 host syncs per MTP step, almost all from this site and
the meta buffer `set_tensor`; the host spent ~82% of decode wall time inside the HIP API. One synchronize
already guarantees the previous use of every input buffer has finished.

Validation: decode t/s and ms/step A/B at 10K and 80K with MTP3; greedy parity without MTP at 10K; sync
count per step from the tracer.

## Result (2026-10-03, flashnext-sync-ab-1, ABBA on 1291+1292)

Neutral. ms per MTP step 10K: base 50.4/50.8 vs new 50.6/50.1; 80K: base 65.0/65.4 vs new 65.2/65.5.
hipStreamSynchronize per 256-token decode: 40325 -> 32894 (-18%). Greedy output without MTP identical at 10K.
The removed syncs were waits on GPU work the next step needs anyway, so host blocking dropped without
shortening the critical path. Not part of any deployment recipe; kept as a correct, upstreamable cleanup.

## Hardware result (2026-10-04 review)

Neutral on hardware (QFN01): sync-count reduction alone did not improve decode.

## b11474 BPB01 review

Current-pin composition and disposition options are recorded in `releases/evidence/bpb01-four-evaluated.md`. This review does not change patch state.

## Result at b11474: superseded by 1326 (2026-10-09)

Two builds of main at pin b11474 (b9acf138): `b-main2` = production, `b-sync2` = production + 1293. Flash-Next
production profile, two ABBAs per depth. Prefill t/s (production / +1293): 8K 1239-1254 / 1228-1255, 24K 1269-1278 /
1274-1280, 98K 1216-1228 / 1226-1231. Decode t/s: 8K 85.2-87.6 / 85.7-88.3, 24K 71.1-73.2 / 71.9-73.0, 98K 70.0-70.6 /
70.3-71.2. No separation at any depth; greedy text identical in all four runs at each depth.

1326 (asynchronous host inputs, validated, in the production set) removes the same scheduler input-copy wait at a
stronger level, and with it in place the single-sync change adds nothing measurable. The `ar-cpu-root-kpool-sync`
experiment set and its two queue scripts were removed with this decision.
