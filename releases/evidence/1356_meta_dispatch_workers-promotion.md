# 1356_meta_dispatch_workers promotion evidence

Promotion evidence, profile-evidence tier (QFP18), pin b11474, 2026-10-10.

Model: Qwen3.8-Flash-Next UD-IQ4_XS (production profile: 2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT, ctx 245760, f16 KV, ub512)

Mechanics: offline tests for 1356 pass, patch-lint clean, composition with the production set clean after the anchor was narrowed to the inner per-backend loop (composes after 1320).
Activation: BIGCHERRY_PATCH_HIT patch=1356 marker at 24K in the threaded arm only (run mdw3-r1-d24576, order A B B A: 0 1 1 0).
Identity: greedy text identical in every run at 8K, 24K and 98K (two ABBAs each), and in the 52-run stress below.
A/B: one binary (b-metamem-mdw3 = production + 1356), BIGCHERRY_META_DISPATCH_THREADS=1 against default (off), two ABBAs per depth: prefill +3.5% at 8K, +1.7% at 24K (completely separated, 4 against 4), +1.6% at 98K; decode and draft acceptance unchanged.

Scope: profile-scoped. The flag stays default off; this promotion covers the Flash-Next multi-device tensor-split profile only.

## Flash-Next, one binary, threads on against off

`tools/lab/flash-next/queue-promotion-followups.sh` step 2, A = default (workers off), B = workers on. Prefill t/s:

| Depth | Off | On | Gain |
|---|---|---|---|
| 8K | 1237.2 / 1245.0 / 1151.1 / 1250.5 | 1284.1 / 1288.1 / 1236.1 / 1287.6 | +3.5% (ranges overlap through one low run in each arm) |
| 24K | 1267.0 / 1271.9 / 1240.9 / 1258.1 | 1284.1 / 1277.6 / 1284.3 / 1275.6 | +1.7% (every on-run above every off-run) |
| 98K (runs mdw3-l1, mdw3-l2, 2026-10-10) | 1187.8 / 1225.8 / 1198.1 / 1220.5 | 1231.2 / 1228.3 / 1225.5 / 1223.3 | +1.6% (one off-run inside the on range) |

Decode 85.3-87.9 t/s at 8K, 66.7-73.2 t/s at 24K and 69.2-70.8 t/s at 98K in both arms. The 98K runs are the re-test the
patch SUMMARY required after the graph-capture fix; the marker was not traced in them (no BIGCHERRY_PATCH_TRACE).

## Stress (host threading condition)

`queue-promotion-followups-2.sh` step 5: 26 ABBAs at 8K on b-metamem-mdw3, 52 runs with the workers on and 52 with
them off. All 104 texts are identical (md5 fd4559a5); server failures: 0.

## Shared-state inventory

From the patch SUMMARY: one persistent worker per simple backend, created once (no thread per graph); the caller
joins all submissions before it reads the submitted subgraphs for AllReduce. CUDA graph capture was the shared state
found in qualification (5 of 12 threaded runs gave a different text before the fix); the package now takes a
process-wide `std::shared_mutex`, exclusive for a capturing call and shared for a replaying call, only when the flag
is on. The stress above was run on the build with that fix.

## No regression on another model

Qwen3.8-27B, two RX 7900 XTX, production flags (prod27b-ab.sh, run nr27-1356), production build b-main2 against
b-metamem-mdw3 with the flag at its default: prefill, decode and draft acceptance unchanged, text identical. The
marker was not captured on this model, so this is a no-regression check of the build, not a second activation.

## Native llama.cpp baseline

With the flag at its default (off) the Meta backend dispatches devices one after another exactly as native llama.cpp
b11474 does, on the same binary. The off arm of the ABBA is therefore the native llama.cpp behaviour for the code
this patch changes; the rest of the build is the BigCherry production baseline in both arms.
