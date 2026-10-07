# QFP31 / 1348 deferred MTP catch-up promotion

## Hardware evidence (Brutus, 2026-10-08, b11474, Flash-Next IQ4_XS, 245760 ctx, f16 KV, production layout)

Build `b-metamem-qfp31b` of `[experiment.mtp-deferred-catchup]` at f26b820d (anchor span fix: `mtp-deferred-enable` spans 6 lines). Compiled clean; smoke 0 error lines; `BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup` present.

ABBA on one binary, A = default (deferred catch-up on), B = `BIGCHERRY_MTP_DEFERRED_CATCHUP=0`. Prefill t/s:

| depth | A (on) | B (off) | gain |
|---|---|---|---|
| 8192 | 1205.2, 1233.8 | 1142.8, 1170.8 | +5.4% |
| 24576 | 1241.6, 1250.5 | 1172.3, 1173.3 | +6.3% |
| 98304 | 1174.8, 1206.2 | 1082.7, 1087.7 | +9.7% |

Complete separation at every depth. Decode unchanged (82.2-83.7 / 71.4-73.4 / 64.3-64.5 t/s), acceptance equal within a few tokens. Greedy text md5 identical A vs B at all depths (e3833264.., 0d8c726f.., 3ee76e81..), and identical to production before the patch.

Not measured: a second model (Qwen3.8-27B MTP, Gemma), slot cancel / context shift under load.

### Why the gain is ~6-10%, not the ~19% upper bound

The earlier 83 ms / 437 ms estimate was the whole MTP-induced serial gap between target chunk submissions, not 83 ms
of draft compute that this patch could hide. Prior timing split that gap into roughly 29-47 ms/chunk of actual draft
catch-up plus host work needed to prepare and submit the next target chunk.

1348 hides the previous chunk's draft catch-up under the current target chunk, but `process_deferred()` then calls
`llama_get_embeddings_nextn(ctx_tgt)` for the current chunk before returning. That getter synchronizes the current
target, and the host copy of the NextN rows follows it. Therefore the server still cannot build/set inputs/rebuild and
launch the following target chunk while the current target is executing. The remaining serialization is primarily:

- current-chunk target synchronization in the NextN getter;
- the host copy of the current chunk's NextN rows;
- next-chunk graph build / set_inputs / CPU split / Meta subgraph rebuild and launches, which still occur after that
  synchronization instead of overlapping target compute;
- the final pending catch-up flush before speculative begin/sampling (one chunk, proportionally larger at 8K).

The measured +5.4/+6.3/+9.7% therefore matches the previously measured 6.6-9.9% draft-decode share much better than
the 19% whole-gap upper bound.

Follow-up: QFP42 should remove the per-chunk blocking NextN getter from the host critical path. Preferred design is an
MTP-owned asynchronous NextN staging slot: enqueue a device->pinned-host copy after target chunk k on the target
stream(s), return immediately so the host can prepare/submit k+1, then after k+1 is submitted wait only for k's staging
event and run draft catch-up k. This needs explicit output-lifetime/event ownership for Meta/split outputs and must
preserve the same row order/identity. Final flush remains serial. Do not add a worker thread unless the event/staging
path proves insufficient.
