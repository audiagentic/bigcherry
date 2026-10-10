---
id: QFP41
order: 41
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:40:00.113369+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Per-device dispatch threads and other host-thread parallelism: design, then implement behind a flag

## Description

External report: each GPU's kernels and collectives launched by its own host thread instead of one thread for all. Owner position updated 2026-10-07: the threaded option can be revisited, and threads may be used to do work in parallel in other areas where it helps. Earlier (same week) the owner called threading dangerous and asked for it to be done carefully; a threaded per-device layout pass in 1340 was written and reverted then. This item now covers design AND implementation: per-device dispatch in the meta backend first, then other candidates (per-device arena planning/binding, host-side input staging, per-device set_tensor uploads at load). The split shows ~40% idle time that this could recover.

## Steps

1. Idle-time timeline of the split per device (where the single dispatch thread is the limiter). 2. Shared-state inventory: meta buffers, split-state cache, simple-tensor containers, pools, CUDA graph cache, collectives ordering, scheduler callbacks (1326/1336). 3. Design with invariants: what each worker owns, where the joins are, what stays single-threaded. 4. Implement behind a flag, default off: persistent per-device workers, no thread creation per graph. 5. Evidence per use: greedy identity, repeated-run determinism, ABBA, a stress run (long decode + deep prefill).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Identical greedy text and probes with the flag on; no run-to-run variation over repeated runs; ABBA with complete separation; stress run without hang or crash; ThreadSanitizer build of the host side where practical.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-10-09 first hardware result for 1356 (per-device dispatch workers, PR #56, build b-metamem-mdw1 = production + 1356, Flash-Next production profile, two ABBAs per depth, A = off, B = BIGCHERRY_META_DISPATCH_THREADS=1, marker present in the B arms). Prefill: 8K 1282.7-1290.0 vs 1252.8-1257.0 t/s (+2.5%), 24K 1285.7-1293.9 vs 1267.1-1279.6 (+0.5 to +1.5%), 98K 1229.8-1233.7 vs 1226.8-1228.9 (+0.3%); every threaded run is above every baseline run at each depth. Correctness: NOT deterministic. Baseline text is the same in all 12 runs; 5 of the 12 threaded runs produced a different text, each one different, and one was garbage (98K: decode 22.3 t/s, accepted 32/1434). That is a data race between the per-device submissions, i.e. the shared-state inventory this plan requires is not complete. The patch stays untested and off; it is not rejected. Next: bisect on the same binary at 8K (fusion off, CUDA graphs off, Q8_1 activation cache off, per-device arena off, async host inputs off, look-ahead off), three ABBAs per variant (chain c155), then fix the shared state the bisect points at inside 1356 or the owning patch.

2026-10-09 re-test of 1356 after the in-patch fix (graph capture/replay lock), build b-metamem-mdw2, chain c157, A = production, B = BIGCHERRY_META_DISPATCH_THREADS=1, order A B B A: output identical in every round (24K and 98K, two rounds each, plus a 2048-token decode at 24K); marker 0 1 1 0. Prefill 24K: A 1264.7/1268.9/1272.4/1272.8, B 1279.8/1279.6/1283.2/1282.5 t/s (+1.0%, every B above every A); 98K: A 1220.7/1227.4/1224.2/1224.5, B 1231.1/1227.9/1231.6/1226.9 (+0.4%). Decode and draft acceptance unchanged. Decision: promote 1356 (lightweight tier) once PR #56 is rebased onto engines/llamacpp/.

2026-10-09 two-build ABBA (chain c167, runs aux3off-r1/r2, A = b-main2 production, B = b-aux3 = production + 1328 with expert offload off, Flash-Next, ctx 245760, f16 KV): text identical in all four runs at each depth; prefill A/B t/s 8K 1233.7 1238.2 1248.0 1229.6 / 1269.2 1281.6 1266.2 1269.0 (+2.8%), 24K 1264.9 1271.7 1263.4 1272.1 / 1309.7 1305.4 1308.2 1309.7 (+3.2%), 98K 1225.0 1225.2 1221.7 1226.0 / 1260.6 1257.2 1258.4 1262.3 (+2.9%); every B above every A; decode and acceptance equal. The only always-on code in 1328 that can do this is its Meta split-state cache edit (erase the stale entry instead of clearing the cache). It is now its own patch, 1358_meta_split_cache_local_evict (PR #93, on by default, off switch BIGCHERRY_META_SPLIT_CACHE_EVICT=0); 1328 requires it. One-binary ABBA queued (chain c171, tag sce1). If it confirms, promote 1358 (lightweight tier) and check Qwen3.8-27B, since the patch is not model-specific.

2026-10-09 1358 isolated on one binary (chain c171, build b-metamem-sce1 = production + 1358, Flash-Next, ctx 245760, f16 KV; A = default on, B = BIGCHERRY_META_SPLIT_CACHE_EVICT=0, order A B B A, two rounds). Greedy text identical in every run at 8K / 24K / 98K; marker count at 24K 1 0 0 1. Prefill t/s 98K: A 1255.7 1259.4 1259.5 1263.5, B 1221.9 1216.0 1223.2 1223.1 (+3.1%, every A above every B); 24K round 2: A 1300.7 1308.9, B 1270.7 1267.2 (+3.0%); 8K round 2: A 1282.9 1248.6, B 1246.1 1217.6 (+2.8%, narrowest gap 2.5 t/s). Decode and acceptance equal. This confirms the two-build result and places the gain in this one edit. Remaining before promotion: Qwen3.8-27B no-regression (chain c175) and the promotion-gate decision.

### Threaded-paths program (owner, 2026-10-10)

Owner direction: try every threaded option and make sure the threads do not block each other; locks placed so the work does not end up serial.

Rule for every step: measure the lock before trusting the thread. A threaded path counts as parallel only when its lock counters show it (share of exclusive acquisitions, time waited, time held), not because a worker exists.

Order:
1. **1356 dispatch workers - measure the lock.** The 1356 lock is process-wide and exclusive for a whole capturing `graph_compute` call. `BIGCHERRY_META_DISPATCH_STATS=1` (PR #129) counts capturing against replaying calls and the wait/hold time, with the workers off and on. If prefill calls mostly capture, the workers are serial and the lock has to be narrowed to the capture/instantiate/update section, with the launch outside it.
2. **1356 on in the Flash-Next profile** only after step 1 shows overlap and the 1348 crash below is understood.
3. **1348 deferred catch-up on its own worker.** The drafter sits on its own card but its catch-up decode runs on the target's thread. First find the fault that surfaces there at ctx 245760 + ub1024 past about 118K tokens (needs 1348 and 1334 both on; clean with either off).
4. **1326 async inputs** (on in the profile): check its copies do not wait on the dispatch lock.
5. **Combination matrix** on Flash-Next at 8K / 24K / 98K: workers x async inputs x deferred catch-up, with the counters on, repeated-run identity for every combination.
6. **radiance two-XTX exchange:** one host thread a card already; check whether the two directions of the pinned-host exchange overlap (it was 33% of two-card time on MXFP4).

Shared-state inventory, persistent workers, default-off flag and identity + repeated-run + stress evidence apply to each new thread, as before.

## Change Log

- 2026-10-07T00:40:00.113369+00:00 (created-by): Created by agent
- 2026-10-07T00:58:28.837689+00:00 (updated-by): Updated: section:title, priority='P2', section:description, section:steps, section:validation
- 2026-10-08T19:11:36.630281+00:00 (updated-by): Updated: section:notes
- 2026-10-09T04:14:45.394562+00:00 (updated-by): Updated: section:notes
- 2026-10-09T04:36:15.933069+00:00 (updated-by): Updated: section:notes
- 2026-10-09T06:25:21.380514+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261009_222135_prefill-is-about-3-faster-on_1298
- 2026-10-09T22:21:46.034434+00:00 (updated-by): Updated: section:ledger-events
- chg_20261009_222149_optional-threaded-per-device-d_5728
- 2026-10-09T22:21:53.512776+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-10T12:13:58.140476+00:00 (updated-by): Updated: section:notes
