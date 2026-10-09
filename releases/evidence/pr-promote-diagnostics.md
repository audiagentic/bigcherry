## What this slice does

Promotes five diagnostic patches to the production set as neutral enablers, so they can be used on the production
binary without a special build: 1319 (process_ubatch timing), 1320 (tensor-split graph_compute timing), 1325
(scheduler split timing), 1342 (fusion bisect: per-family fusion counters and `BIGCHERRY_FUSION_SKIP_OPS`) and 1346
(MTP prompt timing). Plan items: BPB01, QFP16, QFP31, FKE01.

Each diagnostic is dormant unless its flag is set. 1346 now contains only the qualified timing diagnostic; the
unqualified `BIGCHERRY_MTP_PROMPT_WINDOW` collector/replay/lifecycle mechanism and its environment variable were
removed from the package before promotion.

## Changes the production build?

- [x] Yes: hardware evidence below

## Offline checks

Production set composes at b11474 (53 patches, 0 failed); the experiment sets that named these patches are now empty
and compose; catalog, governance and the patches' own tests pass (47 tests); patch-lint reports nothing for the five.

## Hardware evidence

Brutus, Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900
XT), ctx 245760, pin b11474 (b9acf138).

- Builds `b-metamem-mig-diag` (production + 1319 + 1320 + 1325 + 1346) and `b-metamem-mig-fuse` (production + 1342)
  compiled clean for gfx1100 / gfx1201 / gfx1030.
- `BIGCHERRY_SUBMIT_TIMING=1 BIGCHERRY_MTP_PROMPT_TIMING=1`, 38,673-token prompt (run `metamem-mig-diag`): 297
  SUBMIT_TIMING, 108 META_TIMING and 594 SCHED_SPLIT lines; target per 512-token chunk graph build 17.2 ms,
  set_inputs 4.3 ms, graph_compute 50.5 ms (tensor split: rebuild 9.6, launch 20.6 for 97 subgraphs, AllReduce
  enqueue 2.1); `target_sync_ms=24979.7 target_fetch_ms=136.5 draft_decode_ms=2201.9`. Greedy text with the flags on
  equals the flag-off arm of the same binary (md5 fe307bdfb7e1); prefill 1167.5 t/s with the timing on.
- 1342 (runs `metamem-mig-fuse`, `metamem-mig-fuse-skip`): fusion counters per family at exit (MUL 1902, RMS_NORM
  10719, SCALE 8347, SOFT_MAX 1979, MUL_MAT_ID 1314, ...); with `BIGCHERRY_FUSION_SKIP_OPS=MUL` the MUL family
  disappears from the counters and the greedy text changes (94012812de5c against 2571b60b1505); without the flag the
  text equals production (2571b60b1505).

## 1346 scope correction

The original promotion branch still carried an unqualified default-off prompt-window mechanism. Owner policy selected
the QFP18 lightweight tier for the diagnostics but required package-wide consistency, so that mechanism was removed.
1346 now retains only `BIGCHERRY_MTP_PROMPT_TIMING` and the narrow prompt-start / target-process hooks required for
timing attribution. No window replay, draft suppression, cache/cancel lifecycle state or WINDOW activation marker
remains.

Hardware evidence above was collected from the timing path at b11474 with the window disabled; the timing values and
greedy identity therefore exercise the behavior that remains. The reduced post-review patch has not been HIP-compiled
on Brutus in this session; its mechanics/patch-lint status must be taken from the updated PR checks.

## Notes for the reviewer

No speed claim: these are diagnostics. QFP18 lightweight promotion policy applies. Full validation.toml /
evidence/validation.json campaigns are not required for this owner-approved tier.
