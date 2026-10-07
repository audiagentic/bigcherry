# 1343_mtp_nextn_rereserve

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-07). Mechanism is in
SUMMARY.md: with `BIGCHERRY_MTP_RERESERVE=1` a change of the NextN (MTP) output mode asks for one scheduler reserve,
so the graphs that run bind into a worst-case plan of their own shape instead of being re-planned as the context
fills. Opt-in (default off).

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), ctx 245760, f16 KV, ub512, build `b-metamem-rr98` (production set + 1343).

- Activation (`BIGCHERRY_PATCH_TRACE=1`, run `metamem-rr98`): one hit per context, `value=1 masked=0` (target) and
  `value=1 masked=1` (draft).
- Mechanism (1340's counters, 98K fill, per-device arena on): new layouts per device 246 without the flag
  (`metamem-msm2t98`), 0 with it (`metamem-rr98`); the arena sizes are unchanged and `grows=0`.
- Speed (`tools/lab/flash-next/queue-env-ab.sh`, run `rr-ab`, ABBA on one binary, A = production / B = flag on):

| Depth | Prefill A (t/s) | Prefill B (t/s) | Decode A (t/s) | Decode B (t/s) | Acceptance A / B |
|---|---|---|---|---|---|
| 8K | 1073.1 / 1071.0 | 1077.0 / 1078.9 | 84.1 / 85.3 | 83.7 / 83.5 | 351/479 / 348/486 |
| 24K | 1072.4 / 1071.5 | 1077.9 / 1079.9 | 75.2 / 75.3 | 75.4 / 74.8 | 345/498 / 347/490, 345/496 |
| 98K | 985.3 / 988.7 | 998.5 / 997.8 | 57.9 / 58.3 | 60.6 / 60.5 | 346/493 / 353/471, 352/474 |

  Prefill is higher with the flag on every run at every depth (complete separation, n = 2 per arm): +0.5% at 8K,
  +0.6% at 24K, +1.1% at 98K. With fusion off (`rr-nofuse`): +0.5% at 8K (1030.1, 1029.2 / 1035.1, 1034.7) and
  +0.9% at 24K (1031.1, 1031.7 / 1041.5, 1040.5).
- Identity with fusion off (`GGML_CUDA_DISABLE_FUSION=1` on both arms, `rr-nofuse`): greedy text identical at 8K and
  24K. The change computes the same values.
- With fusion on the generated text is NOT identical to production (one md5 per arm at each depth): the plan the
  graphs bind into is another layout, and `ggml_cuda_check_fusion_memory_ranges` decides by address overlap, so
  another set of nodes is fused (FKE01). At the probe positions the distributions are identical to production
  (`rr-ab`, `rr-ref`: 24 probes, top-1 24/24, TV 0.0000) and equally far from the CPU f32 reference (both top-1
  21/24, TV mean 0.0950). The decode figures above therefore compare different texts (acceptance differs) and are
  not attributed to the patch. Stated equivalence, not identity.
- Offline: package tests (apply, idempotence, reserve asked for only on a real change and before the assignment,
  fail-closed), patch-lint, production composition check.

## Native llama.cpp comparison

Arm A's setter is native llama.cpp b11402's, unmodified, so the ABBA is the native reserve behaviour against one
extra reserve inside one BigCherry binary. No separate run against a fully native binary was made for this patch.

Not in a runtime profile yet: switching it on changes production's generated text (fusion set), which is the owner's
call, together with 1340's flags.
