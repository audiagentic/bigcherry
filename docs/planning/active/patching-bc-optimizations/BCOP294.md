---
id: BCOP294
order: 294
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-18'
created-by: agent
priority: P2
work: S
---

# PRBE28: preserve validated 1347 thin-F32 dispatch under prospective row padding

**Discovery:** The unmerged llama.cpp [#26621](https://github.com/ggml-org/llama.cpp/pull/26621) loader would flag `hc_attn_inject`/`hc_ffn_inject` as float `MUL_MAT` weights. Their measured 1347 shape `[10240,4]` has 40960-B packed rows (20×2048). An enabled 128-B row pad changes `nb[1]` to 41088, so `ggml_is_contiguous(src0)` becomes false and BigCherry's default-on 1347 thin-F32 MMVF path is bypassed. No padding port exists on BigCherry and upstream defaults only gfx1151; **conditional source-level compatibility hazard, not a reproduced regression**.

**Owner/consolidation:** PRBE28 owns original float-weight padding and this gate. Completed QFP34/1347 retains the optimized kernel; PRBE29/30/31 own separate quantized-weight F16 shadows; PRBE70 owns typed BLAS cost evidence. No new allocator, selector, patch, experiment queue or duplicate technical plan. 1347 SUMMARY records the dependency; PRBE28 contains all technical design.

**Branch/12-hour/novelty:** Independently active `main` advanced to `c06dd0a9b1c26ba40866158c663ae1c116947810` (2026-10-18 04:08:34 UTC), newer than `feat/rad06-vecio` `cbc6bb9d2e0f29f00562b48b372f13b55e7bf83c` (03:49:56 UTC). Inspected recent main and RAD06 branches/PRs: no independent PRBE28/1347 row-padding change or queued experiment in the preceding 12 hours. **Protected/excluded:** RAD06 MoE FP16 WMMA, AHALF, VECIO, long-row split, helper replay/GEMV, profile and qualification; untouched. Novelty versus 2026-10-09 PRBE28 audit is the newly reconciled **1347 dispatch/loader-flag composition**, not another cache-alias hypothesis.

**Measured boundary:** b11402 Brutus 1347 ON vs OFF prefill +3.1–4.6% (MTP) and +6.8–7.1% (no draft), not a row-pad A/B or current-pin result. No measured row-padding gain, ISA/occupancy or physical traffic. External [AMD #57](https://github.com/AMD-Ecosystem/llama.cpp/pull/57) row-stride helper does not relax 1347's own contiguous guard; vLLM RDNA3 W4A16/GPTQ is a different format.

**Cheap check:** 10/10 source-shaped host dispatch cases and 4/4 row-byte/alias checks passed, including packed/padded `[10240,4]`, width boundaries, negative types and flag off. No repository tests, HIP build, GPU profiling or benchmark.

**Unresolved action:** On a current-pin typed post-Meta per-GPU float-BLAS census, first exclude any weight that would hit 1347 when packed. Reject `1347_ROUTE_CONFLICT` rather than silently changing its dispatch. If a different float-BLAS alias class remains, require measured cache conflict, exposed serialized share >2.9126%, safe per-device memory/upload and default-off patch-composition fixtures before a GPU A/B. Then require four independent gfx1100/gfx1201 sessions, >=10 ABBA pairs, CI95-low >=3% E2E prefill, <=1% decode regression, logits/KLD/greedy, graph/multi-request and exact physical work. Otherwise close `NO_BLAS_FLOAT_CANDIDATE`, `NO_ALIAS_EVIDENCE`, `BELOW_CEILING` or `NO_NET_GAIN`.

**Terminal:** `BLOCKED_1347_DISPATCH` pending a safe census; `NO_CANDIDATE`/`BELOW_CEILING`/`NO_GAIN` if rejected; `QUALIFIED_FLOAT_PAD` only after all gates.
