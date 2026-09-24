---
id: PNRO01
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:01.168387+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Q8_0 wire format for the internal HIP AllReduce

## Description

TODO, NOT-READY (rescoped per GPT review from a false 'validate the existing implementation' framing). Verified via direct read of patches/1250_nro01_allreduce_q8_wire/patch.py: 1250 is explicitly a Q8 scaffold only -- its own comment states "Draft only: these kernels are deliberately not dispatched until NRO01's [implementation lands]", `p->nro01_q8_threshold` is set from env but never consumed anywhere, and no Q8 kernel is ever dispatched or wired into the AllReduce path; activation/wire-byte counters do not exist. Verified patch.toml: `requires = []` (not `1001_hip_internal_allreduce` as this item previously stated) -- b11126 already contains HIP internal AllReduce natively. The qualification steps below cannot run against the current source; real dispatch wiring must be authored first.

## Steps

1. Wire Q8 selection in ggml/src/ggml-cuda/allreduce.cu::ggml_cuda_ar_allreduce() (or the equivalent dispatch function -- verify exact name/location in the current AllReduce provider at implementation time) BEFORE the BF16/exact selection, gated on `p->nro01_q8_threshold`.
2. Launch the quantize kernel (bigcherry_nro01_quantize_q8_0_kernel, already authored but undispatched in patch.py) / Q8-transfer / dequant-add sequence for messages at/above threshold; account logical vs wire bytes; add activation/wire-byte trace counters (none exist today).
3. Retain exact/BF16 fallback for messages below threshold or on any Q8 ineligibility.
4. Remove the stale `--requires 1001_hip_internal_allreduce` from this item's own validation command language (patch 1250 itself already has `requires = []`).
5. Implement block-Q8_0 quantization, logical/wire byte accounting, Q8 finish/dequant-add, opt-in threshold override, and activation counters without importing residual fusion/P2P/graph changes.
6. Preserve exact-FP32 behavior when disabled and keep copy-engine/chunked paths intact; document uncovered size ranges.
7. Validate zeros, alternating signs, dynamic range, small values, non-32 tails, and independent rank inputs against CPU FP32 and exact internal reference.
8. Run model quality/error and greedy-divergence gates before timing.
9. Sweep decode/medium/prefill sizes and compare RCCL, exact, BF16, and Q8 arms; promote only an explicit size/topology envelope.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1250_nro01_allreduce_q8_wire; HIP AllReduce provider; static/package tests; Q8 correctness/quality/performance evidence

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1250_nro01_allreduce_q8_wire`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1250_nro01_allreduce_q8_wire --source bigcherry-tuning`; run the package's own pytest offline. Synthetic correctness: zeros/alternating-signs/dynamic-range/small-values/non-32 tails/independent-rank inputs vs CPU FP32 and exact internal reference. Model quality/greedy-divergence gate before timing. Hardware (Brutus only, not run here): CORRECTED -- `python -m bigcherry.patch.validation_campaign --overlay 1250_nro01_allreduce_q8_wire --arch gfx1100` (no `--requires 1001_hip_internal_allreduce`; patch.toml declares requires=[]) sweeping decode/medium/prefill sizes across RCCL/exact/BF16/Q8 arms; promote only an explicit size/topology envelope.

## Effort & Risk



## Standards

PATCH_SYSTEM/authoring/validation; immutable source SHA; fail-closed applicability; exact/BF16 fallback preservation; contract identity separate from runtime selector.

## Acceptance Criteria

Q8 is inert when disabled, tail-safe, numerically within preregistered tolerances, and activation/wire accounting is proven. A statistically supported winning size envelope exists versus exact internal with no unexplained control regression; no blanket default.

## Notes

Supersedes: NRO01
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro01

Supersedes: NRO01
Inherited semantic scope: preserve source identity, opt-in policy, numerical/tail correctness, four-arm matrix, and explicit-envelope promotion gates.
Migration: capability-rebaseline-v3-2026-09

REAL FINDING 2026-09-12: this draft had never been built before. First real hardware build attempt (gfx1100/Brutus, isolated scratch clone, requires=1001_hip_internal_allreduce composition) found a genuine compile failure -- two Edit anchors in patches/1250's patch.py ended at the '=' sign of a single-line C++ statement; insert_after splices immediately after the matched text, not after the enclosing statement, corrupting both `GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT`'s declaration and `p->bf16_threshold`'s assignment into unparseable C++ (real compiler errors: 'expected expression', 'use of undeclared identifier'). Fixed by extending both anchors to match the complete single-line statement (one needed this project's own LITERAL-placeholder technique to cross a noise-stripped string literal). Re-verified on real hardware: clean build, generated source inspected and confirmed well-formed. This closes acceptance criterion 1 ('Draft patch applies cleanly on b10705, builds HIP') for the first time -- it was never actually true before this session despite the patch being packaged and offline-tested. The 13 existing offline tests never caught this because they check structural properties, not actual compilation. Remaining acceptance criteria (numerical correctness matrix, activation/wire-byte evidence, model quality gates, 4-arm performance sweep) are all still genuinely not started -- this fix only makes the draft buildable, it does not wire the Q8 path to anything reachable (deliberately, per the patch's own inert-by-design scope).

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1250_nro01_allreduce_q8_wire exists, state=untested, kind=enhancement, requires 1001_hip_internal_allreduce (confirmed patches/1250/patch.toml). Prior session (2026-09-12) already found+fixed a real anchor/compile bug and confirmed clean gfx1100 build. No upstream b11126 equivalent found (Q8_0 AllReduce wire format is BigCherry-internal, not a llama.cpp upstream feature). Disposition: validate/qualify existing patch per its own steps; no design work / GPT needed (patch already implements the described scope, only qualification evidence is outstanding).

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified via direct read that 1250 is Q8-scaffold-only (kernels deliberately undispatched per patch.py's own comment, threshold var unconsumed, no activation/wire counters) -- the item cannot be closed by qualification alone; real dispatch wiring is required first. Corrected the stale requires=1001_hip_internal_allreduce reference (patch.toml declares requires=[]; b11126 already has HIP internal AllReduce natively).

## Change Log

- 2026-09-09T10:52:01.168387+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:30.603565+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.053474+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.690115+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:25:56.314497+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.269059+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:33:46.596780+00:00 (updated-by): Updated: section:notes
- chg_20260912_033450_documented-a-real-bug-fix-and_2722
- 2026-09-12T03:34:50.268598+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:25:40.011758+00:00 (updated-by): Updated: section:validation
- 2026-09-24T02:25:51.817653+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:47:44.678692+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes
