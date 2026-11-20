---
id: THA16
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:01.518345+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Critical-path/slack vs contention hypothesis for isolated-vs-e2e tuning divergence

## Description

Determine whether isolated-versus-E2E tuning divergence is caused by critical-path/slack and contention effects, using the real GPU0/27B production path and controlled host/API evidence.

## Steps

1. Reproduce the GPU0/27B mismatch with matched binaries, workload, topology, and diagnostics settings.
2. Separate GPU kernel time from host dispatch/API and contention time using rotation-controlled rocprofv3/kernel/API traces.
3. Compare isolated and E2E arms with identical tuned/native provenance and cache state.
4. Quantify signature/launch shares, critical-path slack, overlap, and contention rather than inferring from family counts.
5. Test candidate mitigations only when the trace identifies an attributable mechanism; preserve native baseline and no-regression controls.
6. Record negative results and tool-fit limitations explicitly.

Conditional candidate follow-ups consolidated from THA09 and THA14 (2026-11-20); do not execute until rotation-controlled traces establish the corresponding dispatch work is exposed on the critical path:
- THA09 lazy native_select: reuse the completed memoized provider/counters; never reimplement. Only in an explicit experiment mode may both `!native.valid` guards (dispatch_try and dispatch_resolve) move after signature/L1. Make shadow validity reachable and non-vacuous so fixtures/signatures can show both agreement and disagreement. Preserve force-once, native fallback and fail-closed invalid handling; prove L1 hits never force. Compare control/experiment with structural counters, graph/correctness, diagnostics-off E2E and interleaved production-like runs.
- THA14 graph-lifetime binding: establish invariance across type/extents/strides/fusion/device/replay generation/cache reload. Store resolved Binding in graph/node metadata with explicit invalidation; use canonical prepared facts for selector/signature consistency. Do not cache tensor pointers unless invalidation covers shape/type, buffer identity/usage, allocation size and view_src. Prototype one family only after attributed host-path opportunity; widen only after correctness, graph capture/replay and E2E proof.

Conditional operator-family candidate backlog consolidated from THA19-THA23 (2026-11-20); this is not authorization to implement or search all families. First require THA18’s decomposition/pilot, measured exposed critical-path headroom, admissible real variants and correctness proof. Then select only the justified family:
- RoPE: map real rope.cu variants and stable resolved-config identity; reuse HI67/HI143/HTR01 evidence; if no real variants, create a separately scoped kernel-authoring proposal rather than synthetic tuning.
- Norm: inspect norm.cu and identity; native telemetry first; candidate only if real variants exist; preserve matmul boundary and gfx1100 correctness.
- Flash attention: wait for THA18 and a completed RoPE/norm pilot; inspect KV-cache, mask and precision structure; native telemetry before candidate search; maintain attention-specific correctness/behavioral coverage.
- Quant/dequant: establish distinct dispatch ownership versus THA17 BLAS conversion telemetry; record reuse/no-duplicate when same surface; no candidate search until distinct ownership is proven.
- Softmax/GLU: inspect standalone softmax/glu outside BLAS fusion; define stable config identity; keep fused GLU signature observation separate; native telemetry before candidates.
For all families, reuse maintained evidence pipeline; exact candidate identity, native-first controls and full correctness/behavioral/E2E gates precede promotion.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

HI172 trace tooling/evidence; E2E benchmark arms; rocprofv3/API analysis; contention and critical-path report

## Validation

Rotation-controlled dual-XTX evidence; kernel trace and HIP API trace; matched native/tuned controls; host dispatch overhead attribution; predicted-vs-measured E2E reconciliation; clean shutdown and provenance receipts.

THA16 base validation remains rotation-controlled GPU0/27B traces, matched native/tuned controls, host dispatch/API attribution, critical-path slack, predicted-versus-measured E2E and clean provenance. For the conditional THA09 experiment, prove both guard sites exercise, shadow validity detects agreement and disagreement, default mode behavior unchanged, force-once/fallback correctness, L1 hits never force, graph/correctness checks, diagnostics-off E2E and interleaved control; require the explicit sub-1% regression policy before promotion. For THA14, test graph reuse and replay-generation invalidation, binding equality/fallback, graph/non-graph correctness, buffer/view/device invalidation and host-dispatch/E2E benefit. No candidate runs before the base trace identifies its mechanism.

For every selected family, establish source-derived real variants and admissible shapes, stable identity, THA18 foundation and measured critical-path opportunity first. Validate native correctness and telemetry before candidate evidence. Preserve the specific source requirements: RoPE candidate/reference correctness and record telemetry; norm correctness/telemetry and full suite/gfx1100; flash-attention reference/PPL correctness across KV-cache/mask/precision plus positive E2E; quant/dequant ownership/boundary audit plus native correctness and telemetry; softmax/GLU native/candidate correctness, standalone-versus-fused boundary, behavioral and E2E evidence. No hardware performance promotion from planning cleanup.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The divergence mechanism is either evidenced or explicitly unresolved with bounded negative results. No tuning winner is promoted from isolated microbenchmarks when the critical-path/E2E contribution is unproven.

The divergence cause is evidenced or explicitly unresolved; dispatch candidate work is admitted only when measured traces place that cost on the exposed critical path. If THA09 is selected, both native-valid guards are delayed only in an explicit experiment mode, the shadow path is non-vacuous, default behavior and fail-safe semantics hold, and promotion passes correctness/graph/diagnostics-off E2E plus the sub-1% policy. If THA14 is selected, binding reuse is proven graph-invariant with explicit complete invalidation, canonical selector/signature facts, no unsafe pointer cache, and measurable correctness-preserving host-path benefit. No change is promoted from isolated microbenchmarks or counter-only results.

No operator family is automatically built or searched. THA16 remains the decision gate: only a real profile showing exposed critical-path opportunity, THA18 foundation, real admissible variant shapes and correctness proof can authorize one bounded family. Preserve distinct gates for RoPE, norm, flash attention, quant/dequant ownership, and standalone softmax/GLU; native remains default until exact identity, correctness, behavioral and measured E2E evidence pass. If prerequisites are absent, family remains unselected and its safety constraints remain available here.

## Notes

Supersedes: HI172
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi172

Supersedes: HI172
Inherited semantic scope: preserve GPU0/27B mismatch, kernel-versus-host attribution, rotation-controlled traces, and no-promotion-without-critical-path evidence.
Migration: capability-rebaseline-v3-2026-09

Conditional follow-ups consolidated from THA09 and THA14 on 2026-11-20 after review RV4248. The prepared manifest named THA15, but live scope shows THA15 owns immutable winner provenance/counterfactual replay; THA16 owns host dispatch attribution and candidate mechanisms only after trace evidence. Source requirements remain gated and unimplemented; THA16 remains pending.

Conditional candidate-family requirements consolidated from THA19-THA23 on 2026-11-20 after RV4249. Sources are superseded only as duplicate candidate-search trackers; this addition makes no implementation, hardware or performance claim. THA18, native-first telemetry, actual critical-path evidence and family-specific correctness remain prerequisites.

## Change Log

- 2026-09-09T10:50:01.518345+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:13.456800+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.923695+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_143622_cleaned-up-the-tooling-registr_2452
- 2026-09-09T14:36:22.958572+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.491373+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:24:22.861057+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.752279+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4248
- 2026-11-20T03:10:22.237486+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- RV4249
- 2026-11-20T03:14:38.640861+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
