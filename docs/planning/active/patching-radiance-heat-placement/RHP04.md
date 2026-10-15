---
id: RHP04
order: 4
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Cost-aware cross-GPU expert ownership ranking in shadow mode

## Description

Use shared Radiance heat observations to recommend GPU ownership for R9700 trunk + two gfx1100 expert helpers, but **do not move weights or alter serving results**. Optimise the *critical path*, not simply counts: one busy remote helper can gate every token in a layer. Compare against existing `R3_MOE_HELPER_HEAT` coldest-first and initial static container profile.

## Steps

1. Define per-`(layer,expert)` owner candidates and strict eligibility from `rad0002` **runtime slot map** and `R3_MOE_HELPER_SKIP_OWNED`, protected BF16 experts, MTP layers, `min/max_rows`, full-width/TP geometry, input/output precision and exact helper source file availability. Never propose a helper assignment for an unsupported layer, expert encoding, or step-size fallback.
2. Consume phase-tagged prefill and decode route counts (RHP02); compute rolling/decayed heat and uncertainty with coverage min/age requirements. Separate active tokens per expert per phase, fraction of steps selecting it and distinct expert reads; raw counts alone do not reflect GEMM/reload cost.
3. Calibrate per-device curves from existing `--profile-ops` and helper per-stage timings (`RAD02/RAD05/RAD06`): local R9700 GEMM + host-pool miss, gfx1100 small-lane GEMV, large-lane grouped GEMM, publish/stage/add/fence, VRAM pressure. Include overlapping local and remote work, helper skew, p95 layer wait and transfer costs. Freeze curve version/provenance.
4. Create a deterministic budget-constrained offline optimiser producing a *whole ownership map*: respect helper VRAM limits, engine host+VRAM weights, whole expert bytes, protected/local nonmovable weights; preserve disjoint helper ownership. Compare predicted layer critical path and total request latency; favour low churn and low transfer cost. Explicit `no-change` outcome.
5. Shadow-run candidate calculations on a background CPU task on *copies* of bounded snapshots. No GPU work, blocking hooks, file writes or map mutation in routing/dispatch. Record recommendation and residual ranking error, never apply. Test representative prefill-heavy, decode-heavy, mixed and shifted distributions.
6. Bench existing static heat-based layout versus offline plan with restart/manual ownership selection in balanced ABBA; report calibration error, p50/p95 wait, prefill, decode, MTP accepted tokens, KL, GPU memory and bytes transferred. Gate deployment on real advantage beyond run-to-run variance.

## Detailed Solution & Technical Design

Radiance's existing decayed-LFU/hysteresis remains owner of **local residency**; RHP04 extends the policy decisions, not the route data. The objective should approximate `sum_over_layers max(T_local_compute, T_helper0_compute+transfer0, T_helper1_compute+transfer1)` with serial dependencies and MTP verify-specific costs; do not sum time of overlapped work. Use a penalty for placement churn; reject candidates with insufficient observed route coverage. No speculative reweighting of the model's router outputs.

## Files

`engines/radiance/components/libr3/helper/` or a dedicated adjacent policy source with card-free tests; experiment runner `tools/lab/radiance/`; `engines/radiance/components/libr3/component.toml` for experimental default-off knobs; `docs/evidence/` for reports. Reuse RHP02 hook and RHP03 snapshots.

## Validation

Synthetic oracle for max-not-sum, identical scores, empty/zero routes, competing helper, protected expert, VRAM full, phase mix, unsupported format, varying step buckets; stable reproducible mapping. Hardware: no serving code path mutation in shadow; paired ABBA, full logits KL vs reference noise band, counters for model/version/cohort, no more than 2% control overhead.

## Effort & Risk

High: cold experts may incur disproportionately expensive gfx11 prefill kernels; route frequency alone may bias allocation to the wrong helper. Predictions are provisional until independent served A/B.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Shadow candidate fits budgets and runtime maps exactly; deterministic with same evidence.
- Distinguishes prefill/decode and balances **per-layer** slowest GPU, not global share only.
- Rejects insufficient/stale heat and predicts measurable speed gain with explicit uncertainty; no change when unsupported.
- Default-off engine quality and runtime behaviour unchanged; baseline and candidate reported with matching prompts/config, KL and variability.

## Notes

Depends on RHP02–03; no automatic activation before RHP05.

## Implementation-grade notes (2026-10-15)

### Objective and constraints

Let \`r[l,e,p]\` be routed rows or demand estimate for phase \`p\` (prefill/decode). For each layer and device \`d\`, predict \`T[l,p,d]\` using measured kernel curves (startup + fixed per-expert weight read + variable rows/tiles + mailbox/merge + possible host-pool misses). Minimise

\`\`\`text
J(owners) = sum_phase demand_weight[phase]
              * sum_layer max_device(T[layer, phase, device])
          + relocation_penalty(owners,current)
          + uncertainty_penalty(coverage,fit_error)
\`\`\`

Do **not** sum concurrent engine+helper work as serial time. Model communication on critical path including helper wait, and count a remote expert only once per layer where applicable. For mixed steps, qualify independent curves rather than interpolating unobserved tensor shapes.

Hard constraints: each \`(layer,global_expert)\` assigned exactly once; \`allowed_owners\` derives from runtime protected expert map, output semantics, \`RAD_MOE_REMOTE_MIN/MAX_ROWS\`, source weight location and GPU ISA; per-device resident byte <= measured safe budget including launch scratch, KV/headroom, helper kernels and shadow-transfer workspace; rank parity and top-k routing identity unchanged. \`R3_MOE_HELPER_SKIP_OWNED\` means a local candidate may need restart reload; no illusion that a bitmap change is free.

### Calibration data and sampling

Record per-layer actual selected expert IDs/rows, per-device active routes, top-k count, per-expert nonzero token rows, hist coverage and duplicate count, occupancy, GEMM shape and padded tile size, helper stage latency (route/publish/gate_up/quant/down/merge), engine local MoE time and time waiting on helper. Cross-link each curve to engine build SHA, libr3 package digest, GPU UUID, ROCm version, model and numerical mode. Fit simple piecewise monotone upper envelopes by phase/shape/device (not high-dimensional ML), with conservative p95 prediction error. If too few samples for a bucket, refuse automatic change.

### Bounded production policy versus oracle

The included \`heat_mock.mjs::Planner\` enumerates single moves and 2-way swaps on **small synthetic models only** to serve as a correctness oracle. Real Flash-Next has tens of thousands of expert units: use a bounded, stable, deterministic candidate shortlist by \`heat * expected net ms\` and per-layer helper imbalance, then limited pair or slot-preserving swaps with strict time budget. Stop after no positive marginal improvement, hitting max swaps, or planner budget. Never enumerate O(N²) swaps in the production dispatch path.

### Replay and acceptance matrix

Cold startup profile vs hot-ranked static ownership vs RHP04 recommended; compare same model prompts and routing windows in prefill-only, decode-only, mixed, MTP verify, protected layers and decode helper-disabled. Replay phase-shift mid-run: prefill-heavy to decode-heavy, alternated and uniform. Report predicted/actual total and p95 layer waits, operator estimate error, bytes loaded, churn, source heat coverage, KLD/top-1 and speed ABBA. Proposed 5% predicted threshold is a search parameter; production gate is independent measured benefit with a confidence interval and no material quality regression.
