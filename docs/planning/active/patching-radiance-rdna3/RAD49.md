---
id: RAD49
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-08T11:39:12.700554+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# XTX TP2 + R9700 expert-helper Flash-Next topology — measured no-go and owned follow-ons

## Description

RAD49 tested dense/attention TP2 on two gfx1100 RX 7900 XTX cards with the R9700 as remote-MoE expert helper. The Nov 9–10 evidence records an explicit negative measured decision: about 1,945 vs 3,100 prefill tok/s and about 72 vs 46 ms per six-stream step against R9700-engine/XTX-helper layouts. The three-card comparison also had different context limits (6x65K vs 8x262K), so capacity is not a speed ratio. These profiled shares and results describe this configuration only. Do not promote the layout by default, infer Strata dual-XTX results, or launch TP3. This item remains pending until every unique residual acceptance is verified against successors.

## Steps

1. Preserve and reference runs/rad49-findings-20261109.md, runs/rad49-3card-split, runs/rad49-tp-prefill-prof and runs/rad49-4gpu-regress. Separate delivered protocol work from hardware decision and identify current merged/source-only status.
2. State the measured no-go with source/model/quant/ABI and context-capacity caveat. Keep profiled all-reduce (~22–23%), helper wait (~15%) and decode GEMM shares diagnostic; do not add them as prospective savings.
3. Map any residual placement/cost attribution to ENG01/RHP04/RHP09/RHP13; local handoff to RAD29/RAD30/RAD54; combine/encoding to RAD23/RAD41; BAR-push only RAD51 under HL02/RAD50; TP host exchange to HL05; phase-specific prefill only to ENG09 if later proven unowned; Strata engine evidence to BCOP37/MEN29 if supported.
4. Audit live item acceptance and linked reviews for every proposed successor. Keep RAD49 pending with only exact uncovered closure work while any unique criterion remains. Supersede only after all residual requirements have explicit live owners and reviews are handled.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Host-only verification records current source/ABI and exact evidence paths, context confounders, merged protocol references, current production layout and every successor's live scope. No GPU retest is part of this reconciliation.

## Effort & Risk



## Standards



## Acceptance Criteria

The negative topology decision and limitations are explicit; no default promotion is claimed. Each residual requirement is either explicitly accepted by a live successor or retained as narrowly scoped RAD49 work. Preserve original receipts and reviews. Set superseded only after that ownership audit is complete; failure performance alone is not completion.

## Notes

No P2P (host-memory transport only). No hand-picked budgets: derive from runtime VRAM, expert sizes, heat. Kernel changes only in libr3/moe-remote or managed patches. MIT attribution for any Strata code.



2026-11-09/10 outcome (findings: runs/rad49-findings-20261109.md): TP2 on the two XTX loses to the R9700-engine layouts on every metric. flashnext-tp2xtx-r9700 vs flashnext-2xtx (A B A B, runs/rad49-3card-split): prefill ~1,945 vs ~3,100 tok/s, decode 6 streams ~72 vs ~46 ms/step, context 6x65k vs 8x262k. TP2 prefill cost: host-memory all-reduce ~22% + helper waits ~15% (runs/rad49-tp-prefill-prof). Best layouts: flashnext-4gpu overall, flashnext-2xtx on three cards. Current main equals pre-RAD49 main on flashnext-4gpu within ~1% (runs/rad49-4gpu-regress). Follow-ons split out: tail kernel (tuning-radiance-tail-kernel TK01-05), host link (run-radiance-host-link HL01-05), speed-aware placement (RHP17).

## Change Log

- 2026-11-08T11:39:12.700554+00:00 (created-by): Created by agent

## Hardware findings (2026-11-08)

Same workload (65536 context), live policy off, flashnext-4gpu as control:

| layout / build | doc prefill tok/s | decode 1s ms/step | decode 6s ms/step | 6x8k sessions ms/step | cold 27k TTFT s |
|---|---|---|---|---|---|
| tp2xtx-r9700, #770 (helper computes each expert per rank) | 1,563 | 41.2 | 127 | 104 | 16.5 |
| tp2xtx-r9700, v25 full-width-once | 1,865 | 40.8 | 80.8 | 70.1 | 15.8 |
| tp2xtx-r9700, v25 + engine native MoE default on | 1,954 | 32.3 | 75.3 | 64.0 | 12.7 |
| tp2xtx (R9700 head + 6900 XT tail), same build | 1,994 | 32.9 | 75.6 | 74.0 | 17.9 |
| flashnext-4gpu | ~3,590 | ~24.9 | ~47.4 | ~43-46 | ~7.9 |

Per-op profile of the TP2 XTX engine (engine only, --profile-ops): routed-expert GEMM (moe_gemm_q) 51% of decode / 44% of
prefill, cross-card exchanges 11% / 23%, int8 trunk GEMM 14% / 6%, hc_read (fp8 mixing matrices) 10% / 7%.
Exchange alone (r3-arbench, exact wire): ~28-31 us at 1 row, ~1.7 ms at 2048 rows (wht6 0.67-0.85 ms).

Next, by expected gain: (1) native RDNA3 integer expert kernels (int8 activations; plain-i4 container converted,
KL gate pending; w4nl needs a two-pass integer form), (2) re-enable the fused gather + all-reduce with helpers (rad0002),
(3) per-layer consumer / combine choice from measured link rates (calibration at the measuring start, then live from
wait counters carried in the existing heat snapshot - one telemetry path), (4) hc mixing matrices int8 in the recipe.
The RX 6900 XT tail does not pay behind TP2 yet (its expert compute ~3.9 ms a request).
- 2026-11-10T00:11:22.301887+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4238
- 2026-11-20T02:23:02.849033+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria
