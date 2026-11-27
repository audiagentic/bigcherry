---
id: VTP60
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-27T10:22:17.436914+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: M
---

# Promote fast scheduler and model-layout qualification to canonical gates

## Description

Turn VTP59 host-burst proof into repeatable, provenance-bound fast tests. Reuse actual native scheduler, CTest, RCS26–28 affected-family resolution, existing replay/cost captures, MRC08 and MEN17. Do not own production tuning or replace integrated serving gates.

## Steps

S1 consolidate native host build/CTest: retain regression assertions and retire ad hoc link launcher; validate fresh combined run. S2 native effective configuration and workload descriptors, no duplicated defaults. S3 reuse rad-tune for declared kernel variants, rad-kbench for independent correctness and isolated replay, rad-info for model/graph scope and native --diagnose for actual phase/serving costs. Extend upstream sched_test sim_burst minimally for mixed-step timing, injected clock, versioned costs, structured schedule records; reuse actual scheduler and existing VTP42 governor. Deterministic policy search is orchestration only; RCS26–28 select affected families. S4 route receipts through MRC08 and MEN17 exact candidate gates. S5 mutation, repeated schedule hashes, held-out hardware correlation, final quality; retire redundant adapters only after equivalent coverage and evidence are verified.

## Detailed Solution & Technical Design

See docs/design/radiance-fast-qualification.md sections 1–9: exact native CTest, effective defaults, workload selection, measured-cost replay, evidence routing and acceptance. S1/S2 independent of GPU and MRC08; S3–5 consume existing owners. Review findings incorporated.

## Code Samples & Guidance



## Files

docs/design/radiance-fast-qualification.md; existing owners only. VTP59 retains tuning, RCS26–28 retain kernels, MRC08 retains record/export and MEN17 retains final acceptance.

## Validation

Native exact build, fake linkage, no model/GPU; fail missing/skipped tests, stale core, malformed cost and unknown geometry. Mutation controls for known overlap loss, default drift, slot/admission/cancel/count; validate cost uncertainty, actual row evidence and held-out hardware. Verify existing record/export privacy, exact candidate/reference, failed host gate prevents GPU. See design §8.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Exact composed native CTest build, effective default gate, missing/skipped fail, mutation controls and fresh combined run.
- [ ] Use native rad-tune for declared kernel variants and rad-kbench for independent correctness/replay; retain RCS family selection and demonstrate equivalent coverage before removing wrappers.
- [ ] Extend native simulator rather than fork it; identical frozen scenario, cost table, clock, seed and policy produce identical schedule hash and metrics across repeated runs; missing/uncertain costs nonpass for qualification.
- [ ] Native diagnose calibration and held-out hardware correlation by model/layout, with predeclared throughput/latency/fairness budgets and independent numerical quality.
- [ ] Existing MRC08 and MEN17 exact candidate/reference, privacy, coverage and fail-closed gates; no duplicate tuner, controller, workload catalog or evidence store.

## Notes

Owner direction: reuse Radiance tooling first. Native 2.3 source slice 90bca121c9dcee4cac6cf2b41a8d7b3e verified; commit 724ebe40c replaced manual linking with attested native CMake/CTest. Seven native cases, 18,221 assertions, 0.21 s rerun; 31 Python checks; injected default drift rejected and restored. Initial host build 151 s. Fresh combined launcher and full profile audit remain. Upstream sim_burst supports seeded acceptance and configurable linear costs; mixed target and clock not yet wired. Prefer upstream-compatible changes over new BigCherry implementations. Evidence docs/evidence/VTP60-native-host/summary.json.

Native sim_burst extension implemented with mixed-step cost injection, strict synchronous burst completion, output traces and schedule SHA-256, frozen synthetic cost curves and Pareto search. 34 policies x 4 workloads x 4 cost hypotheses = 544 cells, two runs per cell, all hashes and metrics identical; rerun with stricter parser byte-identical. 40 Python tests passed. Evidence docs/evidence/VTP60-deterministic-screen/simulation-summary.json. No simulated age clock, real per-phase calibration, acceptance distribution or canonical gate integration yet; do not promote based on synthetic results. VTP59 hardware screen uses exact 2.3 candidate. Extend native simulation only, not a second scheduler.

2.3 candidate IQ8R TP2 prefill execution screen: six managed starts, 30 reports/90 one-token requests. First cache-enabled screen exposes 6K prefix reuse; corrected cold arms use no-prefix-cache=true. Median TTFT (128/512/2048/4096/6144): base 99.01/270.96/987.86/1991.60/3064.25 ms; pipeline on min256 100.29/220.50/755.67/1535.91/2430.31; same with 4096 batch 97.69/219.72/756.20/1486.87/2385.49. Best 4K TTFT -25.3%, 6K -22.2%; no short regression established. TTFT is not pure kernel timing. Scraped counters can lag/include calibration. Native rad-tune and rad-kbench CLI only accept bf16/i8 KV; native rad-tune command also denied by allowlist. Reuse shared set_build_kv_format for exact FP8 declaration, add TP ownership/pipeline settings and rank-aware geometry as needed, validate same serving graph and independent correctness before tuning cache use. Do not tune wrong graph. Next combine pipeline with prior scheduler finalist and test matched mixed arrivals, ABBA, quality; do not alter production yet. Evidence docs/evidence/VTP60-prefill-screen/summary.json.

KV follow-up: native rad0071_native_kv_tools patch commit 1bd1b5d45 on isolated bump-based node/native-kv-tools branch shares engine KV validation and resolver with rad-tune/rad-kbench, all 7 existing formats, capability refusal. Applied/idempotent against composed 2.3 source; both host tools and packed KV test compiled; native execution denied by allowlist; patch untested, not in production recipe. Remaining exact TP ownership/geometry and actual rank-aware tuning. 2.3 IQ8R TP3 8 streams, cache disabled, fixed 0/200ms/2048: FP8, bf16_i8, bf16_fp8, bf16 aggregate 512/128 279.415/285.169/278.786/281.212, 4096/512 180.170/178.669/180.664/176.294; no preemptions, clean stops, same prompt hashes/counts. Single repeat only. KV summary docs/evidence/VTP60-kv-format-screen/summary.json. I8/FP8_V4/K6_V4 queue blocked. Reuse ENG04/ENG06 evidence and RSC12: short and long 8k/32k/100k/248k where capacity allows, matched fixed physical capacity separately, quality KLD and retrieval against independent BF16 reference; assess kernel, prefill/decode, memory and power; no new formats unless existing ones leave a demonstrated gap. Do not promote format or tuning cache without independent quality and exact serving graph.

## Change Log

- 2026-11-27T10:22:17.436914+00:00 (created-by): Created by agent

## Reviews

- 2026-11-27T10:25:47.614063+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria
- RV4753
- 2026-11-27T10:53:09.767000+00:00 (updated-by): Updated: section:steps
- 2026-11-27T10:53:26.112245+00:00 (updated-by): Updated: section:notes
- 2026-11-27T10:53:34.034298+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-27T11:04:47.040381+00:00 (updated-by): Updated: section:notes
- 2026-11-27T11:17:37.522329+00:00 (updated-by): Updated: section:notes
- 2026-11-27T11:27:49.418209+00:00 (updated-by): Updated: section:notes
