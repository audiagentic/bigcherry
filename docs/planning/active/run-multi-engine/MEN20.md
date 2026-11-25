---
id: MEN20
order: 20
plan: run-multi-engine
state: pending
created-at: '2026-10-18T13:04:07.928445+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# bigcherry command for component-setting evidence (replaces helper-evidence.sh)

## Description

Promoting a component setting (libr3 now, moe-remote/backends after RCS04) needs one A/B per setting with an activation record, staged as run.json under docs/evidence/<item>/ and referenced from component.toml. Today helper-evidence.sh does this by looping ab-benchmark. Make it a bigcherry command: `bigcherry component-evidence --engine radiance --component <c> --setting <S>` reads the setting's declared A/B config, builds the lane, runs ab-benchmark on a idle-checked host, verifies the evidence rules (arms differ in that setting alone, executed activation, no failed cell), stages the run.json and prints the component.toml edit. Optional --promote edits the status after verification.

## Steps

1. Declare setting-specific A/B configuration, activation marker, evidence identity and status in the existing component metadata format, consulting RCS08's lint contract; do not invent a separate registry.
2. Implement a thin `component-evidence` command that calls existing `ab-benchmark`, common sidecar lifecycle/idle checks and evidence verification. It must check arm identity (only the intended component setting varies), helper activation, failures and reproducible run metadata.
3. Stage the verified run and print or apply an explicit promotion update only when requested. Reuse the current promotion mechanism and fail closed on an ambiguous or invalid evidence record.
4. Run offline failure fixtures and one real setting run; compare its output with the historical `men11-fire-10171108` ROWS=0/1 8/8 clean run, without claiming the old throughput result as new hardware acceptance.
5. Retire `helper-evidence.sh` only if it still exists and only after the new command demonstrably replaces it; coordinate the inventory with MEN14.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline positive/negative run.json fixtures (wrong arm identity, absent activation, failed helper/cell, mismatched pin, and attempted invalid promotion); one real ROWS or equivalent setting run through the command, with identity-bound staged evidence and graceful teardown.

## Effort & Risk



## Standards



## Acceptance Criteria

1. One engine-aware component-setting command composes the maintained A/B harness and evidence verifier; no duplicate benchmark, worker launcher or patch registry is created.
2. Stage a full identity-bound evidence record (setting, baseline, model, pin, build, per-arm variables, activation, results and checks) and reject incorrect setting variation, missing activation and any failed cell.
3. Promotion/default-on requires explicit request and passing evidence; failure leaves the component status unchanged and auditable.
4. A real helper-setting run reproduces a valid MEN11-class A/B contract and preserves the prior `men11-fire-10171108` eight-cell result as historic evidence, not as an unmeasured fresh pass.
5. Offline tests exercise activation, record validation, worker failure/teardown and promotion denial. Retire any redundant helper-evidence wrapper only after equivalence is demonstrated.

## Notes

Part of MEN14 consolidation. Component lint (RCS08) reads the same evidence.



BACKLOG TRIAGE 2026-11-04 (MERGE into RCS08; revalidated against main 6eb31a708, verdict CONFIRM): RCS08 (component status/evidence CLI) exists and is open; no duplicate command. Evidence: RCS08 (component status/evidence CLI) exists and is open; no duplicate command. Still referenced by: RAD10, MEN14, WSL05. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: MEN20.md:18-31 specifies a bigcherry component-evidence command that launches, verifies and stages a per-setting run and optionally promotes it. RCS08.md:18-27 only defines lint on existing setting declarations/evidence, with no runner or promotion command; the only literal component-evidence verb in the current tree appears in MEN20's plan.

CONSOLIDATION 2026-11-22: owns MEN11's pending component lifecycle/promotion and helper evidence staging; MEN14 consumes it for release flow, RCS08 only lints produced records, RAD08/RAD07 own KLD/documentation. No duplicate helper-launcher implementation.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. bench/workloads.py _manifest_for_server reads hardcoded flashnext PID and --manifest from cmdline; after S18 this breaks provenance. Replace with actual existing owner status/control socket and run-dir receipt, verify PID/start time and bound endpoint/current generation. Explicit evidence path read-only; no newest record or second registry. Ad hoc unknown provenance may be reported but cannot pass candidate acceptance. Migrate all workload/benchmark callers with MEN21 and RSC11; test stale PID, two owners, wrong port, stop during measure.

## Change Log

- 2026-10-18T13:04:07.928445+00:00 (created-by): Created by agent
- 2026-11-04T03:47:27.994982+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:47:31.963094+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:45:22.125658+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:26.281835+00:00 (state-transition): State: superseded → pending

## Reviews

- RV4469
- 2026-11-22T09:19:09.398060+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- RV4666
- 2026-11-25T03:30:47.320533+00:00 (updated-by): Updated: section:notes
