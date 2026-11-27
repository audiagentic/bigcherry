---
id: VTP61
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-27T23:02:09.372590+00:00'
breadth: ''
skill: ''
created-by: agent
work: S
priority: P1
---

# Post-bump: bisect the TP3 short-prompt throughput loss across releases 2020.8.5 -> 2020.24.1

## Description

Qwen 27B IQ8R TP3 (exact wire, eight streams, seed1 512/192 cohort) measured 360.8 aggregate t/s on 2020.8.5, 351.3 on installed 2.2 2020.24.1 (-2.6%) and 342.4-344.5 on the 2.3 candidate. Same-session 2.2 vs 2.3 decode ms/step is 37.36 vs 36.75 with equal draft acceptance (0.70), so the loss is not decode kernel time; the TTFT and mean running requests point at prefill/admission. Long 4K/256 improved from 90.7 (RUM06) to about 107. Find the release that introduced the short-prompt loss, name the change, and decide fix or accept. Runs after the 2.3 bump merges; not a bump gate.

## Steps

1. List published bc-radiance releases 2020.8.5 .. 2020.24.1 and their archive ids; skip identical archives.
2. Install the endpoints and the midpoint into isolated roots (bc-release install, never the production root); bisect by halving.
3. Each probe: the TP3 qwen27b layout, the seed1 512/192 eight-stream cohort from historical-cohort-reconciliation.json, effective config pinned to the historical arm (exact wire, same scheduler flags), three seeds, ABAB against the previous probe under one short gpu lease.
4. Report decode ms/step, prefill/TTFT and draft acceptance separately; aggregate t/s only as context.
5. When the boundary is two adjacent releases, read their changelog and diff; attribute with a native phase trace on the two builds.
6. Record the verdict: fix (new item on the 2.3 pin) or accept as the cost of a deliberate safety/correctness change, with evidence under docs/evidence.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Each probe binds release tag, archive id, effective config receipt and prompt hashes; matched counts; at least two launches per arm at the boundary; no tuning changes inside the bisect.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Boundary release pair identified, or documented that the loss is spread across releases with per-step figures.
- [ ] Cause named from the diff and a phase trace, separating decode step time from prefill/admission.
- [ ] Verdict recorded (fix item opened on the 2.3 pin, or accepted with reason) and VTP59 updated.
- [ ] Evidence committed under docs/evidence by PR; scratch and isolated install roots deleted.

## Notes

Created 2026-11-28 by owner decision: finish the 2.3 bump against a no-regression gate versus installed 2.2, then investigate the historical gap. Inputs: docs/evidence/2026-11-27-radiance-230-bump/tp3-historical-recovery.json, historical-cohort-reconciliation.json. Measure with ms/step and TTFT separately; cross-version aggregate t/s follows draft acceptance. Short gpu leases with --wait. Suggested owner: bigcherry-2.

## Change Log

- 2026-11-27T23:02:09.372590+00:00 (created-by): Created by agent
- 2026-11-27T23:02:30.989082+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
