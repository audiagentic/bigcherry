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

2026-11-28, bisect run (bigcherry-2, 15:45-16:17 AEDT). 2020.8.5's 360.8 t/s is a stored record (ab-tp2 T1/T2 seed1, 2026-11-23), not a sealed release identity.
Setup: qwen27b-tp3 (XTX, XTX, R9700), the 2020.8.5 shipped native config pinned for every arm (exact wire, depth 3, max-num-seqs 8, pipelined prefill min rows 256, the all-reduce pin, fp8 KV), power profile stock in every arm, prompt SHA-256 matching the seed1 cohort; one managed start per arm, seeds 1-3, 8 streams, 512/192, 3 reps, temperature 0; ABAB, one short gpu lease per arm. 2020.24.1 adds libbciqr ahead of libr3 in the kernel list (since 2020.17 the IQR rows live only there) and ships power efficient, pinned back to stock here.
Installability: GitHub retention has removed the bundles of most releases in range. Only 2020.11.1 and 2020.20.0..24.1 keep an archive, and 2020.11.1 has no receipt, so bc-release install refuses it. 2020.11.4 was installed from a cached tag download with its receipt, and 2020.24.1 from its release assets, each into its own isolated root. 2020.8.5 has no archive at all, so it was built from source with the release lane (radiance:stock:linux-radiance) in a standalone clone at the tag, under the work root. That is diagnostic evidence, not a release install.

| arm | decode ms/step | TTFT max ms | draft acceptance | running mean | aggregate t/s (context) |
|---|---|---|---|---|---|
| probe 1: 2020.11.4 (n=6) | 31.41 | 2227 | 0.707 | 5.00 | 350.0 |
| probe 1: 2020.24.1 (n=6) | 31.72 | 2253 | 0.711 | 4.97 | 347.2 |
| probe 2: 2020.8.5 fresh lane build (n=6) | 31.86 | 2261 | 0.714 | 4.97 | 346.7 |
| probe 2: 2020.11.4 (n=6) | 31.35 | 2260 | 0.708 | 4.98 | 348.0 |
| stored 2020.8.5 record (T1/T2 seed1) | 31.9-32.1 | about 2050 | 0.70-0.72 | 5.1-5.6 | 360.8 |

Medians over seeds 1-3 and both launches; decode ms/step spread within an arm is 1.5-2.0 ms (one 24.1 seed-2 run at 37.2 ms; its TTFT was 1632 ms). Seed 1 alone: fresh 2020.8.5 353.6 and 351.5 t/s, 2020.11.4 348.6 to 352.7, 2020.24.1 348.4 and 353.3.
Reading: the fresh 2020.8.5 build is level with 2020.11.4 and 2020.24.1 on decode step, TTFT, acceptance and running requests. The 2020.8.5 code does not reproduce its own stored 360.8 today; the record's higher figure is mostly a lower TTFT max (about 2050 against about 2250 ms, so more running requests per step), not decode time. No release in 2020.8.5..2020.24.1 introduces the short-prompt loss, so there is no boundary pair to bisect. One caveat: the 2020.8.5 arm is a local lane build, not the CI archive the record ran on (that archive is gone).
Power: the 24.1 shipped efficient profile could not be measured. bc-serve refuses to apply it (BC_CFG_CALIBRATION_INVALID: no verified apply, readback and restore receipt for these settings on this board and driver, qualify with --allow-power-apply --qualify-power). The two stock arms of that pair (2020.24.1, seeds 1-3 x 2 launches) give a median of 31.75 ms/step, 2259 ms TTFT and 346.1 t/s, consistent with probe 1. The latest production qwen27b-tp3 run's power receipt shows profile stock, mode observe, so production does not run efficient on the cards either.
Scratch, results and the isolated roots are in <work-root>/vtp61 (probe1, probe2, power); the roots and the build directory are deleted once the power question is settled.

## Change Log

- 2026-11-27T23:02:09.372590+00:00 (created-by): Created by agent
- 2026-11-27T23:02:30.989082+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
