---
id: KAB01
order: 0
plan: run-kernel-ab
state: pending
created-at: '2026-11-04T08:36:20.555693+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Kernel A/B harness: replay, time and compare kernel libraries and builds in minutes (rad-kbench wrapper)

## Description

Trying a kernel idea today means an end-to-end A/B: build, start a server, run streams, read ms/step. rad-kbench (in every radiance build) already replays recorded kernel fixtures against the reference, times each device kernel with cold-weight rotation and a 1% median, and can load any subset of kernel libraries; it needs no model and runs in minutes. What is missing is the layer above it: a repeatable way to record a fixture for a given container, run two sides (two kernel libraries, or two builds, or one library before and after a change), and get one joined table of per-case time and correctness. Needed now for: libr3d (upstream RDNA3 plugin) vs our libr3 on the 27B dense GEMMs on the XTX cards; the RAD04 all-reduce/GEMM overlap and any RDNA3 prefill kernel work (the 27B profile showed GEMMs at about 59% of peak, 52-54% of prefill time); the RAD46 rows kernels; any future quantisation format. This item builds that layer on rad-kbench and the existing tools/lab/radiance/kbench_step.py (which already joins a graph to a timing report), reusing rather than replacing them.

## Steps

1. Record fixtures: a documented, scripted recording per container (skeleton + --record; no card needed): the 27B MXFP4 container first, then Flash-Next, FP8 27B. Store fixtures under the build work root keyed by container digest and radiance pin, not in git; the recipe is in git.
2. Define the side: a side is {radiance_home or build id, --kernels list, optional env}. Provide a build variant that includes libr3d next to libr3 (the lane build set currently produces libr4d + libr3 but not libr3d), keeping the BigCherry components, so both can be loaded from one home with --kernels choosing.
3. A/B driver (tools/lab/radiance/kbench_ab.py or a bigcherry subcommand): runs rad-kbench --timing-only (and optionally the checked --bench run) for side A and side B on the same fixture and op filter under the gpu lease, parses the two reports, joins them on (op, kernel, geometry, operand shapes), and prints/writes one table: median us each side, ratio, correctness margin each side, rows not reached, and a time-weighted summary per op family. Join and arithmetic are card-free and unit-tested on recorded report text.
4. Use it: libr3d vs libr3 on the 27B MXFP4 GEMM families (decode M=1..8, prefill M>=256) on a 7900 XTX; record the table in this item.
5. Make it the standard first step for any kernel change: a short doc in tools/lab/radiance/README.md and the bigcherry-benchmark guidance (kernel A/B before end-to-end A/B), with the rule that an end-to-end run is only taken for kernels that win here.

## Detailed Solution & Technical Design

rad-kbench facts (from its help): replay mode needs no model and no card for the reference; --timing-only times every device kernel and checks nothing, intended for comparing builds a full run has already checked; --bench-flush rotates weight copies over a footprint so weights are read cold as the engine streams them; --bench-reps/--bench-warm-ms control convergence and clock ramp; --kernels A:B selects libraries; --op filters op families; --all-rows exercises every candidate row, not only the selected one; --max-cases limits repeats of a distinct geometry. Fixtures are keyed by reference answers on seeded inputs, so they are machine independent.

Design points. (a) Compare like with like: both sides use the same fixture, flush footprint, seed and iteration policy; record them in the output. (b) Correctness first: a faster kernel that diverges from the reference beyond its tolerance is shown as divergent, never as a win. (c) Time-weight by model usage: join the report to the model graph with kbench_step.py so the summary says how much a per-case ratio is worth per step (calls per layer times layers), not just a mean of ratios. (d) MoE ops are timed on Zipf-skewed routing in rad-kbench, not real traffic (kbench_step.py already notes this): keep MoE conclusions to the replay-kernel-time facility. (e) The fixture recorded from a different container of the same architecture (for example the FP8 27B) reaches different rows than the MXFP4 one, so record per container.

## Code Samples & Guidance



## Files

tools/lab/radiance/kbench_ab.py (new, with the join logic importable), tools/lab/radiance/kbench_step.py (reused/extended), tools/lab/radiance/README.md, tools/tests/lab/test_kbench_ab.py (card-free, recorded report text as fixtures), a lane/build variant that includes libr3d (engines/radiance/recipes.toml platform targets or a lab build option), docs/reference/testing/ (kernel A/B procedure).

## Validation

Card-free: join and ratio arithmetic on recorded report text, a divergent case never reported as a win, rows only on one side listed, op-family summary weights. Hardware: libr3 vs libr3 (the same side twice) shows ratios within the 1% convergence bound; libr3d vs libr3 on the 27B MXFP4 GEMMs gives a table whose decode and prefill rows can be reconciled with the engine's own per-op profile (--profile-ops) within noise; the whole A/B for one op family runs in under 15 minutes of wall time.

## Effort & Risk

Medium. The risk is a misleading comparison: different fixtures, warm vs cold weights, or libr3d rows that decline the same geometry and fall to a different kernel, so the join must show unmatched rows explicitly. libr3d may not declare all the ops (it lacks w4a8 MoE) so it can only be compared on the dense families. Building libr3d next to libr3 in one home depends on the plugin loader's first-library-wins order, so --kernels must be explicit.

## Standards

Managed tooling under tools/ with card-free tests; GPU measurement under the gpu lease only; no machine names or local paths in committed files; no compatibility shims; fixtures and reports stay in the work root, only recipes and results summaries go in git; kernel changes themselves stay managed patches or components.

## Acceptance Criteria

1. Given two sides and a container, one command produces the joined per-case table plus an op-family summary and records the fixture, seed, flush footprint and build ids it used. 2. A fixture for the 27B MXFP4 container is recorded by a scripted, documented step. 3. The libr3d vs libr3 comparison on the 27B GEMM families is recorded in this item with the verdict (adopt libr3d for which ops, or not). 4. The README names kernel A/B as the first check before any end-to-end kernel A/B.

## Notes

2026-11-04: scratch build of upstream 1.16.0 for gfx1100 (libr3d + rad-kbench + libref) started under the work root; MXFP4 27B skeleton recorded (7 MiB for a 19 GB container); fixture recording in progress. Related: RAD04 (reopened; overlap the all-reduce with the next GEMM), RAD37 follow-up (libr3d A/B never done), RAD15 (kbench_step), BU01.

## Change Log

- 2026-11-04T08:36:20.555693+00:00 (created-by): Created by agent
- 2026-11-04T08:36:54.402247+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20261104_083700_a-plan-item-now-tracks-a-fast_9840
- 2026-11-04T08:37:08.326286+00:00 (updated-by): Updated: section:ledger-events
- chg_20261104_110929_a-tool-now-runs-and-compares-k_4952
- 2026-11-04T11:09:33.611568+00:00 (updated-by): Updated: section:ledger-events
