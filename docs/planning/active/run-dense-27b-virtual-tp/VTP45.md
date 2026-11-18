---
id: VTP45
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-18T07:19:07.714328+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P2
---

# Follow-ups from the VTP44 closing: routing determinism, libr3 selftest depth, geometry table, R9700-only test

## Description

Items deliberately left out when VTP44 closed (packed KV types validated on Qwen3.8-27B and Flash-Next; record/replay of the expert route in #1076 and #1089). Each is independent; none blocks VTP44's acceptance.

## Steps

1. Run-to-run routing nondeterminism on long documents (found 2026-11-18, FLASHNEXT-REPLAY-11DOCS.md): of four runs that selected their own experts on the 11-document KLD corpus, three agreed exactly and one diverged at document short-07, position 2,604, first in layer 19 and then 15 layers, persisting to the end of the document (999 token rows). `--deterministic` pins expert placement, not arithmetic. Find which kernel's arithmetic varies and under what condition (host load was high during the divergent run): record the same route N times with and without CPU load, dump the residual stream at layer 18/19 around that position across runs, bisect by layer and op. Decide: fix the kernel, or state the limit in the engine docs. Replayed runs are unaffected (byte-identical across runs and builds).
2. libr3 per-template selftest rewrite: replace the hand-listed (geometry, row, dtype) tuples and the `find 1 or 0` rewrite pairs in native/r4d_selftest.cpp.rw with a table generated from RAD_KV_LAYOUTS / RAD_KV_GEOMETRIES; add an instantiation inventory check (expected gfx1100 rows against the built plugin); remove or reduce the VTP43 model-constant allowlist entries. Deferred at VTP44's close (libr3 exports exactly libr4d's 608 device kernels, exclude.txt is empty, the rewrite exact-count checks pass, and a gfx1100 device selftest passes all 21 packed-KV cases, rad0042 #1126). Do this only if a new geometry is coming.
3. Geometry capability table: key RAD_KV_GEOMETRIES by (layout, head_dim, gqa, block size, operation, architecture) validated against the real row catalogue, and make declaration-time refusals name the whole tuple, with negative tests for an unsupported block size and a fused operation (GPT review finding 3; the row constraints already refuse later, so this is a diagnostic gap).
4. R9700-only (helper-free) engine test for the KV types: the launcher refuses --kv-dtype, --heat and --ngram-placement for a helper-free layout (a stored named config per KV type is the project's preference, not a flag), and the engine's start on that card reads the 47.7 GiB n-gram table through io_uring, which this container's seccomp profile refuses (the table does not fit in RAM beside the expert pool; this build has no --ngram-device). Needs the container run with seccomp unconfined, or a build with --ngram-device, and a layout variant per KV type.
5. rad0043 (`--expert-stats` labels every unit by its tensor name and counts the protected experts): qualify and promote it, since `--expert-profile` placement reads these counts and the old labels were shifted in layers with protected experts. Remove the RADIANCE_DEBUG_ROUTING environment hooks (rad_block_moe_fp8.h, bc_moe_remote_block.h, ctx.cpp): the route record supersedes them and they break the manifest-fields rule.
6. KLD evidence beyond the closing set: score natural-language prose and code separately from the project's id- and hash-heavy documents, and one long context at 32k; score an R9700-engine layout (needs its own record on that model container).
7. Layout smoke: a matched run of the earlier 9700 / 7900 XTX / 6900 XT numbers in the same layouts with the packed types, on a quiet host (the closing smoke ran while builds shared the host), more than one run per arm, the n-gram table in RAM.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per step: 1 a repeatability table (N recordings per load condition) and either a fixed kernel with a byte-identical record across runs or an engine-doc statement of the limit; 2 the inventory check fails when a row is dropped from a build, and the libr3 selftest case counts are unchanged; 3 negative tests name the full tuple; 4 the helper-free layout starts with each KV type and a short client run passes; 5 rad0043 promoted through the lightweight tier with evidence, the env hooks removed with the tests green; 6 KLD tables with prose and code split; 7 a table against the earlier numbers.

## Effort & Risk



## Standards



## Acceptance Criteria

- [ ] Step 1 decided: cause found and fixed, or the limit documented.
- [ ] Step 2 done or explicitly dropped.
- [ ] Step 3 done or dropped.
- [ ] Step 4 done or recorded as blocked on the container profile.
- [ ] Step 5 done.
- [ ] Steps 6 and 7 done or dropped by the owner.

## Notes

Created 2026-11-18 when VTP44 closed. Evidence: docs/evidence/2026-11-16-vtp44-generic-kv-layouts/ (FLASHNEXT-REPLAY.md, FLASHNEXT-REPLAY-11DOCS.md, LAYOUT-SMOKE.md, rad0042-selftest/).

## Change Log

- 2026-11-18T07:19:07.714328+00:00 (created-by): Created by agent
- 2026-11-18T07:19:28.605487+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
