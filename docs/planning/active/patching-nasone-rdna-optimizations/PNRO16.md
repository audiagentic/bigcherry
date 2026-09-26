---
id: PNRO16
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-11T04:32:00.691672+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Offline expert-placement feasibility (resume against current main)

## Description

TODO, NOT-READY, RECLASSIFIED per GPT review as a tooling-implementation item, not a patches/ item. Explore whether BigCherry can pre-compute a reusable, model-fingerprinted MoE expert placement map before llama.cpp gains runtime expert-parallel execution. CORRECTED: this item was explicitly not a patches/ item and cannot satisfy the review's llama.cpp-Edit()-package handoff contract -- no such package should be fabricated for it. Even judged purely as tooling work, the exact vendored GGUFReader import/API and the concrete evidence path/target model remain marked NEEDS-VERIFICATION/TBD in this item's own detailed_solution -- these must be resolved before implementation, not left open.

## Steps

1. Treat this explicitly as a tooling-implementation item under tools/lab/moe-expert-placement/ (per this project's own convention: bench/tool scripts go in tools/lab/<topic>/), NOT a patches/ package -- no llama.cpp Edit() should ever be authored for this item's scope.
2. Before writing inventory.py, resolve the exact vendored GGUFReader import path/API by grepping other tools/lab/* scripts for `import gguf` to match this project's existing convention (currently unresolved/TBD).
3. Resolve the concrete evidence path and target model for the real-model validation run (currently TBD by convention -- check an existing tools/lab/*/README.md for the project's evidence-artifact placement convention and this project's actual current production benchmarking model) before authoring the CLI pipeline.
4. Re-derive the approach fresh against current main -- do not attempt to cherry-pick or rebase the deleted branch's commits.
5. Re-establish the original spike's scope (deliberately runtime-inert: no patching of build_moe_ffn, no GGUF mutation, no runtime expert movement, no new RCCL communicators): GGUF tensor/expert inventory, per-layer packed-expert byte accounting, PLE/Engram and named-MTP byte classification, deterministic static expert-home compilation from explicit per-device expert budgets, model-layout fingerprinting, global expert ID -> device/local-slot maps, and offline routing-trace replay.
6. Register every new tool file in docs/reference/tooling/TOOL_DISPOSITION.md as part of the same change.
7. Produce the real GGUF/routing validation evidence the original spike never got to, against a concretely identified model (resolved in step 3) -- this item should not be closed on code-exists-and-is-tested alone.

## Detailed Solution & Technical Design

This is NOT a patches/ change -- it is pure Python tooling under tools/lab/, using llama.cpp's gguf-py (vendored/available at the pinned b11126 source, e.g. work/upstream/llama.cpp.git tree's gguf-py) to read real GGUF files offline; it must never touch ggml/llama.cpp source or mutate a GGUF, and must never move experts at runtime. Layout: tools/lab/moe-expert-placement/{README.md,TESTING.md,inventory.py,byte_accountant.py,fingerprint.py,placement_compiler.py,routing_replay.py,cli.py,fixtures/}. Module breakdown: (1) inventory.py -- reads a GGUF via gguf-py's GGUFReader, enumerates per-layer expert tensors (gate/up/down per expert id), classifies PLE/Engram and named-MTP tensors separately from ordinary packed-expert tensors by name/metadata pattern, and reports byte size per tensor; (2) byte_accountant.py -- aggregates inventory rows into per-layer and per-expert byte totals, separating the three classes (packed-expert, PLE/Engram, MTP); (3) fingerprint.py -- hashes the model's layout-defining metadata (tensor names, shapes, dtypes, layer/expert counts -- NOT weight bytes) into a short fingerprint string; a placement map records the fingerprint of the model it was compiled for and fails validation (not silently reinterpreted) if the loaded model's fingerprint differs; (4) placement_compiler.py -- given an explicit per-device expert budget dict (device_id -> max expert count or max bytes) and the inventory, deterministically assigns each global expert ID to a (device_id, local_slot) pair (simplest correct policy: sorted greedy bin-packing by byte size, must be reproducible/deterministic for a given input -- no randomness), and emits a placement map JSON; (5) routing_replay.py -- consumes a routing trace (sequence of per-token top-k expert ID selections, one record per forward pass) and a placement map, and estimates: per-device auxiliary-store touch frequency, and minimum activation/result transport bytes implied by that map versus an alternative/baseline map, without executing the model. CLI (cli.py) commands: `inventory <gguf-path> [--out inventory.json]`, `compile-placement <gguf-path> --budgets budgets.json [--out placement.json]`, `validate-placement <gguf-path> placement.json` (checks fingerprint match), `replay-routing placement.json routing-trace.jsonl [--out report.json]`. Data formats: placement.json = {"fingerprint": str, "model_name": str, "devices": {device_id: {"experts": [{"global_id": int, "local_slot": int, "layer": int}], "budget_bytes": int, "used_bytes": int}}}; routing-trace.jsonl = one JSON object per line {"token_idx": int, "layer": int, "selected_experts": [int,...]}. Validation must include real evidence: run `inventory` + `compile-placement` + `replay-routing` against one of this project's actual production GGUF models (not a synthetic fixture) and check the emitted report's numbers are sane (nonzero bytes, expert IDs within range, fingerprint stable across two runs on the same file) -- this satisfies the item's 'real GGUF inventory + routing-trace-replay evidence' requirement as a concrete report artifact checked into docs/evidence/ or referenced by the ledger event, plus synthetic-fixture unit tests for the deterministic compiler and replay-estimator logic.

## Code Samples & Guidance

No patches/ anchors -- this is greenfield tooling, not an edit to existing bigcherry or llama.cpp files. Skeleton for cli.py's inventory command (illustrative, not copy-paste-exact -- the next agent verifies gguf-py's actual GGUFReader API against the vendored copy under work/upstream/llama.cpp.git's gguf-py at b11126 before writing this): `from gguf import GGUFReader` (or the project's vendored equivalent -- NEEDS-VERIFICATION of the exact import path this project uses elsewhere, e.g. grep other tools/lab/* scripts for `import gguf` to match convention) `reader = GGUFReader(path); for t in reader.tensors: classify_by_name(t.name)`.

## Files

tools/lab/moe-expert-placement/{README.md,TESTING.md,inventory.py,byte_accountant.py,fingerprint.py,placement_compiler.py,routing_replay.py,cli.py,fixtures/}; docs/reference/tooling/TOOL_DISPOSITION.md (register every new file, same change); docs/evidence/ artifact for the real-model validation run (path TBD by convention -- check an existing tools/lab/*/README.md for the project's evidence-artifact placement convention before picking one).

## Validation

Unit tests (synthetic fixtures): deterministic placement_compiler output for a fixed budget+inventory input (byte-for-byte reproducible across runs); fingerprint stability across repeated reads of the same file and change-detection across a deliberately altered metadata field; routing_replay estimator correctness against a hand-computed small trace. Real-model evidence (required, not optional per the item's own acceptance criteria): run the full CLI pipeline (inventory -> compile-placement -> validate-placement -> replay-routing) against an actual current production GGUF model this project already uses for benchmarking (check tools/lab/*/README.md or project memory for which model that is), with the resulting inventory/placement/replay report checked in or ledgered as evidence; explicitly re-derive per-device expert budgets from that real model's inventory at pickup time -- do not reuse the deleted spike's old topology.brutus.example.json numbers (already called out in this item's own Notes). Register every new file in docs/reference/tooling/TOOL_DISPOSITION.md in the same change -- this was the original spike's concrete governance blocker.

## Effort & Risk

Work=M. No hardware/build risk (pure Python, offline, read-only against GGUF files) -- the real risk is scope creep back toward the deleted spike's open-question state; keep this strictly to inventory+compile+replay-estimate with no runtime hook, no GGUF mutation, matching the item's own explicitly re-stated inert scope.

## Standards



## Acceptance Criteria

A working offline expert-placement compiler exists, registered in TOOL_DISPOSITION.md, with real GGUF inventory + routing-trace-replay evidence produced against an actual current model (not just unit tests against synthetic fixtures), built fresh against whatever main is current when this item is picked up.

## Notes

Original spike's own topology.brutus.example.json expert budgets were explicitly NOT a validated performance recommendation -- re-derive real budgets from the actual target model's inventory at pickup time, don't reuse the old example file's numbers. (2026-09-14: file moved to the prefixed patching-nasone-rdna-optimizations/ group, so test_nro_patch_packages' completeness check covers it.)

2026-09-24 relevance at b11126: TODO (tooling, not a llama.cpp/patches/ change). This item is pure BigCherry tooling under tools/lab/ per project convention (bench/tool scripts go in tools/lab/<topic>/, never /tmp). GPT design consultation was not used for this item -- the design followed directly from the item's own already-detailed steps plus this project's established tools/lab/ and TOOL_DISPOSITION.md conventions (verified docs/reference/tooling/TOOL_DISPOSITION.md exists and states the registry requirement); no llama.cpp source design questions were open that required GPT. Next agent must verify gguf-py's actual import/API surface against the vendored b11126 copy before writing inventory.py, and must re-derive real per-device expert budgets from an actual current model rather than reusing the deleted spike's example numbers (already flagged in this item's pre-existing Notes).

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: confirmed this item cannot satisfy the review's patch-package handoff contract since it is explicitly tooling, not a patches/ change -- no Edit() package should be fabricated for it. Required the previously NEEDS-VERIFICATION/TBD items (exact vendored GGUFReader import/API, concrete evidence path/target model) to be resolved as explicit prerequisite steps before implementation, rather than left open at authoring time.

## Change Log

- 2026-09-11T04:32:00.691672+00:00 (created-by): Created by agent
- 2026-09-11T04:32:12.763102+00:00 (updated-by): Updated: priority='P2', section:description, section:steps, section:acceptance_criteria, section:notes
- 2026-09-14T10:47:56.388459+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20260914_105034_fixed-the-three-long-standing_1994
- 2026-09-14T10:50:37.293465+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:44.348953+00:00 (updated-by): Updated: section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk
- 2026-09-24T02:32:04.448145+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:51:51.108221+00:00 (updated-by): Updated: section:description, section:steps, section:notes
