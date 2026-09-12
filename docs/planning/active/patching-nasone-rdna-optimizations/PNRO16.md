---
id: PNRO16
order: 0
plan: nasone-rdna-optimizations
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

Explore whether BigCherry can pre-compute a reusable, model-fingerprinted MoE expert placement map before llama.cpp gains runtime expert-parallel execution, and whether routing traces can be replayed offline to estimate device participation and transport cost. This was previously prototyped on a now-deleted branch (`expert-placement-offline`, commits f21d5022..5ccf5ba9, 6 commits, ~740 lines) but was discarded per a GPT deep-review recommendation (2026-09-11): it was 494 commits behind main, its own README marked it an untracked exploratory spike with open question state, its required GGUF/routing validation evidence was never produced, and its files were never registered in the mandatory TOOL_DISPOSITION.md registry. The underlying idea is still worth pursuing -- this item exists so it is resumed as tracked work against whatever `main` is current at pickup time, not by resurrecting the deleted branch.

## Steps

1. Re-derive the approach fresh against current main -- do not attempt to cherry-pick or rebase the deleted branch's commits (they predate 494+ commits of unrelated history and the branch is gone).
2. Re-establish the original spike's scope, which was deliberately runtime-inert (no patching of build_moe_ffn, no GGUF mutation, no runtime expert movement, no new RCCL communicators): exact GGUF tensor/expert inventory via llama.cpp's gguf-py, per-layer packed-expert byte accounting, PLE/Engram and named-MTP byte classification, deterministic static expert-home compilation from explicit per-device expert budgets, model-layout fingerprinting (so a stale map fails validation), global expert ID -> device/local-slot maps for a future llama.cpp executor, and offline routing-trace replay to estimate auxiliary-store touch frequency and minimum activation/result traffic implied by a candidate map.
3. Register every new tool file in docs/reference/tooling/TOOL_DISPOSITION.md as part of the same change (this was the concrete governance gap that blocked the original spike from merging).
4. Produce the real GGUF/routing validation evidence the original spike never got to -- this plan item should not be closed on code-exists-and-is-tested alone, per this project's own patch/tool lifecycle doctrine (current qualification requires real evidence, not just a clean implementation).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

A working offline expert-placement compiler exists, registered in TOOL_DISPOSITION.md, with real GGUF inventory + routing-trace-replay evidence produced against an actual current model (not just unit tests against synthetic fixtures), built fresh against whatever main is current when this item is picked up.

## Notes

Original spike's own topology.brutus.example.json expert budgets were explicitly NOT a validated performance recommendation -- re-derive real budgets from the actual target model's inventory at pickup time, don't reuse the old example file's numbers.

## Change Log

- 2026-09-11T04:32:00.691672+00:00 (created-by): Created by agent
- 2026-09-11T04:32:12.763102+00:00 (updated-by): Updated: priority='P2', section:description, section:steps, section:acceptance_criteria, section:notes
