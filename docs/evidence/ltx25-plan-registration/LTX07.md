---
proposed-id: LTX07
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P1
work: L
skill: advanced
owner: Planning MCP registration required
---

# Add optional native text-stack handoff and measured four-card admission

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Add one bounded native Gemma producer on gfx1030 and fast-card connector consumer through the existing request/owner/admission framework, with no fourth DiT rank or duplicated YAML path.

**Existing engineering gap coverage:** LT3-12 and LT3-13. **Dependencies:** LTX04 and LTX06 accepted; LTX05 baseline as independent comparator.

## Steps

- 1. Define immutable typed producer record: model/build digest, request/epoch, 49-layer stack shape, BF16 length, GPU BDF, source/finalizer and completion fence. Reuse existing native request/config schema and one admission owner.
- 2. Compute Gemma on 6900; host-stage full two-prompt stack (up to 770,703,360 bytes per model geometry) once per conditioning request; fast card runs both connector towers and duration head. Do not transfer per denoise step.
- 3. Use existing shared transfer streams, pinned buffers and native manifest admission, not a new dynamic sockets/messages system. Validate arrival order, checksum, stale or cancelled epoch, worker death and resource release.
- 4. Add exact native YAML worker role, BDF disjointness and memory requirements through the existing edition-3 measured input/fresh replay and owner receipt. Forbid unknown memory, duplicate BDFs and stale model pins.
- 5. On a leased four-card run, prove exact positive/negative prompt output, peak VRAM, measured copy/stall, start/cancel/restart and a comparable TP3 no-text-worker control. Avoid fallback under a measured four-card contract.

## Detailed Solution & Technical Design

One producer/consumer handoff plugged into the existing LTX VCOND boundary and native session admission. No alternate YAML parser, scheduler, epoch service or host-owned arbitrary worker registry. Final connector contexts are ~24MiB for two prompts only after connector placement; default handoff is the full ~735MiB stack.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/ (only needed LTX handoff hooks)`
- `engines/radiance/components/serving/ (existing native config authority)`
- `engines/radiance/configs/ltx2-*`
- `tools/lab/radiance/ltx-tp3/ltx_tp3_handoff_test.cpp`

## Validation

Strict native YAML parser/replay/identity fixture, handoff mock negative cases and real gfx1030→gfx1100 transfer with memory bounds, epoch lifetime and cancellation; full config receipt.

## Effort & Risk

**P1, L.** High: 735MiB host transfer and 6900 worker memory/lifetime; gated optional feature, no release unless measurable benefit.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

Four-card config is exact and guarded, no shared physical GPU ownership, transfer is once per VCOND with correct data/reference, no leaks or stale requests; three-card LTX unaffected. Publish speed difference separately, not as an assumed improvement.

## Notes

undefined

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
