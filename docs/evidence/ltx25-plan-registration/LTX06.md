---
proposed-id: LTX06
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P1
work: L
skill: advanced
owner: Planning MCP registration required
---

# Qualify optional gfx1030 Gemma text worker and actual memory residency

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Check if the RX 6900 XT can independently run the model's Gemma text stack without stealing a TP rank or causing impossible memory admission; implement only missing native operator rows with verified coverage.

**Existing engineering gap coverage:** LT3-10 and LT3-11. **Dependencies:** Independent accepted TP3 three-card serving LTX04; no dependency for LTX01–05.

## Steps

- 1. Inventory exact Gemma operators on gfx1030 `libr2`; identify missing dense GEMM, causal/sliding attention, norms and embeddings before proposing any kernel source.
- 2. Use the released Radiance 2.2 native kernel row/registration framework; add narrowly justified missing gfx1030 rows only with CPU reference and device tests. Do not copy a new Gemma implementation or call into Python.
- 3. Measure real model Gemma weight bytes, runtime temporary/persistent allocations, worst two-prompt 49-layer BF16 stack, fragmentation and reserve on 16GiB gfx1030; reject if actual workspace cannot fit.
- 4. Keep connectors on a fast GPU initially. Execute one- and two-prompt Gemma and compare exact embeddings/hidden stack to TP1 native control, including warm/repeat and cancellation.
- 5. Record qualified/unqualified decision and if unqualified retain three-card TP3 independently with original text path.

## Detailed Solution & Technical Design

Optional gfx1030 is a separate typed worker/role, never a fourth DiT rank, never a MoE helper, and never allowed to overlap an active measured GPU session. Reuse registry, memory facts, native plugin SDK and admission.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/components/libr2/units.txt`
- `engines/radiance/archs.toml`
- `existing Radiance Gemma native kernel packages`
- `docs/design/ltx25-iq8r-tp3.md`

## Validation

Kernel coverage contract, actual gfx1030 native kbench/selftests against TP1 reference, 1/2 prompt Gemma full model and recorded peak VRAM; strict no-fit refusal with reserve.

## Effort & Risk

**P1, L.** High: missing gfx1030 dense kernels and limited VRAM. Do not start this until three-card LTX is useful.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

Complete Gemma text output on 6900 matches control within approved metrics and actually fits measured budget under repeated use. If not, document no-go and retain three-card functionality; never silently offload to system RAM as a claimed same-config result.

## Notes

Standalone source estimate (~13 GiB weights) does not prove fit; the 735MiB two-prompt hidden stack plus intermediate scratch is a separate requirement.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
