---
id: LTX06
order: 6
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:06:56.862017+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
priority: P1
work: L
---

# Qualify optional gfx1030 Gemma text worker and actual memory residency

## Description

Check if the RX 6900 XT can independently run the model's Gemma text stack without stealing a TP rank or causing impossible memory admission; inventory any missing generic native kernel rows but **do not implement libr2 or generic kernel registration here**; hand the exact verified gap to a RUM/native-kernel owner.

**Existing engineering gap coverage:** LT3-10 and LT3-11. **Dependencies:** Independent accepted TP3 three-card serving LTX04; no dependency for LTX01–05.

## Steps

- 1. Enumerate the **LTX checkpoint's Gemma operator signatures** (weight dtypes and extents, dense GEMM, causal/sliding attention, norms and embeddings) as source-backed requirements, without changing shared kernel code.
- 2. Inventory actual gfx1030 Gemma operator/row eligibility and record missing-row evidence. Route any required **shared libr2/kernel-row implementation** to a RUM-owned native kernel plan after checking existing owners; LTX06 stays responsible for the Gemma-specific weight, shape, quality and feasibility test. Do not copy a new Gemma implementation or call into Python.
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

- `engines/radiance/components/libr2/units.txt` (read-only kernel inventory; shared source changes belong to RUM)
- `engines/radiance/archs.toml` (read-only plugin eligibility; shared source changes belong to RUM)
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

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is an authoritative pending plan item, registered through Planning MCP; historic evidence remains at the linked path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX06.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

## Change Log

- 2026-11-22T03:06:56.862017+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:14:33.977963+00:00 (updated-by): Updated: section:notes
