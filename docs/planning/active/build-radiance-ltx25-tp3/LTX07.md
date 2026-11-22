---
id: LTX07
order: 7
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:07:05.959168+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
priority: P1
work: L
---

# Add optional native text-stack handoff and measured four-card admission

## Description

Implement only the **Gemma/LTX tensor producer/consumer adapter** on gfx1030/fast-card connectors using existing RUM05/RUM06 request, transfer and admission APIs. Do not implement generic worker roles, owner/epoch state, native YAML parsing or a fourth DiT rank.

**Existing engineering gap coverage:** LT3-12 and LT3-13. **Dependencies:** LTX04 and LTX06 accepted; LTX05 baseline as independent comparator.

## Steps

- 1. Define only the **model tensor payload view** (49-layer BF16 stack, bytes, checkpoint digest). Reuse existing RUM05 native ownership/BDF and RUM06 generation/request/fence identity; no duplicate generic producer record or lifecycle.
- 2. Compute Gemma on 6900; host-stage full two-prompt stack (up to 770,703,360 bytes per model geometry) once per conditioning request; fast card runs both connector towers and duration head. Do not transfer per denoise step.
- 3. Consume RUM05/RUM06-owned transfer streams, pinned buffers and native manifest admission; any missing **generic** transfer, epoch or drain function is a RUM blocker, not new LTX code. Validate arrival order, checksum, stale or cancelled epoch, worker death and resource release.
- 4. Supply only **LTX-specific requirements** for an optional named native-v3 worker config. **RUM05 exclusively owns implementation of any missing generic worker-role YAML, BDF disjointness and admission rules.** Forbid unknown memory, duplicate BDFs and stale model pins.
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
- `docs/planning/completed/patching-radiance-upstream/RUM05.md` (shared native config/admission owner, read-only dependency)
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

This four-card optional producer must not delay LTX01–LTX05, create a fourth DiT rank or duplicate the Radiance 2.2 AQL, native config or shared-memory transport. Preserve independent three-card operation if the optional worker is not admitted.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is an authoritative pending plan item, registered through Planning MCP; historic evidence remains at the linked path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX07.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

## Change Log

- 2026-11-22T03:07:05.959168+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:14:42.134176+00:00 (updated-by): Updated: section:notes
