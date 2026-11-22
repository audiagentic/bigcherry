---
id: LTX05
order: 5
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:06:51.410896+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
priority: P0
work: L
---

# Qualify three-card LTX video/audio numerical parity and end-to-end performance

## Description

Provide reproducible model-backed video/audio quality, all-stage latency and memory evidence for TP1 versus 3-card TP3 on the exact three-card configuration. Decide release promotion from measured behavior, not LLM throughput.

**Existing engineering gap coverage:** LT3-14 P0 (three-card part). **Dependencies:** LTX04 opt-in serving acceptance.

## Steps

- 1. Freeze pinned Radiance 2.2 code, ABI44, patch composition, model revision/checksums, native YAML bytes, GPU BDF order, seed, prompts, negative prompts, size/frame count and guidance mode.
- 2. Run TP1 control and TP3 candidate alternating ABBA, with identical cooling/warm-start, power profile and no concurrent GPU jobs. Capture DiT, per-layer RMS collective, FFN, conditioning, VAE, codec, transfer wait, peak VRAM, full clip latency and failures.
- 3. Compare intermediate latent hashes and numeric metrics at pre/post attention, every DiT block and decoded output; require explicit owner-approved video/audio quality thresholds (no invented tolerance).
- 4. Profile baseline C05 full-row Q/K network payload; only if transfer/packing dominates, replace **LTX's model-specific Q/K pack/norm graph** using existing native ops and re-run TP1 parity. A missing generic collective/kernel or transport capability is RUM-owned, never implemented in this LTX plan.
- 5. Exercise cancellation, restart, timing variations, missing card and TP1 rollback, then write a qualified/not-qualified verdict with raw run IDs and exact build identity.

## Detailed Solution & Technical Design

Use BigCherry's existing measurement and KLD/quality pipelines and native resource admission. No new benchmark runner or Python engine dependency; Python analysis is fine. Treat TP1 and TP3 outputs as distinct numerical paths with explicit tolerance.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/configs/ (opt-in model config only)`
- `docs/evidence/ltx25-*/`
- `existing BigCherry measurement and qualification contracts`
- `docs/design/ltx25-tp3-component-test-ladder.md`

## Validation

Paired leased-hardware ABBA TP1/TP3 runs at representative/default 121-frame clip, seed/guidance variations, annotated breakdowns and output comparisons. Include result paths, raw receipts, device inventory and rejection cases.

## Effort & Risk

**P0, L.** High: oversized host-staged collectives on PCIe and costly Q/K all-head normalization may erase sharding benefit.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

No quality regression beyond owner-approved thresholds, no unbounded VRAM or deadlocks, full clip generated and decoded; report whole-video speed (target >=20% faster for proposed promotion, not a promised result). If candidate is slower or unqualified, keep TP1 as default and withhold TP3 release.

## Notes

Do not use C08 33/33 collective numeric receipts as a substitute for an end-to-end LTX timing or video quality result. Any replacement for C05's full-width pack/reduce/RMS path requires a measured bottleneck, TP1 parity and **deletion of the replaced graph code** within the existing Radiance runtime. No separate collective/benchmark service. Generic native GPU profiling/collective changes are routed to RUM13 or RUM06; LTX05 owns only model-specific optimization and video quality.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is an authoritative pending plan item, registered through Planning MCP; historic evidence remains at the linked path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX05.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

## Change Log

- 2026-11-22T03:06:51.410896+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:14:23.857584+00:00 (updated-by): Updated: section:notes
