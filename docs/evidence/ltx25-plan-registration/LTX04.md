---
proposed-id: LTX04
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P0
work: L
skill: advanced
owner: Planning MCP registration required
---

# Integrate safe opt-in TP3 native serving, conditioning and guidance

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Bind actual native three-rank LTX Engine sessions, STG/isolated guidance and leader-owned VCOND/VAE/video/audio output without breaking TP1 or adding an alternate config/scheduler.

**Existing engineering gap coverage:** LT3-02, LT3-08, LT3-09. **Dependencies:** LTX03 full 48-block emitted path and approved TP1 parity.

## Steps

- 1. Model precise ownership: all ranks execute DiT, one declared leader performs VAE, codec, output, and any non-TP operations. Capture one native owner+TP-group and exact device/rank/kernel mapping via existing Radiance services.
- 2. Keep the `ltx2` world>1 refusal until a named opt-in native v3 YAML has all verified modules, resident memory, step order and admission. No unknown device topology or unsupported helper must enter partial initialization.
- 3. Wire VCOND inputs, video/audio latents and text contexts to rank consumers once per request; preserve request/epoch/shape identity and completion fences, not a new mailbox ABI.
- 4. Exercise conditional, STG, isolated-guidance passes and block-28 bypass with exactly matched collective schedules; ensure negative prompts, zero/missing tokens, cancelled parents and rank failures drain safely.
- 5. End-to-end first 1-step and short clip with actual VAE/audio decode; fail closed on wrong owner, duplicate rank, hidden kernel/device or mismatch, and ensure TP1 fallback is explicit and unchanged.

## Detailed Solution & Technical Design

Use native v3 session config/measurement, existing Engine rank semantics and a single LTX execution path. Any pass-specific decision is scoped to an immutable native request or rank state, not a mutable process-wide stage. Do not borrow remote-MoE helper state or introduce Python serving.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/patch.py`
- `engines/radiance/components/serving/ (reuse exact native config/readiness where required)`
- `engines/radiance/configs/ltx2-* (new opt-in only after accepted startup)`
- `tools/lab/radiance/ltx-tp3/{ltx_tp3_guidance_test.cpp,ltx_tp3_protocol_test.cpp}`

## Validation

Focused native request/session/guidance mocks; actual mixed-device startup, one-step, cancel, restart and stop on leased GPUs; exact source/build/config receipt; representative output/codec evidence.

## Effort & Risk

**P0, L.** High: mixing single-rank VCOND/VAE with three-rank denoising, group teardown ordering and missing native operators.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

Production TP1 unchanged; named opt-in TP3 only, correct leader output, each guidance mode with identical collective sequence and no deadlocks; valid complete short clip and clean cancel/restart with measured resource admission. No release promotion until LTX05.

## Notes

Do not enable multi-rank LTX as an intermediate diagnostic toggle while graph/serving safety gates are open.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
