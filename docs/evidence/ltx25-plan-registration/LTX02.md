---
proposed-id: LTX02
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P0
work: L
skill: advanced
owner: Planning MCP registration required
---

# Execute exact three-rank Q/K RMS and self-attention with checkpoint activations

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Turn C05/C06's host-declared graph into one actually issued three-card model-weight self-attention operation, including complete Q/K global normalization, RoPE, attention, output reduction and BF16 bias exactly once.

**Existing engineering gap coverage:** LT3-05 and self-attention part of LT3-06. **Dependencies:** LTX01's model-source declarations plus RUM13's reusable AQL collective proof; RUM06 owns native group lifetime. No shared runtime/harness implementation in LTX02. No new collective implementation or additional benchmark runner.

## Steps

- 1. Write a failing **actually issued** single-block RadCtx test using existing patched `ltx2::attn_core`, `Attn::declare`, `Act::step` and `Lin::step`, and consume the RUM13-owned native AQL evidence, not direct-HIP or a second generic launcher. Specify row count, actual model weights, rank order, owner group, per-rank device/kernel, source fingerprint, and TP1 reference.
- 2. Use existing `RadBuilder`, `RadCtx`, device AQL streams and exact native `all_reduce`. Issue the C05 fill/cast/reduce/rmsnorm/cast chain on all ranks with same-token rows; require full-video-4096 and audio-2048 gain and denominator before local `rope_split`.
- 3. Issue fused QKV/gate, local RoPE and `attn_dense` with native head offsets, aligned IQ8R quantizer, exact C04 output reduce and only one post-reduction BF16 bias per replica; compare all ranks with the same TP1 activation output. Preserve ordered collective issue across ranks.
- 4. Exercise 1/16/64 row sizes and 8/8/16 or 12/12/8 splits; replay reversed visible card ordinal, missing kernel or abandoned rank; fail closed with native timeout and owner-tagged errors.
- 5. Measure native fill, insert/cast, exact all-reduce, RMSNorm, extraction, GEMM, local attention, output reduction and AQL transfer waits separately at model-derived rows against LTX01's bare-collective receipt and TP1. Only consider compact per-row-statistic reduction after **measured** hot-path cost and proof of equal numerical semantics; replace C05 in place, never add a parallel communication path.

## Detailed Solution & Technical Design

Consume RUM13's AQL GPU evidence and RUM06's shared-slot/group lifetime. LTX02 owns only model-specific `RadCtx` graph issue, checkpoint oracle and tolerance. Missing generic engine or collective capabilities are RUM blockers; do not implement them here. Do not launch via direct HIP (C07 crash bypassed AQL registration), create a new collective or use per-rank RMS. C05 full-width Q/K reconstruction is the simple correctness baseline; locally compact statistical sums are optimization-only.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/src/{global_rms,attn_core,attn_norm,attn_output,attn_rope}.inc`
- `tools/lab/radiance/ltx-tp3/native-c01-gate/`
- `docs/evidence/rum13-native-qualification/README.md` (shared AQL proof, read-only dependency)
- `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`

## Validation

New source-backed executed native CTest with checkpoint payload and real AQL context, TP1-vs-TP3 BF16/quantized intermediate reference, stale ownership/cancel/timeout negative test and 3-card lease receipts. Show numerical metric/tolerance before acceptance, do not invent quality thresholds.

## Effort & Risk

**P0, L.** High: native graph issue, device code-object admission and all-head RMS precision; do not claim success based on C05/C06 declaration or C08 bare collective alone.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

At least one complete video and one audio self-attention layer genuinely executes on gfx1100/gfx1100/gfx1201 with native AQL exact collectives, matches TP1 within an explicitly approved tolerance, and stops safely on timeout. The old LTX TP1 code path stays unchanged, full world>1 serving refusal remains until LTX04.

## Notes

The latest C08 has 33/33 exact BF16 checks per collective variant; it does not issue the C05 model graph. LTX02 must reuse that working native Radiance backend and produce checkpoint-backed TP1 parity, not reinterpret C07's direct-HIP failure as a new transport feature. This is the immediate production-enabling next slice.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
