---
id: LTX02
order: 2
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:06:26.779657+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
work: L
priority: P0
---

# Execute exact three-rank Q/K RMS and self-attention with checkpoint activations

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

### Smallest issued-C05 slice (open)

The existing 5-case native Builder test **does not issue on GPU**:
it calls `Builder::set_instances(false)` and checks `libref` host
rows, but does not create a native `rad::Ctx` or stage device
weights. RUM13's separate exact AQL collective proof also
does **not** execute LTX's generated graph. Do not equate
these two tests with actual model parity.

Reuse the **same model-backed Builder/Program and existing
Radiance weight loader**, select installed gfx1100/gfx1201
kernel registry rows, plan native arena/buffers, and bind
three `CtxDesc` instances with rank/world/device and
`record_passes=false`. Issue just the existing
`Tp3GlobalRms::step` Q then K sequence with read-only
checkpoint gains, compare rank-local BF16 outputs against
a full-width TP1 oracle, and record numerical tolerance
and source identity. First prove refusal of missing
HIP-enabled core, absent kernel or gain, and duplicate
physical devices. A generic runtime gap belongs to its
RUM owner, not an LTX workaround. After this gate,
extend the same path to the complete C06 block.

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

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is an authoritative pending plan item, registered through Planning MCP; historic evidence remains at the linked path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX02.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

2026-11-22: generic native resident-weight materialization is now assigned to RUM14 (patching-radiance-upstream). LTX02 must consume ONE Engine-owned source-loading pipeline instead of copying plan_container_weight, RelayoutPool or sweep_container. C06 declaration/24 source-backed native relayout hook calls and 108 initialized AQL instances per GPU order are not full checkpoint-resident issue or TP1 parity. After RUM14 acceptance, LTX02 alone owns actual video/audio C06 Act/Lin/attn_core issue and TP1 numerical comparison, reversed-device/fault tests and unchanged production TP3 refusal.

2026-11-22 new execution evidence from merged PR #1315 (c72ed6be3): the actual C06 RadCtx self-attention graph issues real checkpoint IQ8R weights for video/audio block 0, both legal 8/8/16 and 12/12/8 splits, host 4/4 and AQL 4/4 in BOTH gfx1100/gfx1100/gfx1201 orders. Three exact native collective calls/rank and 108 native instances/order; reversed AQL ordinals give 12,288/12,288 identical BF16 outputs. Same-split GPU-vs-host relative L2 is 0.330–0.422%, and changing head partition on host changes 1,605 video / 736 audio BF16 elements; neither comparison is an approved TP1 parity tolerance. Inputs are one-row deterministic seeded BF16, NOT captured TP1 activations. 16/64-row issue, reference activation parity, bounded fault/cancel, phase performance and production TP3 refusal still await validation. The test-only source-plane gather and memcpy/GPU pointer staging MUST be replaced/deleted under generic RUM14 before shared loader acceptance, without discarding this issued C06 coverage. PR #1316 fixes GPU coordinate allocation unwind; source and numeric evidence is docs/evidence/2026-11-22-ltx25-ltx02-c06-issued-attention/README.md.

2026-11-22 (RUM15 numerical source review): GPU XTX and R9700 are bitwise identical at 86,400 captured BF16 C06 intermediate values; CPU/GPU IQ8R QKV difference is fully accounted for in a 198-element checkpoint-backed sample by an extra BF16 rounding of U8*BF16 weight-scale product performed by both libhalo and libr4d layout hooks. CPU libref multiplies the same two levels in F32. Do NOT make libref round its product to BF16 merely to make comparisons match. Generic kernel/layout precision-preserving remediation belongs exclusively to RUM15, with a measured representation/performance and admission decision; LTX02 retains real TP1-activation parity/quality, 16/64 rows, faults, and production refusal until LTX04. See docs/evidence/2026-11-22-ltx25-ltx02-iq8r-scale-root-cause/README.md.

2026-11-22 new multirow C06 issue slice: the existing model-backed RadCtx/Attn graph now executes 16 and 64 rows for video and audio self-attention with real checkpoint weights, both 8/8/16 and 12/12/8 splits, host 13/13 CTests and opt-in mixed AQL in both gfx1100/gfx1100/gfx1201 device orders (all pass). Identical-order reruns and all three rank output replicas are bitwise stable. Cross-device-order 16-row BF16 outputs are identical; 64-row BF16 differences are repeatable: audio 351–360 / 131072 and video 796 / 262144; relative L2 0.01466%–0.01946%, first appearing after attention. No TP1 captured activation/reference, no approved quality tolerance or full production graph. Evidence: docs/evidence/2026-11-22-ltx02-multirow-issued/README.md. Continue pinned TP1 comparison and generic attention precision triage, fault/cancel and RUM14 resident loader integration. Do not remove TP3 serving refusal.

2026-11-26 next execution roadmap (docs/planning/NEXT_EXECUTION_ROADMAP.md; main 725bd8951). Assumes native YAML serving cutover and initial min-cap qualification accepted; does not claim they are done. Stage 6 separate capability programme after RUM14/RUM15; current checkpoint-backed 1/16/64 mixed-order self-attention issued, but TP1 oracle and fault/cancel, native resident loader and scale decision still open. Do not repeat declarations or reopen generic AQL. LTX03–05 then LTX08 release only after actual complete video/audio and quality, LTX06/07 optional. TP3 remains refused until LTX04.

## Change Log

- 2026-11-22T03:06:26.779657+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:14:01.870524+00:00 (updated-by): Updated: section:notes
- 2026-11-22T08:21:48.151528+00:00 (updated-by): Updated: section:notes
- 2026-11-22T08:36:27.861283+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4467
- 2026-11-22T09:18:12.361715+00:00 (updated-by): Updated: section:notes
- 2026-11-22T12:50:39.750825+00:00 (updated-by): Updated: section:notes
- 2026-11-25T18:52:55.652660+00:00 (updated-by): Updated: section:notes
