---
id: RWQ02
order: 2
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Implement and validate WHT8 over the existing no-P2P libr3 transport

## Description

Prototype an orthonormal Hadamard 8-bit wire with BF16 block scales, preserving the proven two-rank handshake and fused epilogues. WHT6 uses 64 coefficients, 6 signed bits and one BF16 scale (50 bytes/64 values = 6.25 bits/value). WHT8 would use 64 signed 8-bit coefficients plus a 2-byte scale (66 bytes/64 = 8.25 bits/value), i.e. 51.5625% of exact BF16 size versus WHT6's 39.0625%. This is a proposed codec, not demonstrated model fidelity.

## Steps

1. Before GPU changes, run offline codec_mock.py and tests; freeze synthetic sample results. Add independent golden vectors with known coefficient/packed values; do not use the mock's logical packed memory as a GPU layout golden.
2. Copy the pinned Wht6Chunk geometry to a named Wht8 codec in the libr3 build-tree rewrite. For an initial A/B, allow R3_AR_WHT8=1 to substitute WHT8 for WHT6 only in the experimental two-rank host paths; require the same setting on both ranks and print activation once; default OFF.
3. Implement all three served fused forms, not only standalone all_reduce: ar_rmsnorm_quant_fp8, ar_hc_write, ar_gather_hc_write. Audit other consumers of WHT6 and avoid leaving a hidden WHT6 path in the supposed WHT8 arm. Keep decode/short-message exact for attribution.
4. Choose explicit scratch offsets, validate pointer arithmetic, alignment, slot parity, delayed peer flags and graph replay across alternated shapes and multiple ranks. Add a synthetic GPU checker fixture for gaussian, heavy-tail, sparse, mixed scales, zeros, non-64 tails and mixed 1/64/512/2048 rows. Reject/explicitly fall back for unsupported shapes.
5. Run device comparative checks against fp32 host oracle and pinned WHT6 kernel, and ensure both rank outputs match under equivalent input. Only after device checks run served Flash-Next/27B KLD and ABBA speed.
6. If WHT8 is useful, add a proper graph-visible wire=8 declaration/selection through a Radiance patch package, rather than permanently masquerading as wire=6. Include patch.toml, SUMMARY, code-line anchors, PIN_BLOBS mechanics tests and recipe experiment; remove temporary wire-masquerading from production.

## Detailed Solution & Technical Design

Pinned source: Quackster/radiance 89cee7c, libr4d/r4d_ar_oneshot_2rank_wht6.h and .hip, libr4d/r4d_ar_entry.hip, arch/common/rad_block_hc.h; BigCherry libr3/native/r4d_ar_entry.hip.rw. For BITS=8: signed qmin=-128, qmax=127, scale=BF16_RNE(amax/127), q=clamp(rintf(rotated/scale),-128,127); scale zero/invalid returns 1.0 as current codec semantics. Retain -ffp-contract=off for reduction commutativity where needed. For CHUNK_GROUPS=32: 2048 elements per chunk, 2048 code bytes per wave chunk, 16 dwords/lane = four uint4/lane versus WHT6's three uint4/lane. A local packet includes a BF16 scale for every 64 values. The current locpack allocation (half slot + slot/32 + guard) is plausibly sufficient for eight bits and f32 local scales, but must be asserted at kernel launch rather than assumed. The peer scale plane currently lives at slot_bytes/2: for 8 bits code bytes equal slot_bytes/2, so scales begin immediately after the payload. Check max slot, alignment and no overwrite of flags/adjacent slot on largest graph.
The host scratch and system-scope publish protocol remain unchanged; do not add CPU polling/stream sync to the critical path. Use GPU-side saturation checks and positive scale finite guards. Validate under native attention and helper-off controls first, then helper-on as a separate factor. Round on both ranks exactly the same way (packed local copy must be read back, not recomputed).

## Code Samples & Guidance

Device test cases must print mode, rank, dimensions, rel-L2/max-abs, saturation count, scale minimum/maximum, and guard status. Test 6/8/4 bit size invariants at 64: 50/66/34 bytes. For 512 elements: 400/528/272 bytes. The old WHT6 code is control, not replaced until evidence is accepted.

## Files

engines/radiance/components/libr3/native/r4d_ar_oneshot_2rank_wht8.hip.rw (or equivalent dedicated WHT8 source/rewrite); native/r4d_ar_entry.hip.rw; CMakeLists and component.toml; tests under engines/radiance/components/libr3/ or tests fixture; later engines/radiance/patches/rad0004_* (check current IDs before choosing); tools/tests/patch/test_radiance_patch_packages.py; engines/radiance/recipes.toml.

## Validation

Unit and GPU kernel checker: exact same-length arbitrary tensor layouts, transformed inverse tolerances, round-to-nearest-even BF16 ties, positive/negative near clamp, +0/-0, all-zero, alternating scale, protocol stress, alignment, out-of-bounds poison, host mapped memory flush/acquire, multi-inflight graph replays and rank symmetry. Acceptance on real model deferred to RWQ05. Disqualify on deadlock, any mismatched flag epoch, or incorrect fused post-op regardless of standalone success.

## Effort & Risk

Risk high: 8-bit can still produce unacceptable KL, fused norm may shift reduction order, and mapped host memory may expose weak memory ordering. WHT8 improves local resolution but cannot guarantee final-token fidelity. Not all 8-bit changes translate into speed: 31.25% more bytes than WHT6.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/radiance/RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- Opt-in, default OFF WHT8 executes on BOTH ranks and each served prefill fused collective.
- GPU kbench/poison/host-reference, rank consistency and long graph replay pass.
- No hidden extra synchronization, no fallback silently counted as WHT8.
- Whole-model quality and end-to-end throughput quantified before any promotion.

## Notes

Depends on RWQ01 measurements; no engine patch is mandatory for the first plugin-only isolated experiment if it replaces the existing wire=6 codec with explicit diagnostic labeling. A real independent wire=8 protocol and selective use need RWQ03.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.
