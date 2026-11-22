---
id: LTX03
order: 3
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:06:36.091610+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
priority: P0
work: L
---

# Complete the 48-block dual video/audio TP3 DiT graph without new dispatch infrastructure

## Description

Extend the single proven LTX native TP3 architecture path to all self, text and cross-modal attention, FFN and modulation operations across 48 DiT blocks, preserving model-specific rank/head ownership while consuming the RUM06-owned collective scheduler and lifetime.

**Existing engineering gap coverage:** Remaining LT3-06 plus cross-modal and FFN work from LT3-03/04/05. **Dependencies:** LTX02 accepted on both modalities.

## Steps

- 1. Enumerate actual six attention kinds at every DiT block: video self, audio self, video-to-text, audio-to-text, video-to-audio and audio-to-video. Check fused KV Q sources, head dimension, real weight/gain/bias geometry for each; use the same C01Slice/Geom authority.
- 2. Reuse `ltx2::Attn::declare`, `attn_core` and `Lin::declare/step` with one explicit owner view, not copied per-attention implementations; add only missing typed differences for cross-modal Q vs KV and data-length mismatch.
- 3. Make output projections, FFN-down and other row-parallel operations use existing exact reduction and once-only bias with replicated residual/modulation; replicate small BF16 gates when cheaper than a new shard loader.
- 4. Verify the **LTX-emitted operation order** for each conditional/pass variant, including STG block 28, zero tokens and uneven audio/video rows; consume RUM06's existing cancellation/fault/drain contract and report any missing generic behavior to RUM06 instead of implementing a new one. Fail before issue on unsupported K group boundaries.
- 5. Build source-backed tests using the real model for every block/kind, then issue representative one-block multi-modal and entire 48-block traces under a lease. Require TP1 intermediate equality / quality constraints prior to adding serving entrypoint.

## Detailed Solution & Technical Design

No new model scheduler, TP-group/collective scheduler, patch loader or DSO API. RUM06 exclusively owns shared runtime sequencing mechanics. The normal LTX DiT loop remains the only step owner; take rank-local operands and explicit global head offsets once. Use native `all_reduce` across the existing TP group, not device P2P assumptions.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/patch.py and src/attn_*.inc`
- `arch/ltx2/ltx2.cpp (patched target only; never edit vendor mirror directly)`
- `tools/lab/radiance/ltx-tp3/native-c01-gate/`
- `tools/lab/radiance/ltx-tp3/ltx_tp3_numerical_test.cpp`

## Validation

Every block x attention kind x legal split source-backed declaration, + actual native issue runs for self/text/cross-modal, FFN, residual and 48-block mixed schedule; TP1 vs TP3 intermediate oracle; owner and deadlock fault injection.

## Effort & Risk

**P0, L.** High: cross-modal Q/K asymmetry, oversized collective transfers, rank-conditional graph drift. Keep PR slices bounded and evidence-specific.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

All six attention variants, FFN and residual updates use a single existing DiT loop and exact TP3 collectives, zero duplicate/missing heads, no per-rank bias duplication, correct all-head RMS before RoPE; full 48-block issued trace compares within approved numeric gates and stops cleanly.

## Notes

C06 currently tests only video/audio self-attention declaration; no cross-modal production path is activated. Reuse the **same** native DiT/attention call sites; once LTX02 has executable parity, delete redundant provisional geometry/state in the same reviewed slice, with lines/concepts deleted and runtime cost measured. Do not create an alternate LTX scheduler.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is now an authoritative pending plan item, registered through Planning MCP. Historical evidence remains at its original path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX03.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

2026-11-22 LTX03 first executable full-depth declaration slice (PR #1396, commit 5902eaa6c): extended the existing C06 model-backed native Attn::declare/Builder test across 48 DiT blocks, video/audio self-attention, both legal head maps (8/8/16 and 12/12/8), and three ranks; 576 real-model declarations passed. Single-worker low-priority host build and all 11/11 CTests pass. Evidence: docs/evidence/2026-11-22-ltx03-all-block-native-self-attention/README.md. This is declaration coverage only, NOT full six-kind issue or TP1 parity. Next owned LTX03 gates: model-backed text/cross-modal Q/KV declarations with head/row geometry, FFN/modulation/residual, full 48-block issue trace. RUM14 owns one production-native resident loader; LTX02 owns captured TP1 activation parity; retain production TP3 refusal.

2026-11-22 next implementation blocker confirmed from the pinned LTX2 source: arch/ltx2/ltx2.cpp attn_declare (near line 1225) deliberately rejects every TP3 non-self attention with the tp && !self guard, and currently requires rows_q==rows_k and qdim==kvdim==odim==H*hd. Production's 48-block declarations include attn2, audio_attn2, a2v and v2a as non-self, with separate Q and KV sources and potentially unequal query/key rows (near lines 1749-1764). LTX03 next RED gate is one source-backed non-self native declaration per variant/split, validating distinct K/V shard, all-head Q/K normalization, exact output reduce and single BF16 bias. Implement by generalizing the existing attn_declare/Act/Lin path in rad0060, not adding another attention engine. Source-level blocker; no GPU execution claimed.

## Change Log

- 2026-11-22T03:06:36.091610+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:13:28.884198+00:00 (updated-by): Updated: section:notes
- 2026-11-22T12:39:29.662317+00:00 (updated-by): Updated: section:notes
- 2026-11-22T12:40:04.410369+00:00 (updated-by): Updated: section:notes
