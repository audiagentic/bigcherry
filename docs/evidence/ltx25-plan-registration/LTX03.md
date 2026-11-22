---
proposed-id: LTX03
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P0
work: L
skill: advanced
owner: Planning MCP registration required
---

# Complete the 48-block dual video/audio TP3 DiT graph without new dispatch infrastructure

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


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

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
