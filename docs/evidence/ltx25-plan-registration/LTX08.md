---
proposed-id: LTX08
group: build-radiance-ltx25-tp3
proposed-state: pending
priority: P0
work: M
skill: advanced
owner: Planning MCP registration required
---

# Release qualified LTX TP3 configurations with reproducible rollback

**Planning status:** Candidate for registration through Planning MCP. This file is NOT an authoritative item. Proposed owning group: `build-radiance-ltx25-tp3` (check uniqueness first). Do not hand-edit `docs/planning/` or state files. Register with the same sections, priority and dependencies when the service becomes available.

**Baseline:** Radiance 2.2.0 / ABI44; BigCherry patch `rad0060_ltx2_tp3_linear_declare` is unselected and the production `ltx2` builder rejects TP3. Shared native TP3, AQL stream dispatch and exact collectives already exist. Reuse them. The reference model is the licensed local LTX-2.5 IQ8R RAD2 checkpoint. Never substitute synthetic-only numeric checks for real-model or live GPU proof.

**Working files:** `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`, `tools/lab/radiance/ltx-tp3/`, `tools/tests/lab/test_ltx_*.py`, `docs/design/ltx25-iq8r-tp3.md`. Use isolated task worktrees from current `origin/main`, coordinate with PR #1262, no redundant patch or forked scheduler.

**General gates:** patch applicability/idempotence against the pinned source, exact ABI/versioned kernel/build identity, CMake/CTest of the real source-backed native test (not just synthetic fixtures), focused pytest and tooling guards, no GPU process without a short `agent-lease` lease; reject rather than fallback if an unqualified world/geometry is selected. Keep results and raw receipts in plan-owned work-root directories and concise stable evidence in `docs/evidence/`.


## Description

Promote only proven 3-card LTX native TP3 and independently proven optional text-worker configuration through BigCherry's existing installed-release process.

**Existing engineering gap coverage:** LT3-15 and final LT3-14 release decision. **Dependencies:** LTX05 accepted; LTX07 only for optional text-worker variant.

## Steps

- 1. Gather actual LTX01–05 accepted source pins, ABI/plugin table, quality, latency, device BDF and build receipts. Reject missing plan gates, draft PR CI, incompatible license or unadmitted device.
- 2. Ensure production patch `rad0060` is selected **only** by named validated LTX TP3 source/config, leaves TP1 and other Radiance recipes unchanged, and all hashes/patch replay are recorded.
- 3. Select LTX-only production recipe/config output using existing RUM09/native release implementation; no Python interpreter, direct-HIP launcher, RUM12 GPU lab or new packager/runtime/config schema in the installed binaries.
- 4. Run actual installed TP3 121-frame video/audio generate, cancellation, resume, process termination and TP1 fallback; compare copied artifacts and exact model/config checksums.
- 5. If LTX07 qualifies independently, add separate four-card optional release only; default fallback remains TP1, no silent worker omission under a four-card measured contract.
- 6. Publish final runbook, license/read access, supported cards, link constraints, known latency/quality gate, rollback, and unreleased Ledger MCP event; merge only after required checks and review.

## Detailed Solution & Technical Design

Generic bundle/manifest/release implementation remains with the existing RUM/native release owners; LTX08 owns only the LTX-specific release decision, recipe and model/output evidence. No new runtime config source or forked launch command. Keep inactive diagnostic-only code until release criteria proved.

## Code Samples & Guidance

```text
main + pinned Radiance 2.2/ABI44 -> existing rad0060 + native CTest
-> narrow model-backed issue -> AQL three-rank numerical proof
-> safe owner/conditioning acceptance -> measured release decision
```

Reuse the exact current source-managed extension and existing tests; no side-path implementation of graph, device collective, kernel discovery, model parsing or config. All GPU jobs require an acquired lease, real build/source identity and exclusive card ownership. The first failing native/assertion test is required before editing a production path.

## Files

- `engines/radiance/recipes.toml`
- `engines/radiance/configs/ltx2-*`
- `docs/reference/engines/radiance/`
- `docs/evidence/ltx25-*/`

## Validation

Installed release inventory/ABI/patch check, forbidden terms, actual installed multi-rank video generation and TP1 restoration on leased hardware, release/CI and ledger gate.

## Effort & Risk

**P0, M.** M: promoting diagnostic patch prematurely or replacing TP1 silently. Refuse release on missing evidence.

## Standards

BigCherry `docs/reference/agents/WORKFLOW.md`, patch and tooling dispositions, stable exact native ABI, source/weight identity, tests before production code, measured proof before speedup claims. No host paths, hand-picked model constants, new config/parser, shims or duplicate path. Planning/ledger state is MCP-owned.

## Acceptance Criteria

Only certified named LTX config(s) ship; default TP1 intact, physical three-card output and quality/parity/performance records available; rollback confirmed, project checks green, planning/reviews/ledger properly closed via respective MCPs. Otherwise remain diagnostic and open.

## Notes

PR #1262 contains foundation and diagnostics, not a qualified production LTX feature; merge decision for that diagnostic PR is distinct from future production promotion. Only the existing Radiance engine/core and tested LTX-specific hooks may ship: **do not install historical C07 direct-HIP or LTX01 AQL lab executables** in the released runtime.

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This item is a **registration candidate** until Planning MCP creates it.
