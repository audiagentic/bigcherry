---
id: LTX08
order: 8
plan: build-radiance-ltx25-tp3
state: pending
created-at: '2026-11-22T03:07:11.270369+00:00'
breadth: ''
skill: advanced
created-by: planning-audit
priority: P0
work: M
---

# Release qualified LTX TP3 configurations with reproducible rollback

## Description

Promote only proven 3-card LTX native TP3 and independently proven optional text-worker configuration through BigCherry's existing installed-release process.

**Existing engineering gap coverage:** LT3-15 and final LT3-14 release decision. **Dependencies:** LTX05 accepted; LTX07 only for optional text-worker variant.

## Steps

- 1. Gather actual LTX01–05 accepted source pins, ABI/plugin table, quality, latency, device BDF and build receipts. Reject missing plan gates, draft PR CI, incompatible license or unadmitted device.
- 2. Ensure production patch `rad0060` is selected **only** by named validated LTX TP3 source/config, leaves TP1 and other Radiance recipes unchanged, and all hashes/patch replay are recorded.
- 3. Select LTX-only production recipe/config output using existing RUM09/native release implementation; no Python interpreter, direct-HIP launcher, RUM13 GPU lab or new packager/runtime/config schema in the installed binaries.
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

References: `docs/design/ltx25-iq8r-tp3.md`, `docs/design/ltx25-tp3-component-test-ladder.md`, `docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md`, PR #1262. This is an authoritative pending plan item, registered through Planning MCP; historic evidence remains at the linked path.

**Original plan design and scope evidence:** `docs/evidence/ltx25-plan-registration/LTX08.md`. Registered by the canonical Planning MCP via AUDiaGentic. Historical C milestones are not separate plan IDs.

## Change Log

- 2026-11-22T03:07:11.270369+00:00 (created-by): Created by planning-audit
- 2026-11-22T03:14:49.654502+00:00 (updated-by): Updated: section:notes
