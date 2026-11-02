---
id: RSC08
order: 8
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T12:14:33.492000+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Serving manifest v1: generated contract, shared loader and one-release cutover

## Description

Replace the interim serving manifest with the locked v1 wire contract (docs/design/serving-manifest-v1: spec, errors, schema and examples, integer-only per the spec's §11 amendments). One C++ field registry generates the typed model, the strict reader and writer, the JSON Schema, the engine and helper bindings, the field docs and a contract fingerprint. One bc_serving library is linked into bc-serve, radiance and r3_moe_worker. The engine takes native typed configuration (no synthetic argv), helpers take typed role views, and the cutover happens once, in one release, with no shims. Source: GPT design review 2026-11-02 (req_main_c69d1519f282, gpt-response-1.md) and the owner's decisions: a curated engine option surface; the manifest owns the initial owner-map selection and digest while live protocol epochs own later placement; a one-release incompatible cutover.

Fed by RSC01 (serving component, schema and loader), RSC02 (bc-serve as the only start-up path), RSC03 (resource plan: detection, calibration store, host ledger, owner map, measured-plan cache, rad0011), RSC06 (typed helper settings and plugin ABI) and RSC07 (the engine's manifest entry point).

## Steps

1. Contract and generator: the canonical field registry, a C++ generator (rejects float types), the complete v1 schema (checked against the committed docs/design/serving-manifest-v1 copy), the typed model, a schema fingerprint and generated docs, plus the JCS encoder with RFC 8785 vectors. No serving behaviour changes. The contract documents landed in #528.
2. Shared strict reader: size-limited open, duplicate-member rejection, apiVersion/kind check, JCS digest verification, structural and semantic validation, provenance coverage, typed engine and helper views, and structured ConfigError (BC_CFG_* on RAD_E_*, numerics checked against abi/rad_types.h).
3. RSC03 resource core on v1: detection (cgroups, memlock, pin cap), the calibration matcher (integer seeds), the host ledger (tier_first), the owner map (spec.plan.owner_map) and the measured plan (spec.plan.engine_memory with the v1 cache key). Evidence-backed budgets only; the receipt echo expected.host_pool_mib is verified equal to resources.host.pool_mib.
4. RSC06 helpers rebased onto v1: typed helper views replace WorkerConfig and manual extraction; the size- and version-checked plugin ABI is kept; helpers verify their initial owned set.
5. Native radiance integration: a managed patch links bc_serving into rad_core (thin CMake edit, no fallback); a generated config_apply_manifest replaces rad0009's argv converter; engine-side R3_* reads move to typed plugin configuration; the engine plans from the manifest's owned sets, and the startup admission check (§5.2) runs before weights.
6. bc-serve supervisor and tooling cutover: one immutable manifest; engine and helpers start with manifest and role only; the engine binary and plan cache come from the package and the work root; the shell launcher, the Python layout duplicate and the raw engine-arg path are deleted and lab callers migrated.
7. Release gate and documentation: schema fixtures, upstream rebase tests, no-legacy checks (no application getenv("R3_*"), no synthetic argv), RHP12 equivalence, generated inventory coverage, clean README and component docs.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

Cutover criteria (GPT review §F):
- The exact same v1 manifest loads successfully in bc-serve, radiance and every helper role.
- No application-level R3_* setting influences execution through environment inheritance.
- The engine's resulting native configuration matches the accepted baseline for all supported fields.
- The host and GPU budget plan is derived or calibrated with evidence; no interim 9,216/12,288/22,528 MiB constants remain as unqualified universal defaults.
- The RSC03 resource ledger and the actual engine allocation agree, including model residency, n-gram placement, checkpoint memory, pinned consumers and reserved capacity.
- RHP12's fresh and retained-session cohorts, step-time checks and numerical-identity checks pass on hardware in the normal integration workflow.
- Invalid, old-version or partly migrated manifests always fail before any GPU work.
- The release packages exactly one supported schema and its matching binaries.

## Notes

Feeds: RSC01, RSC02, RSC03, RSC06, RSC07.
Slice 1's contract documents: PR #528 (docs/design/serving-manifest-v1 and tools/tests/core/test_serving_manifest_v1_contract.py); its ledger event was recorded against RSC01 before this item existed and is relinked here.
Decisions: v1 numbers are integers only (fractions as *_ppm / *_milli); one site-only, tighten-only `constraints` namespace in ServingInputV1; the engine binary resolves relative to bc-serve's installed location and the plan cache from the work root, neither a CLI flag; generated files are committed and CI regenerates and byte-diffs them.

## Change Log

- 2026-11-02T12:14:33.492000+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261102_121239_the-serving-manifest-v1-format_6543
- chg_20261102_121824_the-serving-component-can-prod_8129
- 2026-11-02T12:18:29.161474+00:00 (updated-by): Updated: section:ledger-events
