---
id: RCS21
order: 0
plan: build-radiance-components
state: in_progress
created-at: '2026-11-25T00:40:51.713419+00:00'
breadth: ''
skill: advanced
created-by: lead
work: M
priority: P1
---

# Arch-neutral kernel formulation behind a traits seam, unified fused tile, and the upstream path

## Description

Write new tile code once against a small compile-time traits type (wave size, native WMMA operand layout accessors, accumulator conversion, LDS capacity and granularity, register budget, maximum waves) so gfx12, the gfx11 generated tree and any native gfx11 provider each supply only their traits. Arch-neutral changes then become small upstream-shaped patches.

## Steps

1. Define the traits header and port the fused single-band IQR prefill tile (gated on both architectures) as one source. 2. Port the wave-local embedding butterflies and the widest-token-tile head prefetch. 3. Keep gfx11-shim-specific wins (accumulate-once, native fragments, native head sums, chained long tiles) as gfx11 rewrites; they change or retire with the native provider. 4. Prepare each neutral patch as an upstream-shaped patch (one per unit, description, correctness test, resource table, replay evidence); offer upstream only with the owner's approval; on a pin bump archive an accepted local patch unchanged.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 5, including which changes are neutral.

## Code Samples & Guidance



## Files

engines/radiance/patches (new packages), engines/radiance/components/libr3/native, docs/design

## Validation

One source for the fused tile builds on gfx1100 and gfx1201; resource records, byte hashes and cold replay equal the shipped per-architecture versions; controls unchanged.

## Effort & Risk

Medium. Risk: a traits seam that costs registers; each port is gated on resource records first.

## Standards



## Acceptance Criteria

A shared fused-tile source retains each architecture's native operand and accumulator representation, qualified geometry and dispatch certificate. Both architectures give byte-identical results with no measured performance regression (cold and warm replay) and unchanged control kernels. Resource records and instruction counts (WMMA, lane exchanges, barriers, loads, accumulator conversions) are compared with the shipped per-architecture versions and any difference is reviewed. The embedding butterflies and head prefetch are ported only after this first port shows zero regression.

## Notes

Depends on RCS19 (anchor gate, naming). Evidence for each neutral change: the fused-prefill, embedding, head and weighted-head evidence notes.

Review (2026-11-25): the traits layer must resolve at compile time and inline (no runtime trait object, no conversion through a common accumulator layout); the fused tile is neutral algorithm and policy with architecture-specific data motion; keep the qualified certificate (computed trellis forms, L=16 and the one L=15, scale classes, single band) rather than a broader traits-derived rule; launch bounds, wave count and shared memory placement stay part of the qualified design.

Review (2026-11-25): upstream-shaped patches that come out of an owned overlay are export artifacts for submission, not a second production transformation layer: the overlay stays the production source until upstream adopts the change (change-level adoption, RCS23).

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Keep traits seam narrow and architecture-neutral where valid. Declare actual upstream include closure, row/ABI assumptions and test shared-traits changes across affected arch/pass/provider under RCS19/RCS23. No broad new traits/compat layer solely for bump tooling; upstream adoption is local retirement after equivalence, no upstream contribution assumption.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Native libr3d int8 wave-matrix layout/group-rounding and scalar packed IQR added on candidate; do not force same operand traits where layout/storage differs. Derive shared math only where actual arithmetic and storage allow, use native base for covered rows, local overlay for required deltas. Keep native upstream layout and ABI, no general compatibility shim or backport.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Native libr3d int8 wave-matrix layout/group-rounding and scalar packed IQR added on candidate; do not force same operand traits where layout/storage differs. Derive shared math only where actual arithmetic and storage allow, use native base for covered rows, local overlay for required deltas. Keep native upstream layout and ABI, no general compatibility shim or backport.

2026-11-25 closed as deferred: the traits seam and the unified fused tile are a maintainability and upstream-path refactor with no measurable gain (the harness can prove equal bytes and resources, not speed). The gfx1100 and gfx1201 fused tiles each ship and are gated (rad0065, libr3 rewrites, find-1-or-0 pairing). Reopen when upstream asks for the patch or a bump makes the two copies diverge.

2026-11-25 reopened with the lib rework (see RCS23 notes). Finding that sizes stage S2: the fused single-band IQR tile exists twice as near-identical text, once in the rad0065 patch (libr4d, names IqrFused12*, guarded by #ifndef R3_TARGETS) and once in libr3/native/r4d_gemm_iqr.hip.rw (IqrFused*); the struct, the certificate, the kernel and the one-band dispatch are the same code, and the gemm_8a8_prefill_body Fused parameter already comes from the patch for both architectures (the libr3 rewrites carry find-1-or-0 pairs for it). Plan: one copy, arch-neutral, inside the patch (no #ifndef, one set of names), the libr3 fused blocks deleted from the .rw (the patched source already carries the tile), the naming lint taught the arch-neutral suffix by data (no exception hand-list). Proof by the unit harness on both architectures: iqr family on gfx1100 and gfx1201 against the shipped release, output hashes equal, resource records equal except the renamed gfx1201 fused symbols (listed), replay of the 65 to 128 row points no regression. A traits header is introduced only if a real difference between the two copies is found; none is expected, so the seam stays a note until a second architecture-specific tile appears.

## Change Log

- 2026-11-25T00:40:51.713419+00:00 (created-by): Created by lead
- 2026-11-25T00:41:32.230236+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:44.694931+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-25T01:08:41.319466+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4607
- 2026-11-25T02:05:13.382985+00:00 (updated-by): Updated: section:notes
- RV4632
- 2026-11-25T02:40:25.241991+00:00 (updated-by): Updated: section:notes
- RV4633
- 2026-11-25T02:40:46.557497+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:42:24.891543+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:43:00.957766+00:00 (state-transition): State: pending → deprecated
- 2026-11-25T10:44:47.321276+00:00 (state-transition): State: deprecated → in_progress
- 2026-11-25T10:46:36.297164+00:00 (updated-by): Updated: section:notes
