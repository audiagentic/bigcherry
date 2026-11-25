---
id: RCS21
order: 0
plan: build-radiance-components
state: pending
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

## Change Log

- 2026-11-25T00:40:51.713419+00:00 (created-by): Created by lead
- 2026-11-25T00:41:32.230236+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:44.694931+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-11-25T01:08:41.319466+00:00 (updated-by): Updated: section:notes
