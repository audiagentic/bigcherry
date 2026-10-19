---
id: RCS09
order: 9
plan: build-radiance-components
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# No model constants in the helper: model shape from the container, model settings from an external profile

## Description

Owner direction 2026-10-19: do not tie the helper (or any BigCherry radiance component) to a specific model. Today moe_remote_shm.h hard-codes Flash-Next:
kHidden 2560, kTopK 10, kExperts 512, kMaxLayers 128, kMaxRanks 2 and the activation group, and the shared-memory header rejects any other shape on purpose
(fail closed). Different quants of the same model are already fine (rows per step do not depend on weight format); a different MoE model, expert count, top-k or
hidden size is not. Two kinds of fact are tangled in the code and must be separated:

1. MODEL SHAPE (a property of the container): hidden size, experts, top-k, layers, expert intermediate width, shared experts, activation group, weight formats.
   Read from the .rad container metadata at helper start and written into the shm Header as runtime values; engine and helper compare them at attach (fail closed
   on any mismatch, as now). Buffer capacities become bounds derived from the shape (or a compile-time maximum with an explicit, checked limit), not model-named constants.
2. MODEL SETTINGS (tuning, not shape): lane sizes, row limits, GEMV/grouped crossovers, ownership budget and heat file, helper count and devices, expert pool size.
   Externalised in the model's runtime profile (src/profile/*.ini and the recipes' runtime-profile tables), keyed by model, not baked into code or into a lab config.

Profiles hold only what differs from the defaults (owner rule 2026-10-05), and anything derivable is derived rather than configured (owner rule 2026-10-19): the defaults come from the
shape and the serving flags (--max-num-seqs, speculative depth), see RCS10; a profile key is justified only by what cannot be derived.

## Steps

1. Inventory every model-specific constant and assumption in libr3/helper and the rad0002/rad0004 hooks (kHidden, kTopK, kExperts, kMaxLayers, 2560-wide staging, 512-entry
   tables, E4M3/group 128 activations, W4 layouts) and classify each as shape, capacity bound or tuning.
2. Header v14: shape fields filled from the container; validate() compares them; buffers sized from the shape with a checked maximum; kernels take shape as arguments or
   template parameters selected at load (a small set of instantiations, not a per-model fork).
3. Move tuning knobs behind a profile section with a documented schema and offline validation (bigcherry check): unknown keys fail, ranges are checked.
4. Tests: the shm test and the replay harness run with a second synthetic shape (different hidden, experts, top-k) and must pass bitwise against the CPU reference; a model whose shape
   exceeds the checked maxima is refused with a clear message.
5. Document the model-onboarding checklist in docs/reference/engines/radiance (what to add for a new model: container, profile, shape check, tuning run).

## Validation

Shape read from the Flash-Next container equals today's constants (bitwise identical replay, unchanged acceptance numbers); a synthetic second shape passes the replay
and shm tests; no model name or model constant remains in the helper sources (a lint rule, in the spirit of the engine-name literal lint).

## Notes

Part of the component split (RCS04/RCS05): the moe-remote and backend components take their shape from the header, not from compile-time model constants. Related: RAD11 (other
expert quantisations), RCS07 (other architectures), RCS10 (tuning with the release build).
