---
id: RAD23
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T10:48:18.492687+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# moe-remote: helper count and topology as runtime values (no fixed 2), selectable combine methods

## Description

Owner rules (2026-10-23, CLAUDE.local.md): code for unknown topologies - no fixed helper/device counts; topology-dependent methods stay selectable with a derived auto. The moe-remote protocol and helper hard-code two: kMaxRanks = 2 (moe_remote_shm.h) indexes result planes [rank][helper], combine.lane/seq are 2-element arrays, and R3_MOE_WORKER_COMBINE requires helper_count == 2. Make helper count (1..N) and engine ranks runtime values with separate axes, and make the helper-row combine a selectable method: R3_MOE_WORKER_COMBINE=off|cpu|gpu|auto (later p2p).

## Steps

1. GPT design review (req_bigcherry_97cebda33722 + addendum req_bigcherry_3bb6b763ab97): protocol for N helpers, GPU-through-host combine race-freedom, overflow fallback, auto selection, prior art.
2. Protocol: helper count and engine ranks as runtime header fields; per-helper arrays sized from them; layout_for derives region sizes; version bump. Remove every hard-coded 2 / kMaxRanks used for helpers.
3. Combine methods behind one interface: off, cpu (AVX2/F16C, already on feat/combine-fp16), gpu (combiner helper's stream waits device-side on the other helpers' done words, reads their rows from host memory, adds its own in VRAM, writes one fp16 row with the per-row fallback marker). 1 helper resolves to off.
4. auto: measured once per host and cached - P2P availability, per-card host-link bandwidth, CPU features - choosing method and combiner helper (widest link; tree reduce when N-1 planes over one link loses). Per-model profile may override.
5. Harness first: replay/combine bench for each method with 1, 2 and synthetic N planes before end-to-end.
6. KLD gate and 4-pair A/B per method on Brutus vs the parent build.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Component tests with helper counts 1, 2 and 3 (shm layout, combine correctness per method, fallback rows); KLD gate passes for cpu and gpu; helper-ab A/B per method; grep shows no hard-coded helper count.

## Effort & Risk



## Standards



## Acceptance Criteria

No fixed helper count in moe-remote code or protocol; off/cpu/gpu/auto selectable and auto derived from measured topology; the chosen method on Brutus beats combine-off on prefill with tg not worse and KLD within the gate.

## Notes

Follows RAD18 (combine). fp16 combined row passed KLD (p99 +0.039). CPU scalar fp16 conversion erased the gain (~1 ms/step); F16C version under A/B. Owner: the combine should run on the GPUs where possible.

## Change Log

- 2026-10-23T10:48:18.492687+00:00 (created-by): Created by agent
- 2026-10-23T10:48:36.867246+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
