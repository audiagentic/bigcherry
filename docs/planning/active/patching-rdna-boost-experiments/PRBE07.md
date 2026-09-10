---
id: PRBE07
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:00.396867+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate BridgeSpec (kdheeraj-p/bridgespec) -- HIP MTP/DFlash speculative decoding sidecars + gfx1100 MMVQ tuning

## Description

Investigate the BridgeSpec third-party HIP MTP/DFlash sidecar and gfx1100 MMVQ tuning as an unvalidated candidate, not as production evidence.

## Steps

- Read the pinned README, benchmark README/CSV, integration patches and MMVQ tuning code in full; record source commit and license.
- Compare verification-width 2-8 techniques against THA05/RD-series dispatch without importing unverified claims.
- Assess sidecar ABI, host-mediated KV state, vocabulary slicing/remap, dual-XTX applicability and VRAM co-residency risks for our MTP work.
- Treat published numbers as directional only; define a reproducible local 9B/27B probe with acceptance and fallback gates before any adoption claim.
- Conclude adopt/adapt, inspiration-only, or not applicable with explicit evidence and provenance.

## Detailed Solution & Technical Design

This is research triage for a third-party preview. Keep sidecar architecture and MMVQ tuning separate, preserve our existing runtime-profile VRAM preflight, and do not generalize single-GPU Vulkan/HIP results to dual-XTX tensor-split production.

## Code Samples & Guidance



## Files

pinned BridgeSpec source/README/benchmarks; integration patch review notes; THA05/RD-series comparison; reproducibility probe recipe and evidence.

## Validation

Source/license review; controlled local reproduction where authorized; acceptance-rate, throughput and memory-budget measurements; single/dual topology controls; no promotion without independent evidence.

## Effort & Risk



## Standards

Third-party research-preview provenance; never fabricate promotion evidence; separate controlled A/B from high-water claims; preserve memory-safety and VRAM-budget gates.

## Acceptance Criteria

A written recommendation identifies concrete transferable technique or explicitly rejects applicability; no third-party benchmark is represented as BigCherry evidence; any adopted change receives its own patch and qualification item.

## Notes

Supersedes: RD101
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd101

## Change Log

- 2026-09-09T10:54:00.396867+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:06.504082+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.160426+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.848976+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:32:20.227395+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:33.005493+00:00 (updated-by): Updated: section:ledger-events
