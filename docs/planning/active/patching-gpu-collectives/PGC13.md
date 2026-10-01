---
id: PGC13
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-01T07:56:23.005200+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# AllReduce and allocation auto-calibration pre-run (per-topology policy cache)

## Description

Owner idea (2026-10-01): tune AllReduce and per-card allocation from measured link bandwidth and card response timing, recorded by a calibration pre-run and reused at startup. Design agreed with reviewer-gpt-agent (req_c6a8d61532fb42fc). Value ranking here: (1) provider table per size and phase > (3) in-model arrival timing > (2) per-card -ts weights. Motivation: PCIe topology is uneven (2x 7900 XTX PCIe4 x8, R9700 x4, 6900 XT on chipset, no P2P); a fixed 1 MiB switch is wrong for mixed sets; size alone cannot separate the 80 KB prompt tail from 120 KB MTP verify (PGC12), so phase is part of the key.

## Steps

1. Link calibration (offline): for each exact device set and provider (RCCL, host mapped-host, host copy-engine, root3, adaptive), time AllReduce at log sizes 4 KB..64 MB plus decode points 20/40/80/120/256/512 KiB; median + p90 after warmup; reject winners with < ~3% margin.
2. Card calibration (mostly offline): time representative MMVQ/MMQ shapes for the model family/quant per card; derive throughput weights; quantize to legal split-row granularity (128 rows for FFN); test neighbouring ratios.
3. Arrival timing (in-model, once per calibration): 32-128 representative tokens/prompts with compute_done / AR_enter / AR_done per rank; detect consistent laggards; adjust -ts or choose a tree/root; freeze, never adapt continuously.
4. Cache: key = {device UUIDs/order, PCI topology, ROCm/driver, model arch+dims+quant, context/MTP config}; entries = [{phase (prefill, decode, mtp_verify, optionally prompt_tail), size_lo, size_hi, provider, root, ts_weights}]. Recalibrate when the key changes.
5. Runtime: 0860 replaces scalar switch_bytes with a policy object loaded and validated once at comm configuration (no env lookup in the hot path); 0840 snapshots it into the comm context and does a {phase,size} table lookup per call, failing closed to RCCL then meta. Phase hint comes from PGC12.
6. Tooling: bigcherry calibrate-collectives CLI (reuses lab A/B and trace tooling), cache under work_root, recorded in tools/lab/results.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/0860_allreduce_provider_cli, patches/0840_hybrid_allreduce_dispatch, new tools/bigcherry/... calibrate command, cache schema

## Validation

Calibrated policy vs fixed 1 MiB adaptive vs RCCL on dual XTX, 3-card, and Flash-Next layouts: decode and prefill A/B, MTP acceptance, decode KLD 0. Re-run calibration twice to check the policy is stable.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Risks: thermal/DVFS drift, background PCIe traffic, insufficient warmup, device-order or topology changes, model-dependent compute ratios, oscillation near crossovers. Depends on PGC12 for the phase hint.

## Change Log

- 2026-10-01T07:56:23.005200+00:00 (created-by): Created by agent
