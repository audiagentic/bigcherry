# HD27 — heterogeneous three-GPU dense 27B deployment

> Archived 2026-11-12 from branch `exp/qwen4-27b-hetero-dense` (git tag `archive/exp-qwen4-27b-hetero-dense`), which never merged. The offline estimator and lab scripts (tools/lab/dense27) are in that tag. Heterogeneous dense serving was delivered by VTP14 and VTP33 (native TP3).

Status: experimental planning group
Prefix: HD27
Owner scope: dense Qwen3.8-27B as an existing measured proxy; Qwen4-27B only after verified architecture and weights are published.

## Objective

Determine when two RX 7900 XTX (gfx1100, 24GB each) plus Radeon AI PRO R9700 (gfx1201, 32GB) improve quality-qualified end-to-end serving for dense 27B, versus two-XTX tensor parallel or a lower-precision single-card model.

Model real memory ownership, layer/cross-stage costs and per-device KV/recurrent state. Avoid assuming one big shared VRAM pool, evenly splittable heads, or that the fastest individual kernel produces the fastest end-to-end system.

## Plan index

- [HD2701](HD2701.md) — Baselines, card-free placement estimator, live topologies, KV and quantization gates, hybrid design review.

## Existing dependencies and boundaries

- Radiance qwen27b-2xtx layout (Qwen3.8-27B MXFP4), RAD49 TP2 kernel/exchange work, RAD48 gfx11 GEMMs and KAB01 fixture A/B.
- llama.cpp tierL-qwen27b-q8, docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md, current tensor split mode and hetero RCCL admission runbook.
- RAD40/RAD42 measured transport and RAD50/RAD51 CPU BAR-push ideas in the separate experimental branch. Those target MoE result planes, NOT an available dense-layer handoff.
- RSC configuration/provenance conventions, per-device META allocation accounting MSM01–03 and existing quality/benchmark harnesses.

No new production layout, vendor modifications, model assumptions or default changes in stage 1. GPU work requires the exclusive lease.
