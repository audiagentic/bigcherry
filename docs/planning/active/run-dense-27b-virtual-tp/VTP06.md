---
id: VTP06
order: 5
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Q6/Q8 model and four-plus-stream quality/capacity qualification

Separate this work from VTP05 kernel-only screening and VTP04 isolated transport.

1. From the original, fingerprinted Qwen3.8-27B BF16 checkpoint, select a W8A8 baseline and calibrated IQ6R variants. Never equate GGUF Q8_0/Q6_K import support with a native Radiance Q8/IQ6R kernel. Do not re-quantise Q4 weights. Use converter plan-only and matching kernel ABI and producer activations first. IQ6R needs qualified calibration Hessians and the selected model recipe.
2. Build opt-in candidates in isolated scratch. Confirm tensor/split legality, scales, graph dispatch and gfx11/gfx12 kernels, numerical oracle and independent top1/KLD against BF16. Reject immediately on quality regressions or unsupported operator paths.
3. Audit per-physical-card weights, full-attention KV, recurrent/GDN state, embedding/head, activation/graph scratch, drafter and driver reserve at 1/4/6/8 streams. No silent host spill. Context tests 8K/32K/64K/128K/256K within the current checkpoint's native limit. One future 1M request requires a model that genuinely supports it.
4. Compare same-format controls: dual-XTX TP2, R9700 independent replica if it fits, TP2-XTX plus R9700 whole-layer stage when present, and only then virtual TP4 (VTP03). Match checkpoint, context, quant, KV precision, sampling, MTP, runtime pin and benchmark corpus.
5. Go/no-go based on four-plus-stream aggregate tokens/sec, p95 latency, resident context, safe memory admission, numerics and clean shutdown. No microkernel-only speedup authorizes production promotion.

Do not create another worktree. Keep vendor/radiance pinned and read-only; managed patches only. No model download unless the local BF16 checkpoint or required calibration evidence is truly absent. No hardware runs without the GPU lease.

PENDING VTP05 screening and VTP01 execution design.
