# VTP27 — virtual four-way tensor sharding on three physical GPUs

Status: research and card-free qualification only. No virtual rank runtime, GPU benchmark, production layout or memory admission exists.

**Owner requirement:** dense 27B served at Q6 or Q8 (not Q4), maximise *resident* KV and sustain 4+ simultaneous streams. Physical hardware: gfx1100 RX 7900 XTX 24 GiB ×2 and gfx1201 R9700 32 GiB ×1. PCIe upstream: XTX Gen4 x8 each, R9700 Gen4 x4. No proven GPU-to-GPU P2P.

Three independent, bounded work items:

- [VTP01](VTP01.md): logical-rank/physical-executor contract, supported-geometry analysis and safe three-device collective design.
- [VTP02](VTP02.md): Q6/Q8 + resident-KV feasibility, card-free ratio/capacity experiment and geometry gates.
- [VTP04](VTP04.md): fast local R9700 and three-physical-GPU microbench harnesses (compile-only and leased runs).
- [VTP05](VTP05.md): actual Radiance IQ6R/W8A8 model-shaped GEMM screening on each physical GPU. Includes an offline preflight and resumable single-case benchmark campaign.
- [VTP06](VTP06.md): separate Q6/Q8 model quality and resident-KV validation gate.
- [VTP07](VTP07.md): actual Q6/Q8 M4 GPU results and the gfx11 IQ6R layout compatibility blocker.
- [VTP08](VTP08.md): native W8A8 RAD2 FFN plus FP8 attention-KV strict physical-VRAM lower bounds and no-go cases.

- [VTP03](VTP03.md): gated Q6/Q8 full-model prototype, 4+/long-context A/B benchmark and go/no-go.

Dependencies only, not duplicated:
- XDP01/XDP03: Radiance dual-XTX current-pin and BF16-derived Q8/Q6 kernel/quality work.
- XDP02: TP2 plus R9700 **whole-layer** staging and capacity alternative.
- HD2701 (separate dirty worktree): generic three-GPU layer pipeline/llama.cpp controls.
- PGC09/PGC11/PGC15, RAD04: small-message host collectives, wire quality, prefill overlap.
- KAB01: repeatable kernel A/B. RAD50 is MoE BAR push, not a dense TP service.

Work in the existing `exp/xtx27-prefill-opt` worktree. Do not touch the separate hetero worktree, XDP09's untracked content, vendor sources, current engines, GPU without an exclusive lease, or production defaults. Commit separately; no push/PR without request. Memory/perf predictions below are **not** verified admission/performance claims.
