# HD2701 execution progress (2026-11-09)

## Completed in isolated worktree

- [x] Created exp/qwen4-27b-hetero-dense from origin/main fa2416f67.
- [x] Located Radiance Qwen3.8-27B MXFP4 TP2 layout and llama.cpp Q8_0 lane.
- [x] Inspected RAD49, RAD40/42, separately developed RAD50/RAD51,
      KAB01, mixed RCCL runbook and large-model validation pitfalls.
- [x] Created HD27 planning group and detailed staged HD2701 plan.
- [x] Implemented card-free contiguous-stage optimiser for proposed
      xtx_tp2, r9700_only, layer3 and hybrid_tp2_r9700 layouts.
- [x] Added classical KV size and non-even TP3 head split checks.
- [x] Targeted tests: 12 passed; with docs catalog: 16 passed; Ruff clean.

## Remaining, gated by evidence

- [ ] Capture official Qwen4-27B card/weights and confirm topology and
      support; do not infer architecture from prerelease third-party notes.
- [ ] Confirm model, builds, runtime and effective per-card VRAM headroom.
- [ ] Under exclusive lease, collect single-card and dual-XTX TP2
      same-quant controls with correctness, timings and allocator report.
- [ ] Attempt minimal, isolated -sm layer canary; fail closed if loading
      or KV placement is unsupported.
- [ ] Measure stage compute and transfer with real activation payloads;
      replace synthetic estimator inputs.
- [ ] Establish optional CPU push vs GPU pull transport safety gates
      without lifting RAD50's MoE-specific assumptions.
- [ ] Reconcile pipeline latency/throughput model with real multiple-session
      measurements and scheduler/PCIe contention.
- [ ] Quant and KV dtype sweep, quality gates and long-context memory.
- [ ] Write design review for hybrid graph boundaries; only then consider
      an opt-in serving patch on a separately reviewed branch.
- [ ] End-to-end A/B, power, p95/p99, KL and production promotion review.

No GPU inference benchmark or hybrid engine implementation was performed
during the card-free milestone. Synthetic example numbers are not evidence.
