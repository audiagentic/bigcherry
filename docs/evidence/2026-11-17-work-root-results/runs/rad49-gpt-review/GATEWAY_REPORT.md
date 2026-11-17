# Gateway GPT review (reviewer-gpt-agent), 2026-11-10

Request req_main_4f8153bbfeb1, session ses_9ce133dfbaad4bd0. Full text is retrievable with agent_task_response; summary:

Ranked (estimates are the reviewer's hypotheses):

1. Fuse the quad reduction's `v_mov_b32_dpp` + `v_add_f32` pairs into `v_add_f32_dpp` (gfx1030 DPP16 supports it): 3 fewer
   instructions per reduction, ~640 reductions per 16-row gate/up batch per column; ~5-15% kernel, very low KL risk if the same
   values are added in the same order. Check the ISA.
2. Wasted row work and reuse: activation staging repeated by every N-tile CTA (24 for gate/up, 80 for down); quad reduction per
   row per K64 group; LDS LUT decode; and the inner loop runs all RB=16 rows even when nr < 16. Capture rows-per-expert
   histograms; specialise RB (4/8/16) by occupancy; try N=64 per CTA; only then pre-decode activations. ExLlamaV3
   EXL3_MOE_MTILE precedent (its ROCm path disables it).
3. Speed- and phase-aware placement: per-expert marginal cost C(e, device, phase) = dispatch + transfer + compute + combine;
   minimise per-layer max completion; two profiles (cold latency vs sustained prefill); replicate a few hot tail experts.
   Refs: HybriMoE, MoE-Infinity phase policies, PowerInfer.
4. Cut real PCIe bytes: unique activation rows once per helper, return only nonzero-contribution rows, double-buffered
   H2D/compute/D2H streams; adaptive DMA-BUF push only for large row counts. Stage times imply sparse transfers already; verify
   bytes on the wire.
5. Expert-major prefill scheduling (vLLM fused_moe, SGLang, llama.cpp Metal PR #13388).
6. INT8 dot4 (RDNA2 has V_DOT4_I32_I8; llama.cpp IQ4_NL -> int8 + Q8_1 in mmq.cuh / OpenCL dp4a kernel; Vulkan PR #31764
   +20% pp512 on RX 6600M but mean KL 0.00072, 98.73% same top token): not bit-exact; separate gated branch. ExLlamaV3 disables
   its INT8 GEMV on RDNA because fp16 dot2 was faster.
7. CPU execution of cold experts (Fiddler, KTransformers AMX): weak fit for an AVX2-only i7.

Verification items: 35.5 vs 58+ GiB expert accounting (resolved: real intermediate is 640, 59.6 GiB); 46 TFLOPS peak overstates
the opportunity; exact-result constraints on cross-device summation order.

Next time: this agent can read local folders through its Brutus plug-in when told to.
