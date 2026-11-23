# LTX uneven 48-block guidance-pass placement

**Status:** Native three-card opt-in lab prototype, not a production placement feature. All cards hold one complete copy of native-selected IQ8R DiT weights; no P2P and no changes to the shipped engine or scheduling. The fourth gfx1030 card is not assigned DiT work.

## Method

- One native 48-block video+audio DiT invocation is performed for each of four guidance variants. Original 2/1/1 schedule runs variants 0 and 1 successively on gfx1201 and variants 2 and 3 on separate gfx1100 cards.
- Generate a range-enabled function from the pinned Radiance LTX2 `dit_eval` at CMake configure time. This reuses native declarations, GPU operation issue paths, kernel registry, IQ8R weight shares/relayout, and existing per-card Ctxs; it is compiled **only** with `LTX_PASS_UNEVEN=ON`.
- A split after block k keeps the start-up pin operations and blocks [0,k) on an XTX, finishes blocks [k,48) and the output projections on the R9700. Four live BF16 buffers cross the boundary: video/audio hidden states and the pending video/audio FFN outputs. A single hidden buffer would discard the residual gating needed by the next block. Transfer is GPU → pinned host → GPU, with both streams explicitly synchronized.
- Execute the first block and the remaining 47 blocks separately on each relevant card and verify their full output is bit-exact to the unsplit same-card evaluation. Derive first-block, average remaining-block, and transfer times from these **executed kernels**. Exhaustively search the per-pass block boundaries and both R9700 tail orders against those measured components. No hand-picked cut constant or per-card coefficient. The optimizer's prediction is a model, **not** proof of the global timing optimum.
- Verify R9700→pinned→R9700 transfer across two actual contexts against a complete unsplit R9700 DiT reference, bit-for-bit. Verify every concurrent split output against the serial split output on the corresponding devices, bit-for-bit. Report cross-architecture deviation separately, not as a passing identity check.
- All full weights are loaded before timing. Each arm has its own short GPU lease. The numbers represent one full four-variant DiT evaluation, not real text conditioning, VAE decode, full video generation or startup.

## Measured results

| Geometry, frames | One R9700 four-pass | Original 2/1/1 | Uneven measured | 2/1/1 / uneven | Capacity-only bound | XTX-first blocks (variants 2,3) |
|---|---:|---:|---:|---:|---:|---|
| 256×256×17 (smoke) | 440.571 ms | 269.984 ms | 244.977 ms | 1.1021× | 233.333 ms | 43, 48 |
| 768×512×121 | 6,341.026 ms | 3,638.172 ms | 3,491.913 ms | 1.0419× | 3,358.818 ms | 43, 48 |
| 1216×704×121 | 16,845.074 ms | 10,674.068 ms | 9,737.068 ms | 1.0962× | 9,391.444 ms | 39, 46 |

At 768×512×121, the active hand-off moved 101,703,680 bytes through pinned host memory in 22.565 ms while concurrently placed (24.922 ms in isolated calibration). The measured saving relative to this run's 2/1/1 baseline is 146.259 ms, so the hand-off overhead does **not** consume the gain. Selected cut prediction was 3,456.146 ms; measured 3,491.913 ms. The capacity-only bound assumes ideal divisibility with no synchronization or copying, and is not a real achievable measured target.

At 1216×704×121, both active pinned transfers move 220,192,768 bytes: concurrent hand-offs are 48.566 and 48.878 ms, while isolated calibrations are 50.503 and 49.436 ms. The measured saving is 937.000 ms over the in-run 2/1/1 baseline. Predicted chosen-cut makespan was 9,710.509 ms, measured 9,737.068 ms. Both same-device pinned hand-off exactness checks pass (cuts at 39 and 46).

Same-device full-vs-split bit equality and sequential-vs-parallel equality pass on all three measured workloads, across all four variants. Cross-gfx1100-vs-gfx1201 outputs are **not bit equal**: 768×512 variants 2 and 3 have relative BF16 video RMS deviation 1.526% and 1.480% respectively, with audio 2.079% and 1.893%, consistent with the earlier lab's heterogeneous-kernel drift. At 1216×704, the corresponding cross-device relative BF16 video RMS deviations are 1.521% and 1.477%, with audio 2.033% and 1.891%. This result **does not** authorize production heterogeneous numerical equivalence.

Raw path-free native receipts: `smoke.txt`, `768x512x121.txt`, and `1216x704x121.txt`.

## RX 6900 XT (gfx1030): pipeline alternatives, analysis only

The native Gemma-4 text-encoder declaration calls BF16 embedding lookup, IQ8R dense projections for Q/K/V/O and FFN gate/up/down, head RMS norms, Neox RoPE, causal dense attention for the full/sliding classes, BF16 activation/residual/scale/cast and the tanh-GELU gated FFN. The existing `libr2/units.txt` exports only `r4d_moe`, `r4d_fused_quant_fp8`, `r4d_quant_act_fp8`; it provides **neither** the `gemm_iq8r_tiled` dense IQ8R GEMM row **nor** a compatible attention row. `archs.toml` admits gfx1030 only in the helper role. The libr2 component's recorded native ABI also must be requalified for the present engine.

The RDNA2 16×16×16 tile in `r3_mma16.h` uses f16 VALU dot2 and BF16 cross-lane FP32 FMA emulation. It may serve as a building block for a correctly decoded BF16 dense tile, but **cannot be registered as an IQ8R row or Gemma attention kernel as-is**. Kernel metadata, weight/scale decode, activation layout, Q/K per-head operators, rotary, causal GQA attention, norm/FFN primitives, entire model shape sweep, correctness, and runtime selftests are still needed. It is likely compute-limited compared with native RDNA3/4 matrix hardware.

Gemma weights are approximately 12.08 GiB. A two-prompt 49-level BF16 hidden-state stack is 770,703,360 bytes (~735 MiB), leaving limited but potentially testable space on a 16 GiB card for additional activation arena and a 1 GiB nominal reserve; **not yet measured or admitted**. Co-resident connectors add ~2.98 GiB, exceeding the remaining practical envelope before temporary buffers and fragmentation. Therefore connectors should stay on a faster card, with the full stack transferred through pinned host once per prompt condition request, not once per denoising step. The much smaller ~24 MiB final contexts are possible only if the connector towers also move, which currently has neither memory nor kernel proof.

The video/audio VAE is also not currently executable by libr2: it requires convolution, norms, up/downsampling, resampling, codec/other support and measured workspace beyond the MoE/quantizer-only plugin. Moving a decoder would further stress the 6900's PCIe3 x4 link. It could be investigated as a **separate** kernel-qualification project, but no decode-time benefit is established.

**Prompt encode time is unmeasured and cannot be quoted as a benchmark.** An illustrative two-prompt dense-compute floor: if ~12 billion effective 8-bit weights participate in 2×1,024-token encode, approximately `2 × 12e9 × 2048 = 49.2e12` MAC-counted-as-2-FLOP work is needed, or about 2.4 s at an *unattainable sustained* 20 TFLOP/s FP32 ceiling, before attention/dequant/transfer. For **hypothetical** effective 1–5 TFLOP/s from software-emulated dense kernels, the compute component alone would take roughly 10–50 s, plus attention, transfer, and synchronization. Neither throughput assumption has been established on the 6900; the current plugin cannot run this workload at all. The ~735 MiB text-stack copy over PCIe3 x4 itself has a lower-bound order of 0.2 s in the slow direction, with real copies potentially slower. A lease-gated full Gemma kernel and end-to-end VCOND run is mandatory before promising any latency advantage.

## Decision

Both requested 121-frame laboratory splits show measurable speed gains without an extra DiT rank and preserves same-device exactness. It does not resolve cross-architecture numerical acceptance, demonstrate full generation quality or prove a deployable 6900 worker. Keep the feature lab-only until those gates pass.
