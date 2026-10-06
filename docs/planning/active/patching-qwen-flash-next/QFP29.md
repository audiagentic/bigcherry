---
id: QFP29
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T09:37:29.282620+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P1
work: M
---

# Qualify RDNA4 ROCmI4 W4A4 target verification before any BigCherry port

## Description

Authoritative owner for the ROCmI4/IU4 experiment. Sources inspected at implementation level:

- charlie12345/ROCmFPX at `fb08d7cdb67082ddda15a25e64dc7ec413afc41c`: original gfx1151 `Q4_0_ROCMI4` W4A4 path.
- OwNhj/llama.cpp-RDNA4-FP (current October 2026 fork): an independent RDNA4 adaptation of the same mechanism to current llama.cpp MMQ, using `v_wmma_i32_16x16x32_iu4` on gfx1201.

This materially changes the old QFP29 premise. The candidate is no longer gfx1151-only: there is now a concrete gfx1201 implementation and single-R9700 measurement. It still does **not** accelerate BigCherry's existing Flash-Next UD-IQ4_XS or 27B Q8_0 files. The kernel is selected for the custom `Q4_0_ROCMI4` weight type and therefore requires a separately quantized model from BF16/F16 (or an already-published matching ROCmI4 GGUF).

The useful mechanism is narrow: for batched MMQ (prompt processing and MTP target verification), keep signed 4-bit weights packed in LDS, quantize activations to signed 4-bit, and feed native RDNA4 integer WMMA. Ordinary small-batch decode remains the existing q8_1/W8A8-style path. Do not port the wider ROCmFPX format family, TurboQuant, or another MTP scheduler.

### Repository evidence and current BigCherry interaction

BigCherry already owns MTP scheduling/verification through the existing QFP/FMTP patches and Q8_1/MMQ work. ROCmI4 changes the target matrix-multiply representation; it does not replace draft scheduling, adaptive depth, graph ownership, KV policy, or 1307-1313 Q8_1 work.

The R9700's PCIe 4.0 x4 attachment is not the first-order discriminator for this experiment because the proposed lane keeps the entire ~15 GiB-class ROCmI4 27B model resident on the 32 GiB R9700. No per-token host expert refill is required. If a tested configuration spills weights or experts to host memory, reject that result as a different experiment and hand placement/offload ownership to MET01/RPL01.

### External measured evidence (not BigCherry measurements)

The RDNA4 fork reports on one R9700:

- Qwen3.8-27B perplexity: Q4_0 3.7755; ROCmI4 exact/W4A4-off 4.6937; ROCmI4 W4A4-on 5.4493.
- 1B dense prefill, p2048: Q4_0 25,049 t/s versus ROCmI4 W4A4 29,588 t/s (~18%).
- Qwen3.8-27B ROCmI4 has a severe `ssm_out` quantization failure if those tensors are left in ROCmI4 (reported PPL ~4.3e5). Pinning all `ssm_out.weight` tensors to MXFP4 or Q8_0 restores finite quality; Q8_0 produces a ~14.62 GiB file in the reported configuration.
- The fork also reports a tuned SYM4 tensor assignment that improves KL metrics at the same size/speed, but this is a separate quantization-quality mechanism and is **not** part of the first BigCherry W4A4 gate.

These numbers establish feasibility, not a BigCherry performance claim. The original ROCmFPX gfx1151 result (+18.66% mean MTP decode) is also mechanism evidence only.

## Code-level mechanism

ROCmFPX's `ggml/src/ggml-cuda/mmq.cuh` adds a `GGML_ROCMI4_W4A4` compile gate and a dedicated packed-nibble MMA tile for `GGML_TYPE_Q4_0_ROCMI4`. The original path is gated to gfx1151. The RDNA4 fork re-adapts that loader/packing/vec-dot path to current upstream's compile-time `ncols_dst`, `calc_nwarps` and `calc_rows_per_block` structure and substitutes RDNA4's K=32 IU4 WMMA instruction, avoiding the two K=16 operations used by the gfx1151 implementation.

Consequences:

1. Standard `IQ4_XS`, `Q4_K` and `Q8_0` weights cannot simply dispatch into this path; their block layouts/scales are different.
2. W4A4 is intentionally lossy because activation quantization changes from q8_1/int8 to signed 4-bit. Greedy identity is not an applicable promotion gate.
3. MTP is useful because target verification creates a batch large enough to enter MMQ; single-token generation remains memory-bound/MMVQ and is not the target.
4. The cheapest discriminator is therefore an **external-fork R9700 A/B on one ROCmI4 file**, not a BigCherry port.

## Plan

### Phase 0 - obtain a valid comparison artifact

1. Use Qwen3.8-27B BF16/F16 source or a provenance-matched ROCmI4 GGUF. Quantize `Q4_0_ROCMI4` with every `ssm_out.weight` pinned to Q8_0. Record source model hash, quantizer commit, tensor-type file, resulting GGUF hash and size.
2. Do not compare the ROCmI4 file's quality directly against BigCherry's production IQ4_XS and call the difference a kernel effect. Establish two controls on the **same ROCmI4 GGUF**: W4A4 build OFF and W4A4 build ON.
3. Keep all model weights resident on the R9700. Record device memory and confirm no host-weight/offload traffic.

If a valid source/model artifact cannot be obtained, stop QFP29. Do not implement a converter from IQ4_XS to ROCmI4.

### Phase 1 - external-fork mechanics and quality gate

On the R9700/gfx1201, build the RDNA4 fork twice from the same commit:

- control: `GGML_HIP_ROCMI4_W4A4=OFF`
- candidate: `GGML_HIP_ROCMI4_W4A4=ON`

Require an activation marker or profiler/kernel symbol proving the IU4 path executed. Run backend/op tests supplied by the fork before model benchmarking.

Quality gate on the same ROCmI4 file:

- fixed deterministic prompt corpus covering code, JSON/tool-shaped output, reasoning and long-context retrieval;
- target-only logits/KLD capture at short context and >=24K;
- report top-1 agreement, mean KLD, p99 KLD/probability delta and exact-answer regressions;
- MTP acceptance distribution must also be recorded because target-logit changes can change acceptance even when kernel time falls.

Reject W4A4 if it causes >1 fixed exact-answer regression versus ROCmI4-W4A4-off, top-1 agreement <95%, or a catastrophic tail outlier. Do not weaken the quality gate to recover a throughput result.

### Phase 2 - performance discriminator

Run ABBA/5-repetition lanes on the single R9700:

- target-only pp512/pp2048 and one long prefill lane;
- MTP decode at short, ~24K and ~80K context;
- draft depth fixed to the current production winner for the model; no adaptive-depth changes in this experiment;
- f16 KV first, q8_0 only as a capacity control;
- capture target verify batch-size histogram, accepted tokens/round, target-verify wall time, end-to-end t/s and VRAM.

Promotion threshold to justify a BigCherry port:

- >=10% reduction in target-verify wall time **and**
- >=5% median end-to-end MTP decode improvement at both short and ~24K, with no >2% regression at ~80K **and**
- quality gate above passes **and**
- no host-weight traffic / PCIe-spill confound.

If prefill improves but MTP decode does not clear the threshold, record ROCmI4 W4A4 as a prefill-only external option and close the BigCherry MTP-port branch.

### Phase 3 - only after external gate passes

Port only the minimum RDNA4 mechanism into a disposable BigCherry patch:

Likely upstream files:
- `ggml/src/ggml-cuda/mmq.cuh` - ROCmI4 tile layout, loader/vec-dot selection and RDNA4 IU4 WMMA path.
- `ggml/src/ggml-cuda/mmq.cu` / quant dispatch files - type dispatch/build wiring as required by the current pin.
- quant/type registration files required for `Q4_0_ROCMI4`.
- CMake HIP option for an opt-in `GGML_HIP_ROCMI4_W4A4` build.

Architecture gate: compile and dispatch only for gfx1201 initially. Do not enable on gfx1100 merely because it is RDNA3; the audited implementations provide no gfx1100 IU4 qualification.

Reuse existing BigCherry MTP scheduling and telemetry. Do not create a ROCmI4-private MTP controller, graph cache, placement policy, KV path or benchmark harness.

## Validation

Correctness/performance order:

1. quant artifact validation and CPU/reference dequant check;
2. fork backend/op tests, W4A4 off then on;
3. deterministic target-only quality/KLD gate;
4. target-only prefill ABBA;
5. MTP short/24K/80K ABBA with verify/acceptance telemetry;
6. only after all above pass, BigCherry patch composition tests and R9700 hardware qualification.

For any BigCherry port, also require repeated requests in one process, multi-ubatch prompts, long-context graph-allocation stability and memory-safety testing. A faster lane with reduced target work, broken SSM tensors, acceptance collapse or host spill is a failure.

## Ownership / consolidation

- QFP29 owns only ROCmI4/IU4 target-matmul qualification.
- QFP05/FMTP own MTP depth/scheduling/acceptance policy.
- 1307-1313 retain Q8_1/MMQ ownership for standard production quants.
- MET01/RPL01 own host residency/placement; QFP29 must not add a cache or offload mechanism.
- Any SYM4/tensor-class quantization-quality experiment is separate follow-up work and must not be bundled into the first W4A4 gate.

## Acceptance Criteria

- Actual gfx1201 implementation is used; no inference from gfx1151-only numbers.
- Same ROCmI4 GGUF is compared W4A4-off versus W4A4-on.
- `ssm_out.weight` is explicitly protected from ROCmI4.
- IU4 activation is proven.
- Quality/KLD and MTP acceptance are measured before throughput promotion.
- >=5% end-to-end MTP decode gain at the required contexts plus >=10% target-verify reduction is demonstrated before any BigCherry port.
- No new MTP scheduler/cache/placement mechanism is introduced.
- If the external gate fails, QFP29 terminates without a BigCherry code patch.

## Notes

The RDNA4 fork is the important new evidence: it removes the architecture-availability question for gfx1201 and makes the external A/B cheap enough to run before touching BigCherry. The R9700's x4 host link does not invalidate an all-resident ROCmI4 lane; it does invalidate any result that silently relies on host expert churn.

## Change Log

- 2026-10-05T09:37:29.282620+00:00 (created-by): Created by claude
- 2026-10-05T09:37:46.705264+00:00 (updated-by): Initial ROCmFPX assessment.
- 2026-10-06: Deep audit: inspected ROCmFPX IU4 code and independent RDNA4 port; replaced speculative gfx1100/gfx1201 investigation with a bounded gfx1201 external-fork A/B and explicit quality/termination gates.
