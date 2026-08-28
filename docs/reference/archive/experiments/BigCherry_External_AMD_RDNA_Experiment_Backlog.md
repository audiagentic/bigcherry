*Atomic ports, upstream candidates, side-fork ideas and driver/runtime experiments*

**Companion to BigCherry_Experiment_Contract_Implementation_Guide • 20 August 2026**

# 1. Purpose and scope

This companion document captures the AMD/RDNA optimization opportunities discussed outside BigCherry's current tuning-code-rebase patch set. Every item is expressed as an atomic experiment-contract candidate so an agent can port, reproduce, reject or promote it without importing a compound external branch wholesale.

- Source-reported gains are hypotheses until reproduced in BigCherry.

- Do not use PR or branch names as runtime identity; pin immutable commit SHAs in external-sources.toml before implementation.

- If an item has since landed upstream, treat it as a baseline/pin-rebase check rather than a new patch.

- If BigCherry already carries an equivalent fix, do not duplicate it; cross-link the existing patch.

- Every candidate must be tested on intended signatures, boundaries, controls and regressions. The goal is to discover the safe selection envelope, not merely to demonstrate one fast benchmark.

# 2. Standard experiment contract template

| Field               | Requirement                                                                          |
|---------------------|--------------------------------------------------------------------------------------|
| Hypothesis          | One falsifiable performance/correctness claim.                                       |
| Source identity     | Repository, PR/discussion/issue, immutable commit(s) once ported.                    |
| Prerequisites       | Other atomic experiments or baseline features that must exist first.                 |
| Target backend/arch | HIP/Vulkan; gfx1100/gfx1151/gfx1201 etc.                                             |
| Target op/signature | Exact op family plus captured canonical shapes/metadata.                             |
| Positive models     | Models expected to exercise the target.                                              |
| Controls            | Models/paths that should not trigger or regress.                                     |
| Boundary sweep      | The dimension that decides where it wins/loses.                                      |
| Correctness         | Backend-op, greedy, PPL/KLD, recurrent/cache, MTP identity as applicable.            |
| Performance         | Kernel time plus PP/TG/MTP/agent-level metrics.                                      |
| Regression budget   | Default: no \>1% unexplained non-target regression; stricter for correctness work.   |
| Promotion result    | Promote globally, promote conditionally with selector, keep experimental, or reject. |

# 3. AMD-Ecosystem / ROCm downstream atomic backlog

The AMD downstream discussions describe a larger gfx11 program. The entries below split compound downstream PRs into independently testable BigCherry hypotheses.

## AMD-MMQ-001 — RDNA MMQ tile-width reduction for LDS occupancy

| Class          | AMD-Ecosystem / MMQ                                |
|----------------|----------------------------------------------------|
| Backend / arch | HIP; validate gfx1100, gfx1151, gfx1201 separately |
| Source         | AMD PR Set 2 / downstream \#32                     |
| Source status  | Recheck before port                                |
| Prerequisites  | None                                               |
| Q8_0 relevance | High                                               |

Source link: [<u>AMD PR Set 2 / downstream \#32</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

Reducing MMQ X tile width for LDS-limited shapes increases resident workgroups and improves prefill throughput.

### What / why

Wide tiles can reduce scheduling overhead but consume enough LDS/VGPRs to collapse occupancy. AMD's gfx11 work explicitly retunes MMQ geometry rather than assuming CUDA-derived values transfer to RDNA.

### Where to change

ggml-cuda MMQ config/launch selection. Port only the tile-selection change, not the whole downstream \#32 bundle.

### Trigger / positive cases

- Capture high-cost dense and MoE MMQ signatures from Qwen3.6-27B and Qwen3.6-35B-A3B.

- Test gfx1100 and gfx1201 independently; gfx1151 if hardware becomes available.

### Controls / hostile cases

- Small matrices already achieving high occupancy.

- Quant types not targeted by the geometry change.

- Non-MMQ family winners to ensure routing still selects BLAS/MMVQ where appropriate.

### Boundary sweep

- mmq_x candidates around native winner; physical M/ubatch 64,128,256,512,1024,2048,4096.

- Record LDS/workgroup, VGPRs, waves/SIMD and kernel occupancy.

### Correctness gate

Backend-op tensor parity and temp-0 model output. Geometry-only changes should be bit-identical where arithmetic order is unchanged.

### Performance evidence to collect

Kernel median/MAD, pp128/512/1024/4096, long-prompt PP; time×calls ranking.

### Acceptance / regression rule

Promote only as an arch/signature selector if \>3% repeatable kernel gain or \>1% end-to-end gain with \<=1% non-target regression.

### Implementation notes

## AMD-Q8-001 — Tiny-M Q8_0 MMQ specialization

| Class          | AMD-Ecosystem / Q8             |
|----------------|--------------------------------|
| Backend / arch | HIP; gfx1100/gfx1201 primary   |
| Source         | AMD PR Set 2 / downstream \#32 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | Very high                      |

Source link: [<u>AMD PR Set 2 / downstream \#32</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

A Q8_0 specialization for very small physical M beats both ordinary MMQ and MMVQ for gate projections and speculative-verify batches.

### What / why

Tiny-M operations pay disproportionate launch/tile overhead and occur in MTP verification and small gate projections. AMD \#32 explicitly carries a tiny-M Q8 gate path.

### Where to change

Q8_0 MMQ dispatch/kernel specialization.

### Trigger / positive cases

- Qwen3.6-27B MTP Q8_0.

- Qwen3.6-35B-A3B Q8_0.

- Captured Q8 gate/output signatures with physical M \<=64.

### Controls / hostile cases

- Llama 3.x dense Q8 as non-Qwen dense control.

- Q4_K/Q6_K equivalent shapes.

- Q8 signatures with M\>=128.

### Boundary sweep

- M=1,2,3,4,8,16,24,32,48,64,128,256,512; use real captured N/K.

- Force MMVQ, native MMQ, specialized path.

### Correctness gate

Backend reference + temp-0 output; MTP speculative-vs-nonspeculative bit identity where applicable.

### Performance evidence to collect

Kernel us; tg128/tg512; MTP depth 2/3/5/8; context 0/32K/128K.

### Acceptance / regression rule

Derive explicit arch+M(+shape) winning envelope. No blanket Q8 override.

### Implementation notes

This should be one of the first external candidates BigCherry backfills into an Experiment Contract.

## AMD-MOE-001 — MoE-aware MMQ tile sizing from average expert occupancy

| Class          | AMD-Ecosystem / MoE MMQ                       |
|----------------|-----------------------------------------------|
| Backend / arch | HIP; gfx1100/gfx1201; gfx1151 source evidence |
| Source         | AMD PR Set 2 / downstream \#39                |
| Source status  | Recheck before port                           |
| Prerequisites  | None                                          |
| Q8_0 relevance | High                                          |

Source link: [<u>AMD PR Set 2 / downstream \#39</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

Selecting MMQ tile width from expected/average tokens per expert reduces empty work and improves MoE prefill versus worst-case ncols_max sizing.

### What / why

Real routing rarely sends every token to one expert. Worst-case sizing selects tiles too wide for typical expert occupancy. Source discussion reports large pp128/pp1024 gains on Qwen3.6-35B-A3B.

### Where to change

MUL_MAT_ID MMQ host-side tile selector.

### Trigger / positive cases

- Qwen3.6-35B-A3B Q8_0 and Q4_K_M.

- Natural routing captures plus synthetic routing generated from the same expert dimensions.

### Controls / hostile cases

- Dense Qwen3.6-27B.

- Highly skewed single-expert routing where mean-based sizing can become wrong.

- Alternative MoE family such as Gemma/DeepSeek if available.

### Boundary sweep

- ubatch 128,256,512,1024,2048,4096.

- Routing: uniform, captured, mild skew, Zipf, concentrated, single-hot. Record mean/p95/max tokens/expert.

### Correctness gate

MUL_MAT_ID backend parity and model temp-0/PPL.

### Performance evidence to collect

Selected tile, useful/empty workgroups, kernel time, PP.

### Acceptance / regression rule

A mean-based rule must include a safe skew fallback if hostile routing regresses \>1%. Let generalise.py discover a compact selector.

### Implementation notes

Do not hard-code '2×mean' unless boundary evidence supports it on each architecture.

## AMD-MOE-002 — GPU compact MoE MMQ block-map construction

| Class          | AMD-Ecosystem / MoE scheduling |
|----------------|--------------------------------|
| Backend / arch | HIP; gfx11+                    |
| Source         | AMD PR Set 2 / downstream \#63 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 2 / downstream \#63</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

Building an exact expert block map on GPU is cheap enough to enable compact launches without host synchronization.

### What / why

This is the correctness/overhead prerequisite to eliminating empty expert workgroups. It must be proven separately from the launch change.

### Where to change

New device-side prefix/map kernel producing block_start/block_expert from expert token counts.

### Trigger / positive cases

- Synthetic routing matrices for real Qwen expert counts.

- Natural captured routing at multiple ubatches.

### Controls / hostile cases

- All experts empty except one; all experts equally full; maximum-skew routing.

- Very small batches where map overhead may exceed savings.

### Boundary sweep

- Tokens 1..4096; varying experts_used; distributions uniform/Zipf/single.

- Measure map build time independently.

### Correctness gate

GPU map must exactly match CPU reference for every distribution.

### Performance evidence to collect

Map-build us, synchronization count, temporary bytes.

### Acceptance / regression rule

Keep only if exact and map overhead is small relative to saved MMQ work; no host readback in steady state.

### Implementation notes

Prerequisite for AMD-MOE-003.

## AMD-MOE-003 — Compact per-expert MMQ launch grid

| Class          | AMD-Ecosystem / MoE scheduling |
|----------------|--------------------------------|
| Backend / arch | HIP; gfx11+                    |
| Source         | AMD PR Set 2 / downstream \#63 |
| Source status  | Recheck before port            |
| Prerequisites  | AMD-MOE-002                    |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 2 / downstream \#63</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

Launching only actual per-expert column blocks reduces wasted workgroups and improves MoE PP.

### What / why

Rectangular expert grids launch blocks for empty/lightly-loaded experts that immediately exit. Compact enumeration turns routing sparsity into real launch reduction.

### Where to change

MUL_MAT_ID MMQ grid enumeration.

### Trigger / positive cases

- Qwen3.6-35B-A3B at pp128..pp4096.

- Captured routing with many partially filled experts.

### Controls / hostile cases

- Perfectly dense/uniform expert routing.

- Tiny batches where map/indirection overhead could dominate.

### Boundary sweep

- Same routing distribution matrix as AMD-MOE-001/002.

- Count launched vs useful blocks.

### Correctness gate

Exact output parity.

### Performance evidence to collect

Launched blocks, empty-block fraction, kernel us, end-to-end PP.

### Acceptance / regression rule

Promote conditionally if compact launch saves \>=2% PP on target routing and is neutral (\<1% loss) on dense distributions.

### Implementation notes

## AMD-MMV-001 — Decode matvec without Q8_1 activation quantization

| Class          | AMD-Ecosystem / decode matvec                             |
|----------------|-----------------------------------------------------------|
| Backend / arch | HIP; source default was RDNA3.5, validate gfx1100/gfx1201 |
| Source         | AMD PR Set 2 matvec item                                  |
| Source status  | Recheck before port                                       |
| Prerequisites  | None                                                      |
| Q8_0 relevance | Medium                                                    |

Source link: [<u>AMD PR Set 2 matvec item</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

For n=1 bandwidth-bound decode, skipping the q8_1 activation quantization pass and dequantizing weights directly can reduce overhead.

### What / why

Activation quantization is pure overhead when matrix-vector decode is dominated by weight reads. AMD's downstream set includes this path and a fused gate+up SwiGLU variant.

### Where to change

Single-column quantized matvec path.

### Trigger / positive cases

- Qwen3.6 dense Q4/Q6/Q8 decode signatures.

- Small and medium dense models to expose launch overhead.

### Controls / hostile cases

- Prefill/small-batch n\>1.

- Large MoE expert matmuls.

- Accuracy-sensitive quants.

### Boundary sweep

- ncols_dst 1,2,3,4,8; model sizes 4B,9B,27B; context 0/64K/128K.

### Correctness gate

Backend parity and PPL; arithmetic changes may require tolerance rather than byte identity.

### Performance evidence to collect

tg128/512, kernel count, activation-quantize time removed.

### Acceptance / regression rule

Only enable for n=1 (or proven narrow range) if E2E decode gain \>1% and quality gate passes.

### Implementation notes

## AMD-MMQ-002 — Dedicated RDNA3.5 MMQ device table

| Class          | AMD-Ecosystem / hardware table |
|----------------|--------------------------------|
| Backend / arch | HIP; gfx1151 only              |
| Source         | AMD PR Set 2 / downstream \#25 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 2 / downstream \#25</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

gfx1151 needs a distinct MMQ tile/warp table rather than inheriting another RDNA generation.

### What / why

AMD reports mmq_y 128-\>64 and warps 8-\>4 producing significant prefill gains across many models on gfx1151.

### Where to change

MMQ parameter table selection.

### Trigger / positive cases

- gfx1151 only; dense + MoE Qwen corpus.

### Controls / hostile cases

- gfx1100/gfx1201 must prove non-selection.

- Decode should remain neutral.

### Boundary sweep

- pp128/512/1024/4096 across Q4/Q6/Q8; candidate table entries independently if possible.

### Correctness gate

PPL/temp-0 parity.

### Performance evidence to collect

PP/TG and resource stats.

### Acceptance / regression rule

No promotion without gfx1151 hardware evidence. Keep hardware-scoped.

### Implementation notes

## AMD-TEST-001 — Controlled MoE routing benchmark generator

| Class          | AMD-Ecosystem / test infrastructure    |
|----------------|----------------------------------------|
| Backend / arch | Backend-neutral harness around HIP MMQ |
| Source         | AMD PR Set 2 / downstream \#62         |
| Source status  | Recheck before port                    |
| Prerequisites  | None                                   |
| Q8_0 relevance | High                                   |

Source link: [<u>AMD PR Set 2 / downstream \#62</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

### Hypothesis

A synthetic routing generator makes MoE dispatch heuristics reproducible and provides hostile cases real models may not naturally generate.

### What / why

AMD \#62 already defines uniform/single/Zipf/concentration routing modes; this matches BigCherry's need for positive and hostile routing contracts.

### Where to change

Testing/harness, preferably BigCherry tooling rather than permanent runtime patch.

### Trigger / positive cases

- Use real expert dimensions from Qwen3.6-35B-A3B.

### Controls / hostile cases

- Validate synthetic output against equivalent CPU/reference routing.

### Boundary sweep

- Uniform, single-hot, Zipf, configurable concentration; token counts 1..4096.

### Correctness gate

Generated expert assignment statistics and MUL_MAT_ID output must be deterministic.

### Performance evidence to collect

Roofline/utilization and exact routing statistics.

### Acceptance / regression rule

Adopt as reusable test family infrastructure if deterministic and easy to integrate; no production runtime impact.

### Implementation notes

## AMD-GEMM-001 — 128-byte row padding for cache-set aliasing

| Class          | AMD-Ecosystem / prefill memory layout                 |
|----------------|-------------------------------------------------------|
| Backend / arch | HIP; validate gfx1100/gfx1201                         |
| Source         | AMD PR Set 5 / downstream \#57                        |
| Source status  | Recheck before port                                   |
| Prerequisites  | None                                                  |
| Q8_0 relevance | Low direct; high if combined with dequantized shadows |

Source link: [<u>AMD PR Set 5 / downstream \#57</u>](https://github.com/ggml-org/llama.cpp/discussions/26379)

### Hypothesis

Adding one cache line when a float-weight row stride is a multiple of 2048B breaks LLC cache-set aliasing and improves GEMM prefill.

### What / why

AMD split this originally as \#34 and later folded it into \#57. This is independent from the F16-shadow optimization and must be tested separately.

### Where to change

Weight allocation/layout; consumers must honor nb\[1\].

### Trigger / positive cases

- Captured float/BF16/F16 GEMM tensors with aliasing row bytes.

- Qwen dense projection shapes known to hit 2048B stride multiples.

### Controls / hostile cases

- Non-aliasing strides.

- Quantized representations whose packed rows are not affected.

### Boundary sweep

- Padding 0,64,128,256 bytes where legal; group by row_bytes mod cache geometry.

### Correctness gate

Tensor/layout correctness and model parity.

### Performance evidence to collect

Per-op GEMM time, cache counters, PP.

### Acceptance / regression rule

Enable only on proven alias classes/architectures. No unconditional padding if neutral or memory-costly.

### Implementation notes

## AMD-GEMM-002 — Persistent F16 shadow of quantized dense weights

| Class          | AMD-Ecosystem / prefill representation |
|----------------|----------------------------------------|
| Backend / arch | HIP; gfx1100/gfx1201                   |
| Source         | AMD PR Set 5 / downstream \#57         |
| Source status  | Recheck before port                    |
| Prerequisites  | None                                   |
| Q8_0 relevance | Very high                              |

Source link: [<u>AMD PR Set 5 / downstream \#57</u>](https://github.com/ggml-org/llama.cpp/discussions/26379)

### Hypothesis

For large-batch prefill, paying a one-time Q-\>F16 expansion at load can beat repeated quantized MMQ execution.

### What / why

This trades VRAM for faster dense GEMM. Q8 is especially interesting because expansion ratio is only ~2× rather than ~4× from Q4.

### Where to change

Model load/device buffer path; create F16 shadow only for eligible dense weights.

### Trigger / positive cases

- Qwen3.6-27B Q8_0 primary.

- Q4_K/Q6_K as controls/alternative economics.

### Controls / hostile cases

- Decode-only workloads.

- VRAM-constrained configurations.

- MoE expert weights if shadowing them would explode memory.

### Boundary sweep

- Shadow selected tensors only vs all eligible dense tensors; M/ubatch 64..4096; track VRAM.

### Correctness gate

Shadow contents vs reference dequant; model output/PPL.

### Performance evidence to collect

Model-load overhead, VRAM delta, PP, TG neutrality.

### Acceptance / regression rule

Keep only if there is a clear target workload where PP gain justifies memory. Never make global default.

### Implementation notes

## AMD-GEMM-003 — K-pad F16 shadow to avoid aliasing

| Class          | AMD-Ecosystem / prefill representation |
|----------------|----------------------------------------|
| Backend / arch | HIP                                    |
| Source         | AMD PR Set 5 / downstream \#57         |
| Source status  | Recheck before port                    |
| Prerequisites  | AMD-GEMM-002                           |
| Q8_0 relevance | High                                   |

Source link: [<u>AMD PR Set 5 / downstream \#57</u>](https://github.com/ggml-org/llama.cpp/discussions/26379)

### Hypothesis

Padding the F16 shadow's K dimension fixes aliasing shapes even when the original quantized storage was not aliased.

### What / why

The shadow has its own row stride and cache behavior, so its padding must be decided independently from the original weight.

### Where to change

F16-shadow allocation/stride.

### Trigger / positive cases

- Captured down_proj and other shapes whose F16 row bytes alias.

### Controls / hostile cases

- Non-aliasing shadow strides.

### Boundary sweep

- K padding 0/one-cache-line and affected dimensions.

### Correctness gate

GEMM output equivalence.

### Performance evidence to collect

GEMM/PP and memory overhead.

### Acceptance / regression rule

Condition on aliasing stride only.

### Implementation notes

## AMD-GEMM-004 — Large-M F16 shadow -\> tuned hipBLASLt crossover

| Class          | AMD-Ecosystem / dispatch            |
|----------------|-------------------------------------|
| Backend / arch | HIP; gfx1100/gfx1201                |
| Source         | AMD PR Set 5 / downstream \#57      |
| Source status  | Recheck before port                 |
| Prerequisites  | AMD-GEMM-002; AMD-GEMM-003 optional |
| Q8_0 relevance | Very high                           |

Source link: [<u>AMD PR Set 5 / downstream \#57</u>](https://github.com/ggml-org/llama.cpp/discussions/26379)

### Hypothesis

Above an architecture/shape-specific physical M, tuned hipBLASLt on a persistent F16 shadow beats native quantized MMQ.

### What / why

AMD's downstream \#57 uses hipBLAS for sufficiently large batches. BigCherry should derive the crossover from measurements rather than inherit a fixed threshold.

### Where to change

Dense MUL_MAT dispatch between native Q MMQ and F16-shadow BLAS.

### Trigger / positive cases

- Qwen3.6-27B Q8_0: every major dense projection signature.

- Q4/Q6 secondary.

### Controls / hostile cases

- M\<=128, decode, memory-constrained lane.

### Boundary sweep

- M=64,128,192,256,384,512,768,1024,2048,4096; compare MMQ, hipBLASLt default, hipBLASLt tuned.

### Correctness gate

Exact/tolerant output and PPL.

### Performance evidence to collect

Kernel + end-to-end PP, VRAM, load time.

### Acceptance / regression rule

Derive per-arch/per-quant crossover. If tuned MMQ always wins for Q8, reject shadow dispatch while retaining evidence.

### Implementation notes

## AMD-STREAM-001 — Honor active HIP stream in non-split matmul

| Class          | AMD-Ecosystem / concurrency correctness |
|----------------|-----------------------------------------|
| Backend / arch | HIP                                     |
| Source         | AMD PR Set 6 / downstream \#36          |
| Source status  | Recheck before port                     |
| Prerequisites  | None                                    |
| Q8_0 relevance | High                                    |

Source link: [<u>AMD PR Set 6 / downstream \#36</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

A matmul scheduled to an auxiliary stream must actually execute on that stream; otherwise concurrency experiments are invalid.

### What / why

AMD \#36 carries this as a correctness fix beneath shared-expert concurrency.

### Where to change

ggml_cuda_op_mul_mat stream selection.

### Trigger / positive cases

- Force independent matmuls to aux stream and inspect timeline.

### Controls / hostile cases

- Normal single-stream decode/prefill.

### Boundary sweep

- Main vs aux stream; graph opt on/off.

### Correctness gate

Output equality and profiler proof of actual stream execution.

### Performance evidence to collect

No required gain; \<=0.5% single-stream regression.

### Acceptance / regression rule

Prerequisite correctness patch if missing upstream/current pin.

### Implementation notes

## AMD-STREAM-002 — Per-(device,stream) BLAS handles

| Class          | AMD-Ecosystem / concurrency correctness |
|----------------|-----------------------------------------|
| Backend / arch | HIP                                     |
| Source         | AMD PR Set 6 / downstream \#36          |
| Source status  | Recheck before port                     |
| Prerequisites  | AMD-STREAM-001                          |
| Q8_0 relevance | Medium                                  |

Source link: [<u>AMD PR Set 6 / downstream \#36</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Concurrent BLAS work requires independent handle stream/workspace state.

### What / why

One mutable handle per device can race or serialize when separate branches run concurrently.

### Where to change

HIP/cuBLAS compatibility backend context.

### Trigger / positive cases

- Two concurrent BLAS GEMMs on separate streams.

### Controls / hostile cases

- Single-stream path.

### Boundary sweep

- 1/2 aux streams, repeated thousands of iterations.

### Correctness gate

Deterministic tensor output; no races/errors.

### Performance evidence to collect

Single-stream overhead and concurrent overlap.

### Acceptance / regression rule

Prerequisite if concurrency needs BLAS. No promotion if it slows ordinary path materially.

### Implementation notes

## AMD-STREAM-003 — Dedicated scratch for concurrent branches

| Class          | AMD-Ecosystem / concurrency correctness |
|----------------|-----------------------------------------|
| Backend / arch | HIP                                     |
| Source         | AMD PR Set 6 / downstream \#36          |
| Source status  | Recheck before port                     |
| Prerequisites  | AMD-STREAM-001                          |
| Q8_0 relevance | Backend-wide                            |

Source link: [<u>AMD PR Set 6 / downstream \#36</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Scratch lifetimes must be isolated across concurrently executing graph branches.

### What / why

Allocator reuse that is safe sequentially can alias temporary buffers when branches overlap.

### Where to change

Graph optimizer/concurrent buffer allocation.

### Trigger / positive cases

- Shared-expert diamond and other independent branches.

### Controls / hostile cases

- Sequential graph execution.

### Boundary sweep

- Stress repeated concurrent graph runs with varying shapes.

### Correctness gate

Bit-identical output; no memory corruption.

### Performance evidence to collect

Memory overhead and any allocation cost.

### Acceptance / regression rule

Correctness prerequisite; optimize memory only after safety proven.

### Implementation notes

## AMD-STREAM-004 — Overlap MoE shared expert on auxiliary stream

| Class          | AMD-Ecosystem / MoE concurrency                  |
|----------------|--------------------------------------------------|
| Backend / arch | HIP; validate gfx1100/gfx1201                    |
| Source         | AMD PR Set 6 / downstream \#36                   |
| Source status  | Recheck before port                              |
| Prerequisites  | AMD-STREAM-001 + AMD-STREAM-002 + AMD-STREAM-003 |
| Q8_0 relevance | High                                             |

Source link: [<u>AMD PR Set 6 / downstream \#36</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Shared-expert compute can overlap routed-expert compute during decode and fill otherwise idle GPU resources.

### What / why

Source reports +7.4% tg128; this attacks scheduling/underutilization rather than arithmetic.

### Where to change

Graph optimizer shared-expert diamond scheduling.

### Trigger / positive cases

- Qwen MoE with gated shared expert, decode n=1.

- Q8 and Q4 variants.

### Controls / hostile cases

- Prefill where both branches may already saturate GPU.

- MoE without shared expert.

- Dense models.

### Boundary sweep

- tg128/512; context 0/64K; graph opt on/off.

### Correctness gate

Byte-identical output; profiler verifies branch overlap and proper join.

### Performance evidence to collect

Overlap %, kernel concurrency, TG, power/utilization.

### Acceptance / regression rule

Promote only when workload predicate guarantees independent branches and E2E gain \>1%.

### Implementation notes

## AMD-STREAM-005 — Protect concurrent-region join node from fusion

| Class          | AMD-Ecosystem / graph correctness |
|----------------|-----------------------------------|
| Backend / arch | HIP                               |
| Source         | AMD PR Set 4 / downstream \#71    |
| Source status  | Recheck before port               |
| Prerequisites  | AMD-STREAM-004                    |
| Q8_0 relevance | High                              |

Source link: [<u>AMD PR Set 4 / downstream \#71</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

### Hypothesis

The shared-expert join node must not be consumed by unrelated op fusion while graph concurrency is active.

### What / why

AMD reports capture abort when fusion removes the join and the aux stream never rejoins.

### Where to change

Graph fusion eligibility around concurrent region join.

### Trigger / positive cases

- MoE decode with graph-opt + shared-expert overlap.

### Controls / hostile cases

- Same model graph-opt off; dense models.

### Boundary sweep

- Repeated graph capture/replay.

### Correctness gate

No capture abort; output parity.

### Performance evidence to collect

No material regression; enables safe concurrency.

### Acceptance / regression rule

Mandatory prerequisite before enabling graph concurrency by default.

### Implementation notes

## AMD-STREAM-006 — Enable graph-opt by default on RDNA3.5

| Class          | AMD-Ecosystem / default policy  |
|----------------|---------------------------------|
| Backend / arch | HIP; gfx1151 only initially     |
| Source         | AMD PR Set 6 / downstream \#56  |
| Source status  | Recheck before port             |
| Prerequisites  | AMD-STREAM-004 + AMD-STREAM-005 |
| Q8_0 relevance | High                            |

Source link: [<u>AMD PR Set 6 / downstream \#56</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Once concurrency correctness prerequisites settle, graph-opt can be default-on for gfx1151 without regressions.

### What / why

This is a policy/default flip, not a kernel optimization. AMD explicitly recommends landing it last.

### Where to change

Architecture default configuration.

### Trigger / positive cases

- Broad gfx1151 model regression suite.

### Controls / hostile cases

- gfx1100/gfx1201 unaffected by default.

- Long-context graph edge cases.

### Boundary sweep

- Dense/MoE/GDN; MTP on/off; graph capture stress.

### Correctness gate

No capture failures or output divergence.

### Performance evidence to collect

Broad regression distribution, not one model.

### Acceptance / regression rule

Only after hardware-wide confidence; otherwise keep opt-in.

### Implementation notes

## AMD-FUS-001 — Fuse GEMV epilogue activation

| Class          | AMD-Ecosystem / decode fusion  |
|----------------|--------------------------------|
| Backend / arch | HIP                            |
| Source         | AMD PR Set 4 / downstream \#67 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 4 / downstream \#67</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

### Hypothesis

Applying SILU/SIGMOID in the GEMV epilogue removes a launch and memory round-trip.

### What / why

Decode is launch-heavy, especially small models. AMD reports the combined \#67 fusion set reduces launches/token and yields modest TG uplift.

### Where to change

mul_mat_vec_f/q epilogue.

### Trigger / positive cases

- Qwen dense and hybrid decode paths with non-gated activation after GEMV.

### Controls / hostile cases

- Prefill; graphs without matching activation.

### Boundary sweep

- Q8/Q4/Q6; model sizes 4B/9B/27B.

### Correctness gate

Bit/tolerance parity.

### Performance evidence to collect

Launch count, HBM traffic, TG.

### Acceptance / regression rule

Promote if pattern-matched only and non-target paths untouched.

### Implementation notes

## AMD-FUS-002 — Fuse GEMV activation + elementwise MUL

| Class          | AMD-Ecosystem / decode fusion  |
|----------------|--------------------------------|
| Backend / arch | HIP                            |
| Source         | AMD PR Set 4 / downstream \#67 |
| Source status  | Recheck before port            |
| Prerequisites  | AMD-FUS-001                    |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 4 / downstream \#67</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

### Hypothesis

Folding the post-activation multiply into GEMV epilogue removes another launch/memory pass.

### What / why

Extends the activation fusion to common tails.

### Where to change

GEMV epilogue.

### Trigger / positive cases

- Graphs with GEMV -\> SILU -\> MUL.

### Controls / hostile cases

- Other MUL broadcasting/layouts.

### Boundary sweep

- Different tensor shapes/broadcast forms.

### Correctness gate

Backend parity.

### Performance evidence to collect

Launch count and TG.

### Acceptance / regression rule

Pattern must prove exact supported layout; otherwise decline.

### Implementation notes

## AMD-FUS-003 — Fuse GEMV -\> view -\> residual ADD

| Class          | AMD-Ecosystem / decode fusion  |
|----------------|--------------------------------|
| Backend / arch | HIP                            |
| Source         | AMD PR Set 4 / downstream \#67 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 4 / downstream \#67</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

### Hypothesis

Writing GEMV output directly into residual-add destination through a reshape/view removes a frequent decode kernel.

### What / why

Existing fusion misses layouts where a view interrupts matmul and residual add.

### Where to change

GEMV fusion matcher/epilogue.

### Trigger / positive cases

- Hybrid/GDN model decode graph patterns.

### Controls / hostile cases

- Different view strides; non-contiguous aliasing; standard transformer paths.

### Boundary sweep

- Relevant view shapes from captured graphs.

### Correctness gate

Exact output and aliasing safety.

### Performance evidence to collect

Launch count/TG.

### Acceptance / regression rule

Only supported view semantics; reject if any alias ambiguity.

### Implementation notes

Related conceptually to BigCherry RD13 but this is GEMV epilogue fusion; avoid duplicate implementation.

## AMD-KVPROJ-001 — Fuse attention K and V projection into one MMVQ dispatch

| Class          | AMD-Ecosystem / decode fusion  |
|----------------|--------------------------------|
| Backend / arch | HIP; source gfx1151            |
| Source         | AMD PR Set 4 / downstream \#59 |
| Source status  | Recheck before port            |
| Prerequisites  | None                           |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 4 / downstream \#59</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

### Hypothesis

Concatenating K/V weights at load and issuing one MMVQ improves occupancy on small models.

### What / why

Source reports +1–2% on small models, flat on large ones. This is a model-load representation change plus graph simplification.

### Where to change

Model load tensor concatenation + graph projection emission.

### Trigger / positive cases

- Small Qwen models 0.5B–4B on gfx1151/gfx1201.

- Attention layers with compatible K/V shapes/types.

### Controls / hostile cases

- 27B+ where FFN dominates.

- Models with incompatible K/V layout or tied metadata.

### Boundary sweep

- Model size and head dimensions.

### Correctness gate

K and V outputs individually identical to two-matmul baseline.

### Performance evidence to collect

MMVQ occupancy, launch count, TG.

### Acceptance / regression rule

Conditional small-model optimization only if memory/layout complexity remains low.

### Implementation notes

## AMD-SSM-001 — Channels-major SSM_CONV input mode

| Class          | AMD-Ecosystem / shared ggml layout |
|----------------|------------------------------------|
| Backend / arch | CPU+HIP support required           |
| Source         | AMD PR Set 6 / downstream \#52     |
| Source status  | Recheck before port                |
| Prerequisites  | None                               |
| Q8_0 relevance | High                               |

Source link: [<u>AMD PR Set 6 / downstream \#52</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Allowing SSM convolution to consume channels-major input eliminates a per-layer physical transpose in DeltaNet models.

### What / why

AMD reports -29% total prefill at ub4096 in the favorable case. This changes shared ggml semantics, so backend fallback/correctness must be strong.

### Where to change

ggml_ssm_conv API/op and CPU/HIP implementations.

### Trigger / positive cases

- Qwen3.6 hybrid/GDN model prefill.

### Controls / hostile cases

- Existing time-major callers must remain bit-identical.

- Backends without implementation must cleanly fall back.

### Boundary sweep

- ubatch 256/512/1024/2048/4096; sequence lengths.

### Correctness gate

CPU reference equality for both layouts; model PPL/output.

### Performance evidence to collect

Removed transpose/copy time, total PP.

### Acceptance / regression rule

Adopt layout mode only if API remains backward compatible and at least HIP+CPU are correct.

### Implementation notes

## AMD-GDN-001 — Chunked fused GatedDeltaNet recurrence

| Class          | AMD-Ecosystem / GDN kernel                    |
|----------------|-----------------------------------------------|
| Backend / arch | HIP; source gfx1151, validate gfx1100/gfx1201 |
| Source         | AMD PR Set 6 / downstream \#54                |
| Source status  | Recheck before port                           |
| Prerequisites  | None                                          |
| Q8_0 relevance | High                                          |

Source link: [<u>AMD PR Set 6 / downstream \#54</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Chunking the delta-rule recurrence and retaining state in registers/LDS drastically reduces token-by-token GDN overhead.

### What / why

Source reports 1.89x GDN-op improvement and up to +20% end-to-end at ub4096. Restriction set is narrow, so eligibility must be explicit.

### Where to change

GATED_DELTA_NET HIP kernel.

### Trigger / positive cases

- Qwen hybrid models satisfying scalar gate, K==1, S_v==128.

- Prefill at ubatch 512..4096.

### Controls / hostile cases

- Different S_v, K, gate mode; decode n=1; standard attention models.

### Boundary sweep

- Chunk size, ubatch, context length, architecture.

### Correctness gate

Recurrent state + output vs old kernel over long sequences; PPL.

### Performance evidence to collect

GDN op time, PP, workspace, VGPR/LDS.

### Acceptance / regression rule

Promote only under exact supported shape predicate; fallback untouched.

### Implementation notes

## AMD-GDN-002 — DPP row-shift reduction inside chunked GDN

| Class          | AMD-Ecosystem / RDNA ISA       |
|----------------|--------------------------------|
| Backend / arch | HIP RDNA                       |
| Source         | AMD PR Set 6 / downstream \#54 |
| Source status  | Recheck before port            |
| Prerequisites  | AMD-GDN-001                    |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 6 / downstream \#54</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

RDNA DPP lane operations beat shuffle/DS reductions in the chunked GDN kernel.

### What / why

Avoids competing with LDS/DS traffic and is a clean architecture-specific micro-optimization.

### Where to change

Reduction primitive within AMD-GDN-001.

### Trigger / positive cases

- Exact GDN target shapes on gfx1100/gfx1151/gfx1201.

### Controls / hostile cases

- Portable fallback on non-RDNA.

### Boundary sweep

- DPP vs shuffle only; same launch geometry.

### Correctness gate

Tensor/state parity.

### Performance evidence to collect

DS utilization, kernel us.

### Acceptance / regression rule

Architecture-specific selector if \>2% kernel gain.

### Implementation notes

## AMD-GDN-003 — Native exp2 decay in chunked GDN

| Class          | AMD-Ecosystem / RDNA math      |
|----------------|--------------------------------|
| Backend / arch | HIP RDNA                       |
| Source         | AMD PR Set 6 / downstream \#54 |
| Source status  | Recheck before port            |
| Prerequisites  | AMD-GDN-001                    |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 6 / downstream \#54</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Base-2 decay formulation using native AMD exp2 lowers scalar math overhead without changing meaningful recurrent behavior.

### What / why

Source kernel uses \_\_builtin_amdgcn_exp2f to avoid heavier expf/OCML path.

### Where to change

GDN decay calculation.

### Trigger / positive cases

- Long prefill sequences with decay range coverage.

### Controls / hostile cases

- Numerically sensitive short sequences and edge decay values.

### Boundary sweep

- Input decay distribution, context length.

### Correctness gate

State/logit tolerance + PPL; exactness may not be expected.

### Performance evidence to collect

Kernel instruction profile and PP.

### Acceptance / regression rule

Only if quality metrics stay within defined tolerance and kernel win survives E2E.

### Implementation notes

## AMD-GDN-004 — Launch-bounds / VGPR occupancy tuning for chunked GDN

| Class          | AMD-Ecosystem / occupancy      |
|----------------|--------------------------------|
| Backend / arch | HIP; per-arch                  |
| Source         | AMD PR Set 6 / downstream \#54 |
| Source status  | Recheck before port            |
| Prerequisites  | AMD-GDN-001                    |
| Q8_0 relevance | High                           |

Source link: [<u>AMD PR Set 6 / downstream \#54</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

### Hypothesis

Explicit register/occupancy constraints keep enough GDN blocks resident and avoid spills.

### What / why

Register-resident recurrent state can otherwise collapse occupancy. Source chose a gfx1151-specific target; other RDNA generations need independent tuning.

### Where to change

GDN kernel launch bounds.

### Trigger / positive cases

- Target GDN signatures.

### Controls / hostile cases

- All non-target architectures.

### Boundary sweep

- Occupancy target/launch-bounds; inspect VGPRs/spills.

### Correctness gate

Unchanged output.

### Performance evidence to collect

Kernel us, occupancy, spills, PP.

### Acceptance / regression rule

Arch-specific winner only.

### Implementation notes

# 4. Upstream llama.cpp candidates and baselines

## UP-HIP-001 — Dynamic MMVQ warp count for narrow MoE matrices

| Class          | Upstream llama.cpp candidate                                  |
|----------------|---------------------------------------------------------------|
| Backend / arch | HIP; gfx1201 strongest source evidence, gfx1100 also positive |
| Source         | llama.cpp PR \#20831                                          |
| Source status  | Recheck before port                                           |
| Prerequisites  | None                                                          |
| Q8_0 relevance | Very high                                                     |

Source link: [<u>llama.cpp PR \#20831</u>](https://github.com/ggml-org/llama.cpp/pull/20831)

### Hypothesis

Clamping MMVQ warps based on matrix width avoids idle warps/synchronization on narrow expert matrices.

### What / why

Source updates reported ~9–10% R9700 MoE decode gains and smaller positive gfx1100 gains. This is a dispatch/launch-policy optimization.

### Where to change

MMVQ nwarps selection.

### Trigger / positive cases

- Qwen MoE Q4/Q5/Q6/Q8 decode on R9700 and XTX.

### Controls / hostile cases

- Wide dense matrices where 8 warps maximize bandwidth.

- Prefill.

### Boundary sweep

- Matrix width, nwarps 1/2/4/8; tg128/512.

### Correctness gate

Backend parity.

### Performance evidence to collect

Warp utilization, kernel us, TG.

### Acceptance / regression rule

Let BigCherry MMVQ candidate tuner determine if this policy is already discoverable; if yes, prefer learned dispatch over hard-coded backport.

### Implementation notes

## UP-VK-001 — MoE density-aware Vulkan MMV routing

| Class          | Upstream llama.cpp candidate |
|----------------|------------------------------|
| Backend / arch | Vulkan; RADV RDNA3/3.5/4     |
| Source         | llama.cpp PR \#27332         |
| Source status  | Recheck before port          |
| Prerequisites  | None                         |
| Q8_0 relevance | High                         |

Source link: [<u>llama.cpp PR \#27332</u>](https://github.com/ggml-org/llama.cpp/pull/27332)

### Hypothesis

Routing MUL_MAT_ID to MMV based on expert density rather than a fixed batch\<=8 cutoff removes the batch-9 performance cliff.

### What / why

Source reports a large recovery at B=9 and positive gains through higher concurrency.

### Where to change

Vulkan MUL_MAT_ID dispatch heuristic.

### Trigger / positive cases

- Qwen MoE -np 8,9,16,32,64 on XTX/R9700 RADV.

### Controls / hostile cases

- B\<=8, PP512, dense models.

### Boundary sweep

- Batch/concurrency 1..64 and routing density.

### Correctness gate

Output parity.

### Performance evidence to collect

TG aggregate, kernel path selection, per-request latency.

### Acceptance / regression rule

Promote only if threshold generalizes to target drivers/architectures; preserve explicit boundary cases 8/9.

### Implementation notes

## UP-HIP-002 — AMD DPP/native shuffle path

| Class          | Upstream llama.cpp candidate |
|----------------|------------------------------|
| Backend / arch | HIP RDNA                     |
| Source         | llama.cpp PR \#26466         |
| Source status  | Recheck before port          |
| Prerequisites  | None                         |
| Q8_0 relevance | Medium-high                  |

Source link: [<u>llama.cpp PR \#26466</u>](https://github.com/ggml-org/llama.cpp/pull/26466)

### Hypothesis

Replacing generic shuffle operations with native DPP instructions reduces decode reduction overhead on RDNA.

### What / why

An ISA-level specialization may yield modest broad improvements and proves compiler-neutral CUDA idioms are not always optimal.

### Where to change

Relevant reduction/shuffle helpers.

### Trigger / positive cases

- Qwen Q6/Q8 decode hot kernels on gfx1100/gfx1201.

### Controls / hostile cases

- Non-RDNA architectures; kernels where DPP cannot express required lane pattern.

### Boundary sweep

- DPP vs generic per kernel; inspect ISA.

### Correctness gate

Exact output.

### Performance evidence to collect

Instruction counts, DS usage, TG.

### Acceptance / regression rule

Adopt only where generated ISA and runtime both improve; no blanket intrinsic replacement.

### Implementation notes

## UP-HRX-001 — AMD-native HRX backend comparative lane

| Class          | Upstream strategic backend      |
|----------------|---------------------------------|
| Backend / arch | AMD/ROCm; scope evolves         |
| Source         | llama.cpp PR \#27218 (ggml-hrx) |
| Source status  | Recheck before port             |
| Prerequisites  | None                            |
| Q8_0 relevance | Very high                       |

Source link: [<u>llama.cpp PR \#27218 (ggml-hrx)</u>](https://github.com/ggml-org/llama.cpp/pull/27218)

### Hypothesis

An AMD-native backend with its own tuned Qwen kernel corpus can outperform or simplify the CUDA-derived HIP path.

### What / why

This is strategic, not a cherry-pick. It should be measured as a separate backend lane with identical model/workload contracts.

### Where to change

Separate ggml backend; do not mix into HIP candidate patches.

### Trigger / positive cases

- Qwen3.6 dense/MoE Q8/Q4; PP/TG/MTP; gfx1100/gfx1201 if supported.

### Controls / hostile cases

- HIP and Vulkan matched builds.

- Correctness/backend-op coverage.

### Boundary sweep

- Same captured workload recipes and contexts as production HIP lanes.

### Correctness gate

Backend tests, PPL/temp-0 output, server protocol compatibility as applicable.

### Performance evidence to collect

PP/TG/MTP, VRAM, load time, kernel coverage, compile/build maturity.

### Acceptance / regression rule

Keep experimental until correctness and feature parity are adequate. Promote backend choice only after repeatable advantage on required workloads.

### Implementation notes

## UP-HIP-003 — Pinned host buffer for multi-GPU prompt-cache/state restore

| Class          | Upstream correctness/stability candidate |
|----------------|------------------------------------------|
| Backend / arch | HIP multi-GPU                            |
| Source         | llama.cpp PR \#27405                     |
| Source status  | Recheck before port                      |
| Prerequisites  | None                                     |
| Q8_0 relevance | Backend-wide                             |

Source link: [<u>llama.cpp PR \#27405</u>](https://github.com/ggml-org/llama.cpp/pull/27405)

### Hypothesis

Temporarily host-registering restore buffers prevents ROCm async-copy faults during multi-GPU state restore.

### What / why

Source reports pageable-host async copy faults with 2+ devices and a large stability improvement after pinning.

### Where to change

Prompt-cache/checkpoint state restore H2D path.

### Trigger / positive cases

- Dual XTX tensor/layer split repeated prompt-cache restores.

- Long-running agent sessions.

### Controls / hostile cases

- Single GPU; cache disabled; ordinary prompt processing.

### Boundary sweep

- 100+ restore cycles; state sizes; graph on/off.

### Correctness gate

Restored logits/output identical; no device faults.

### Performance evidence to collect

Restore latency and host registration overhead.

### Acceptance / regression rule

Correctness/stability first. Promote if fault reproduces and pinning eliminates it without material latency regression.

### Implementation notes

## UP-VK-002 — Vulkan FA MMQ FP32 quant-scale calculation

| Class          | Upstream numerical correctness candidate |
|----------------|------------------------------------------|
| Backend / arch | Vulkan                                   |
| Source         | llama.cpp PR \#27413                     |
| Source status  | Recheck before port                      |
| Prerequisites  | None                                     |
| Q8_0 relevance | High                                     |

Source link: [<u>llama.cpp PR \#27413</u>](https://github.com/ggml-org/llama.cpp/pull/27413)

### Hypothesis

Computing FA-MMQ Q quantization scale/reciprocal in FP32 avoids FP16 denormal/overflow corruption.

### What / why

This is a robustness fix for a path BigCherry may rely on heavily at long context.

### Where to change

Vulkan FA MMQ quantization.

### Trigger / positive cases

- Long-context FA with inputs containing very small scales; Q8/Q4 KV variants.

### Controls / hostile cases

- Normal scale ranges; non-MMQ FA.

### Boundary sweep

- Synthetic tiny-amplitude Q tensors plus real model deep-context runs.

### Correctness gate

Reference FA output; no NaN/Inf; model output/PPL.

### Performance evidence to collect

Only regression budget; correctness primary.

### Acceptance / regression rule

Carry if current pin lacks equivalent fix and correctness case reproduces or source patch is low-risk/verified.

### Implementation notes

## UP-VK-003 — Quantized GET_ROWS view-offset correctness

| Class          | Upstream Vulkan correctness |
|----------------|-----------------------------|
| Backend / arch | Vulkan                      |
| Source         | llama.cpp PR \#26854        |
| Source status  | Recheck before port         |
| Prerequisites  | None                        |
| Q8_0 relevance | High                        |

Source link: [<u>llama.cpp PR \#26854</u>](https://github.com/ggml-org/llama.cpp/pull/26854)

### Hypothesis

Quantized GET_ROWS must honor non-zero view offsets to avoid crashes/wrong rows while staying GPU-resident.

### What / why

Relevant to Qwen3-TTS/VL and any future view-based quantized gather.

### Where to change

Vulkan quantized GET_ROWS shader.

### Trigger / positive cases

- Backend-op offset cases for Q4/Q8; Qwen VL/TTS if available.

### Controls / hostile cases

- Offset zero, F16/F32/I32 paths.

### Boundary sweep

- Offsets across block boundaries.

### Correctness gate

Exact reference rows; no fallback/crash.

### Performance evidence to collect

Ensure GPU path remains resident; no material regression.

### Acceptance / regression rule

Correctness baseline if absent from pin.

### Implementation notes

## UP-HIP-004 — Full GPU input-layer offload on AMD UMA

| Class          | Upstream hardware-scoped candidate  |
|----------------|-------------------------------------|
| Backend / arch | HIP integrated GPUs / gfx1151 first |
| Source         | llama.cpp PR \#27426                |
| Source status  | Recheck before port                 |
| Prerequisites  | None                                |
| Q8_0 relevance | High on UMA                         |

Source link: [<u>llama.cpp PR \#27426</u>](https://github.com/ggml-org/llama.cpp/pull/27426)

### Hypothesis

Keeping input-layer tensors GPU-resident avoids repeated CPU/GPU synchronization on UMA AMD systems.

### What / why

Source reported a few-percent prefill improvement; may expose generic synchronization pathology.

### Where to change

Input tensor placement/offload policy.

### Trigger / positive cases

- Strix Halo/gfx1151 Q8 model with full offload.

### Controls / hostile cases

- Discrete XTX/R9700; CPU-offload workloads.

### Boundary sweep

- Prefill batch sizes and model sizes.

### Correctness gate

Output parity.

### Performance evidence to collect

Synchronization count, PP, memory residency.

### Acceptance / regression rule

Hardware-scoped only. Do not generalize to discrete GPUs without independent evidence.

### Implementation notes

## UP-MTP-001 — Adaptive MTP draft depth

| Class          | Upstream speculative decoding                    |
|----------------|--------------------------------------------------|
| Backend / arch | Backend-neutral; source includes dual R9700 ROCm |
| Source         | llama.cpp PR \#27210 / adaptive MTP work         |
| Source status  | Recheck before port                              |
| Prerequisites  | None                                             |
| Q8_0 relevance | Very high                                        |

Source link: [<u>llama.cpp PR \#27210 / adaptive MTP work</u>](https://github.com/ggml-org/llama.cpp/pull/27210)

### Hypothesis

Dynamically adjusting draft depth to observed acceptance beats one fixed depth across heterogeneous coding/prose phases.

### What / why

Source dual-R9700 results showed meaningful coding uplift versus fixed depth while not universally winning on prose.

### Where to change

Speculative controller, not kernel backend.

### Trigger / positive cases

- Qwen3.6/3.8 27B Q8 MTP coding workloads on dual R9700.

- High/medium/low acceptance prompt classes.

### Controls / hostile cases

- Fixed depth 2/3/5/8; speculation off.

### Boundary sweep

- Context 0/32K/64K/128K; content type; min/max depth policies.

### Correctness gate

Speculative temp-0 must match nonspeculative baseline; acceptance accounting correct.

### Performance evidence to collect

Effective TG, draft/verify time, mean depth, acceptance, accepted tokens/step.

### Acceptance / regression rule

Promote only if policy improves weighted production workload; keep fixed-depth escape hatch.

### Implementation notes

## UP-FA-001 — Mixed K/V FlashAttention dispatch for asymmetric KV types

| Class          | Upstream FA dispatch                              |
|----------------|---------------------------------------------------|
| Backend / arch | CUDA/HIP shared dispatcher; validate AMD          |
| Source         | llama.cpp PR \#27150 + related asymmetric KV work |
| Source status  | Recheck before port                               |
| Prerequisites  | None                                              |
| Q8_0 relevance | Very high                                         |

Source link: [<u>llama.cpp PR \#27150 + related asymmetric KV work</u>](https://github.com/ggml-org/llama.cpp/pull/27150)

### Hypothesis

Mixed K/V cache types should use MMA/tile FA when those kernels independently support K and V, instead of being rejected by vector-path constraints.

### What / why

A too-early type-equality check can cause catastrophic fallback. Asymmetric KV is valuable at long context.

### Where to change

CUDA/HIP FA dispatcher eligibility.

### Trigger / positive cases

- K/V: f16/q8_0, q8_0/q4_0, q4_0/q8_0 on XTX/R9700; 32K-240K.

### Controls / hostile cases

- Symmetric f16/f16, q8/q8, q4/q4; unsupported mixed combinations.

### Boundary sweep

- Q rows 1,3,128,512; depths 0/32K/64K/128K/240K.

### Correctness gate

FA reference/PPL; verify actual kernel route and absence of fallback.

### Performance evidence to collect

PP/TG, VRAM, FA kernel selected.

### Acceptance / regression rule

Enable only combinations with correctness coverage and real GPU FA path.

### Implementation notes

## UP-KV-001 — Dedicated F16 dequant path for Q4/Q5 KV

| Class          | Upstream KV prefill candidate                   |
|----------------|-------------------------------------------------|
| Backend / arch | CUDA/HIP shared source; AMD validation required |
| Source         | llama.cpp PR \#27140                            |
| Source status  | Recheck before port                             |
| Prerequisites  | None                                            |
| Q8_0 relevance | Control baseline                                |

Source link: [<u>llama.cpp PR \#27140</u>](https://github.com/ggml-org/llama.cpp/pull/27140)

### Hypothesis

Vectorized half2 dequantization eliminates pathological lower-bit KV prefill overhead and approaches q8_0-speed FA input preparation.

### What / why

Source NVIDIA results were enormous because generic elementwise dequant was the bottleneck. Even partial AMD gains could matter at long context.

### Where to change

Q4/Q5 -\> F16 KV dequant kernels.

### Trigger / positive cases

- Q4_0/Q4_1/Q5_0/Q5_1 KV on long-context Qwen; HIP XTX/R9700.

### Controls / hostile cases

- Q8_0 KV; F16 KV; decode.

### Boundary sweep

- Depth 8K/32K/64K/128K; q_rows 128/512/2048.

### Correctness gate

Exact/tolerant dequant and model output.

### Performance evidence to collect

Dequant kernel time, PP/TG, memory.

### Acceptance / regression rule

If current pin lacks it, promote per KV type only where AMD benchmark wins and tests pass.

### Implementation notes

## UP-VK-004 — Small-N speculative execution avoids inappropriate MMVQ route

| Class          | Upstream Vulkan MTP routing |
|----------------|-----------------------------|
| Backend / arch | Vulkan AMD                  |
| Source         | llama.cpp PR \#25666        |
| Source status  | Recheck before port         |
| Prerequisites  | None                        |
| Q8_0 relevance | Very high                   |

Source link: [<u>llama.cpp PR \#25666</u>](https://github.com/ggml-org/llama.cpp/pull/25666)

### Hypothesis

MTP verify widths 2-4 should be treated as decode-like, not ordinary batch MMVQ, when the latter hurts speed and acceptance.

### What / why

Source gfx1151 results showed both throughput and acceptance improvement by changing the route.

### Where to change

Vulkan matmul routing for small N/speculative batches.

### Trigger / positive cases

- Qwen MTP widths 2,3,4,5,8 on XTX/R9700.

### Controls / hostile cases

- Normal TG N=1; prefill N\>=128; non-MTP small batch if semantics differ.

### Boundary sweep

- N=1..8, context 0/32K/128K; acceptance classes.

### Correctness gate

Temp-0 identity; acceptance must not decrease unexpectedly.

### Performance evidence to collect

Kernel path, verify time, effective TG and acceptance.

### Acceptance / regression rule

Derive small-N selector; MTP workload tag may be measurement context but runtime should prefer shape/arithmetic criteria where possible.

### Implementation notes

## UP-CACHE-001 — Recurrent-state prompt-cache repair for Qwen hybrid models

| Class          | Upstream cache correctness/performance           |
|----------------|--------------------------------------------------|
| Backend / arch | Backend-neutral server/core; AMD source evidence |
| Source         | llama.cpp PR \#24785                             |
| Source status  | Recheck before port                              |
| Prerequisites  | None                                             |
| Q8_0 relevance | High                                             |

Source link: [<u>llama.cpp PR \#24785</u>](https://github.com/ggml-org/llama.cpp/pull/24785)

### Hypothesis

Correct recurrent-state shrink/expand/restore prevents full re-prefill on multi-turn hybrid Qwen sessions.

### What / why

A saved checkpoint is only valuable if recurrent state remains valid. Source RX7900XTX agent runs reported avoidance of huge re-prefill costs.

### Where to change

Hybrid/recurrent memory and prompt-cache handling.

### Trigger / positive cases

- Multi-turn Qwen hybrid agent workload, 20K-100K+ prefixes, cache save/restore.

### Controls / hostile cases

- Standard transformer prompt cache; cache disabled.

### Boundary sweep

- 5-10 turns, growing prompts, tool-call divergence positions.

### Correctness gate

Next-token logits after restore; recurrent state; temp-0 output.

### Performance evidence to collect

Reprocessed prompt tokens, wall time per turn, PP saved.

### Acceptance / regression rule

Correctness primary. If overlapping with BigCherry 1005, compare functionality and do not duplicate.

### Implementation notes

## UP-MGPU-001 — Speculative scheduler caching / reduced cross-GPU synchronization

| Class          | Upstream multi-GPU speculative ideas            |
|----------------|-------------------------------------------------|
| Backend / arch | Multi-GPU; backend concepts need AMD validation |
| Source         | llama.cpp PR \#27173                            |
| Source status  | Recheck before port                             |
| Prerequisites  | None                                            |
| Q8_0 relevance | Very high                                       |

Source link: [<u>llama.cpp PR \#27173</u>](https://github.com/ggml-org/llama.cpp/pull/27173)

### Hypothesis

Caching recurring scheduler plans and eliminating redundant synchronization lowers MTP overhead on multi-GPU systems.

### What / why

The upstream bundle is compound. BigCherry should mine individual scheduler ideas, especially valuable on non-P2P consumer RDNA.

### Where to change

Speculative scheduler and tensor-split coordination.

### Trigger / positive cases

- Dual XTX/R9700 MTP with widths 2..8.

### Controls / hostile cases

- Single GPU; speculation off.

### Boundary sweep

- Isolate plan-cache only, sync-removal only, output-mirroring separately; measure transfers.

### Correctness gate

Temp-0 identity and rollback correctness.

### Performance evidence to collect

Scheduler CPU time, GPU sync count, copies/token, effective TG.

### Acceptance / regression rule

Port only atomic sub-changes with independent wins. Avoid output-layer mirroring unless VRAM/traffic trade is favorable.

### Implementation notes

## BASE-VK-001 — Merged q8_0 KV dequant-once Vulkan FlashAttention baseline

| Class          | Upstream merged baseline                        |
|----------------|-------------------------------------------------|
| Backend / arch | Vulkan                                          |
| Source         | llama.cpp PR \#25494                            |
| Source status  | Merged upstream; verify pinned base contains it |
| Prerequisites  | None                                            |
| Q8_0 relevance | Very high                                       |

Source link: [<u>llama.cpp PR \#25494</u>](https://github.com/ggml-org/llama.cpp/pull/25494)

### Hypothesis

BigCherry Vulkan comparisons must include the dequant-once q8_0 KV FA path if present upstream.

### What / why

This substantially changes long-context q8 KV prefill. Treating an older base as stock Vulkan would misattribute gains to local patches.

### Where to change

Pin/rebase validation, not a new patch if ancestral.

### Trigger / positive cases

- Q8_0 KV pp512 at 32K/64K/128K.

### Controls / hostile cases

- F16/Q4 KV; TG.

### Boundary sweep

- Depth 32K/64K/128K/240K.

### Correctness gate

Backend ops + model output.

### Performance evidence to collect

PP/TG and scratch memory.

### Acceptance / regression rule

Record as baseline capability; port only if pinned base genuinely predates it.

### Implementation notes

## BASE-VK-002 — Merged tiled 0\<-\>2 Vulkan transpose baseline

| Class          | Upstream merged baseline            |
|----------------|-------------------------------------|
| Backend / arch | Vulkan                              |
| Source         | llama.cpp PR \#26585                |
| Source status  | Merged upstream; verify pinned base |
| Prerequisites  | None                                |
| Q8_0 relevance | Indirect                            |

Source link: [<u>llama.cpp PR \#26585</u>](https://github.com/ggml-org/llama.cpp/pull/26585)

### Hypothesis

Performance investigations involving lightning-indexer/transpose-heavy graphs must include the tiled transpose baseline.

### What / why

A generic strided copy previously dominated some prefill workloads; this changes the baseline dramatically.

### Where to change

Pin/rebase validation.

### Trigger / positive cases

- DeepSeek V4/lightning-indexer graph; any captured 0\<-\>2 transpose signatures.

### Controls / hostile cases

- Other CONT/permutation patterns.

### Boundary sweep

- Relevant tensor sizes.

### Correctness gate

Backend parity.

### Performance evidence to collect

Bandwidth and end-to-end PP.

### Acceptance / regression rule

Baseline check only unless absent from pin.

### Implementation notes

# 5. Side-fork candidates to mine, not merge wholesale

## FORK-VK-001 — AMD large cooperative-matrix Vulkan route

| Class          | MrLordCat side-fork                             |
|----------------|-------------------------------------------------|
| Backend / arch | Vulkan; source Windows RDNA4 proprietary driver |
| Source         | MrLordCat/llama.cpp-with-GUI                    |
| Source status  | Recheck before port                             |
| Prerequisites  | None                                            |
| Q8_0 relevance | High                                            |

Source link: [<u>MrLordCat/llama.cpp-with-GUI</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Larger cooperative-matrix pipelines and bn256-like variants can outperform conservative upstream S/M routes on specific RDNA4 shapes.

### What / why

The fork enables this automatically on its tested device. This is a selector research source, not a reason to merge the fork.

### Where to change

Vulkan matmul shader/pipeline selection.

### Trigger / positive cases

- R9700/9070XT captured Qwen prefill shapes; proprietary Vulkan and RADV separately.

### Controls / hostile cases

- Small matrices; gfx1100; drivers without the same coopmat behavior.

### Boundary sweep

- Exact signature map across pipeline variants; ubatch 128..4096.

### Correctness gate

Backend parity.

### Performance evidence to collect

Kernel/PP, shader compile/cache behavior.

### Acceptance / regression rule

Encode only shape+driver/arch conditions proven to win; otherwise reject.

### Implementation notes

## FORK-MGPU-001 — Explicit output-device placement

| Class          | MrLordCat side-fork / topology               |
|----------------|----------------------------------------------|
| Backend / arch | Vulkan and HIP multi-GPU                     |
| Source         | MrLordCat AGENTS / output placement controls |
| Source status  | Recheck before port                          |
| Prerequisites  | None                                         |
| Q8_0 relevance | Very high                                    |

Source link: [<u>MrLordCat AGENTS / output placement controls</u>](https://github.com/MrLordCat/llama.cpp-with-GUI/blob/master/AGENTS.md)

### Hypothesis

Keeping output/vocabulary tensors on the device that naturally owns the last compute stage avoids an unnecessary cross-device boundary.

### What / why

The fork reports strong device-order sensitivity and warns that forcing output to the wrong GPU can severely hurt long-prompt PP.

### Where to change

Model tensor placement / output device policy.

### Trigger / positive cases

- Dual XTX and dual R9700 layer split; both device orders.

### Controls / hostile cases

- Single GPU; tensor split if ownership differs.

### Boundary sweep

- Output owner GPU0/GPU1; normal/reversed order; context 0/32K/100K+.

### Correctness gate

Output equivalence.

### Performance evidence to collect

PCIe copy count/bytes, PP/TG.

### Acceptance / regression rule

Topology-specific policy; never universal GPU0/GPU1 rule.

### Implementation notes

## FORK-MTP-001 — Independent NextN/MTP tensor placement

| Class          | MrLordCat side-fork / MTP    |
|----------------|------------------------------|
| Backend / arch | Vulkan/HIP multi-GPU         |
| Source         | MrLordCat fork MTP placement |
| Source status  | Recheck before port          |
| Prerequisites  | None                         |
| Q8_0 relevance | Very high                    |

Source link: [<u>MrLordCat fork MTP placement</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Placing NextN tensors according to speculative handoff cost rather than output placement reduces cross-device traffic.

### What / why

MTP introduces hidden-state ownership that is different from target output ownership.

### Where to change

Speculative model/tensor device assignment.

### Trigger / positive cases

- Dual XTX/R9700 Qwen MTP.

### Controls / hostile cases

- Single GPU; MTP off.

### Boundary sweep

- NextN owner first/last/main target GPU; device order.

### Correctness gate

Temp-0 identity and acceptance.

### Performance evidence to collect

Copies/token, TG, PP cost.

### Acceptance / regression rule

Choose lowest total transfer-cost placement per topology.

### Implementation notes

## FORK-MTP-002 — Remove four-copy Vulkan MTP handoff

| Class          | MrLordCat side-fork / scheduler      |
|----------------|--------------------------------------|
| Backend / arch | Vulkan multi-GPU                     |
| Source         | MrLordCat fork MTP pipeline controls |
| Source status  | Recheck before port                  |
| Prerequisites  | FORK-MTP-001 analysis recommended    |
| Q8_0 relevance | Very high                            |

Source link: [<u>MrLordCat fork MTP pipeline controls</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Avoiding an expensive four-copy pipeline handoff improves speculative throughput on non-P2P consumer GPUs.

### What / why

Duplicate staging/cross-device copies can erase MTP gains.

### Where to change

Vulkan MTP scheduler/copy path.

### Trigger / positive cases

- Dual RDNA4 Vulkan MTP.

### Controls / hostile cases

- Single GPU; host staging alternatives.

### Boundary sweep

- Copy path variants, draft depth, context.

### Correctness gate

Hidden-state/output identity.

### Performance evidence to collect

Copy count/bytes/time and effective TG.

### Acceptance / regression rule

Fewer copies must translate to E2E win without increased PP/VRAM cost.

### Implementation notes

## FORK-MTP-003 — Warm speculative verification widths

| Class          | MrLordCat side-fork / scheduler cache     |
|----------------|-------------------------------------------|
| Backend / arch | Vulkan primarily; concept backend-neutral |
| Source         | MrLordCat fork warm MTP topology          |
| Source status  | Recheck before port                       |
| Prerequisites  | None                                      |
| Q8_0 relevance | High                                      |

Source link: [<u>MrLordCat fork warm MTP topology</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Prebuilding recurring verification widths 1..n_max+1 removes first-hit graph/scheduler overhead from MTP.

### What / why

Speculation revisits a small width set repeatedly; warmup can make latency predictable.

### Where to change

Speculative scheduler/graph-cache initialization.

### Trigger / positive cases

- MTP widths 1..8; long-lived server.

### Controls / hostile cases

- Short one-shot requests where startup cost matters more.

- MTP off.

### Boundary sweep

- Warmup on/off, width set size, startup time.

### Correctness gate

No stale graph/output errors.

### Performance evidence to collect

First-hit and steady-state verify latency, startup overhead.

### Acceptance / regression rule

Keep if steady-state production benefit exceeds startup/memory cost.

### Implementation notes

## FORK-MGPU-002 — Asynchronous event-chained host staging for ROCm GPU-to-GPU copies

| Class          | MrLordCat side-fork / copy engine  |
|----------------|------------------------------------|
| Backend / arch | HIP multi-GPU without peer access  |
| Source         | MrLordCat ROCm scheduler/copy work |
| Source status  | Recheck before port                |
| Prerequisites  | None                               |
| Q8_0 relevance | Very high                          |

Source link: [<u>MrLordCat ROCm scheduler/copy work</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Event-chained host staging overlaps unavoidable GPU-\>host-\>GPU copies and reduces transfer stalls when P2P is unavailable.

### What / why

Consumer RDNA often lacks usable peer access; synchronous staging serializes both devices.

### Where to change

Cross-device copy path.

### Trigger / positive cases

- Dual XTX/R9700 non-P2P layer/tensor split.

### Controls / hostile cases

- Single GPU; peer-capable hardware if available.

### Boundary sweep

- Transfer sizes from actual hidden/output tensors; prompt 30K/100K.

### Correctness gate

Exact copied tensors.

### Performance evidence to collect

Transfer latency, overlap, PP/TG.

### Acceptance / regression rule

Promote if \>1% E2E and no synchronization/correctness issues.

### Implementation notes

## FORK-CKPT-001 — Batched recurrent-checkpoint reads by backend

| Class          | MrLordCat side-fork / cache I/O         |
|----------------|-----------------------------------------|
| Backend / arch | Vulkan multi-GPU                        |
| Source         | MrLordCat fork batched checkpoint reads |
| Source status  | Recheck before port                     |
| Prerequisites  | None                                    |
| Q8_0 relevance | High                                    |

Source link: [<u>MrLordCat fork batched checkpoint reads</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

Grouping checkpoint tensor reads per backend and synchronizing once reduces restore overhead.

### What / why

Sequential per-tensor staging/sync magnifies latency in recurrent checkpoint workflows.

### Where to change

Checkpoint read/restore orchestration.

### Trigger / positive cases

- Hybrid/recurrent multi-GPU agent workload.

### Controls / hostile cases

- Single tensor/checkpoint; single GPU.

### Boundary sweep

- Checkpoint size and number of tensors.

### Correctness gate

Restored state/logits identical.

### Performance evidence to collect

Restore time, prompt TPS after restore.

### Acceptance / regression rule

Adopt if deterministic and meaningful multi-turn wall-time improvement.

### Implementation notes

## FORK-HIP-001 — Restore / benchmark rocWMMA FlashAttention on RDNA4

| Class          | MrLordCat side-fork / FA research |
|----------------|-----------------------------------|
| Backend / arch | HIP gfx1201                       |
| Source         | MrLordCat fork RDNA4 rocWMMA FA   |
| Source status  | Recheck before port               |
| Prerequisites  | None                              |
| Q8_0 relevance | High                              |

Source link: [<u>MrLordCat fork RDNA4 rocWMMA FA</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

### Hypothesis

For deep-context RDNA4 prefill, rocWMMA FA may materially outperform the generic/native replacement on selected head/KV shapes.

### What / why

The fork reports large long-context PP gains after restoring rocWMMA. Upstream has also moved away from some rocWMMA paths, so this is a research comparison, not an automatic reversion.

### Where to change

FA backend/kernel selection.

### Trigger / positive cases

- R9700 Qwen long-context BF16/Q8 KV; depths 32K/64K/128K.

### Controls / hostile cases

- Decode; gfx1100; current upstream FA kernels.

### Boundary sweep

- Head size, q_rows, depth, KV type, graph on/off.

### Correctness gate

FA backend tests, PPL, long-run stability.

### Performance evidence to collect

PP/TG, VRAM/scratch, compile/runtime stability.

### Acceptance / regression rule

Keep only as a conditional experimental kernel if it clearly wins and is stable; beware graph interaction/crash reports.

### Implementation notes

# 6. Driver, runtime and external-library experiments

## SYS-PCIE-001 — PCIe ASPM performance policy

| Class          | System configuration experiment      |
|----------------|--------------------------------------|
| Backend / arch | Linux discrete RDNA                  |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High for bandwidth-bound dense       |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Disabling aggressive PCIe link power saving reduces latency for frequent transfers and may improve dense decode on some systems.

### What / why

The R9700 experiment corpus reported a notable dense decode gain in one setup, while MoE was neutral. This is topology/system dependent.

### Where to change

OS PCIe ASPM policy, outside llama.cpp.

### Trigger / positive cases

- R9700/XTX discrete GPU lanes with known host-device traffic.

### Controls / hostile cases

- MoE vs dense; single-GPU fully resident; idle/power measurements.

### Boundary sweep

- Default vs performance policy; repeat after cold boot; capture PCIe link state.

### Correctness gate

No output impact expected.

### Performance evidence to collect

TG/PP, PCIe transaction latency, power/idle behavior.

### Acceptance / regression rule

Document as optional host tuning only if local hardware reproduces; never bake into runtime.

### Implementation notes

## VK-TUNE-001 — Vulkan rm_kq specialization sweep

| Class          | Driver/kernel tuning experiment      |
|----------------|--------------------------------------|
| Backend / arch | Vulkan RDNA                          |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | Medium                               |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Lower rm_kq may reduce register pressure and improve selected FA/matvec paths on RDNA4.

### What / why

The experiment corpus found driver-dependent gains; this is a tuning dimension, not a universal constant.

### Where to change

Vulkan kernel compile/tuning constant.

### Trigger / positive cases

- Captured FA/matvec signatures on R9700.

### Controls / hostile cases

- RADV vs AMD proprietary/AMDVLK; gfx1100.

### Boundary sweep

- rm_kq=1,2,3,4 by signature and driver.

### Correctness gate

Backend parity.

### Performance evidence to collect

VGPR/registers, kernel time, PP/TG.

### Acceptance / regression rule

Only per-driver/arch/signature if repeatable.

### Implementation notes

## VK-TUNE-002 — Large physical ubatch for Vulkan MoE prefill

| Class          | Runtime tuning experiment            |
|----------------|--------------------------------------|
| Backend / arch | Vulkan RDNA4 primarily               |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High                                 |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

MoE Vulkan prefill may need larger physical ubatch to reach efficient occupancy/tiles.

### What / why

The source experiment saw substantial MoE PP gains at ubatch 2048 while dense models did not share the same behavior.

### Where to change

Runtime batch/ubatch selection; may inform autotuner recommendation rather than code patch.

### Trigger / positive cases

- Qwen MoE Vulkan.

### Controls / hostile cases

- Dense Qwen; memory-constrained context.

### Boundary sweep

- ubatch 256,512,1024,2048,4096 where possible.

### Correctness gate

Output unchanged.

### Performance evidence to collect

PP, VRAM, latency.

### Acceptance / regression rule

Store as model/workload tuning recommendation or campaign search dimension, not runtime kernel patch.

### Implementation notes

## VK-DRV-001 — RADV vs AMD proprietary/AMDVLK driver lane

| Class          | Driver comparison                    |
|----------------|--------------------------------------|
| Backend / arch | Vulkan AMD                           |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High                                 |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Driver/compiler choice materially changes which Vulkan kernels/tuning constants win.

### What / why

Source measurements show different prefill/decode winners and different responses to coopmat/rm_kq changes.

### Where to change

Benchmark environment identity.

### Trigger / positive cases

- Same exact build/model/config under each available ICD.

### Controls / hostile cases

- Record driver version and shader cache state.

### Boundary sweep

- PP/TG/MTP and critical microkernels.

### Correctness gate

Outputs must match within backend tolerance.

### Performance evidence to collect

All standard metrics.

### Acceptance / regression rule

Make driver identity mandatory in evidence; never merge benchmark corpora across ICDs.

### Implementation notes

## VK-DRV-002 — AMD graphics queue for compute on proprietary Vulkan

| Class          | Driver queue experiment              |
|----------------|--------------------------------------|
| Backend / arch | AMD proprietary/AMDVLK               |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High                                 |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Using the graphics queue can help dispatch-heavy MoE but hurt dense workloads.

### What / why

Source found opposite effects by workload class, making it a good example of a conditional environment selector.

### Where to change

Vulkan queue selection.

### Trigger / positive cases

- MoE decode.

### Controls / hostile cases

- Dense Qwen decode/prefill.

### Boundary sweep

- Queue choice by workload.

### Correctness gate

Output parity.

### Performance evidence to collect

TG/PP, queue utilization.

### Acceptance / regression rule

Never global; if retained, condition on proven workload/driver.

### Implementation notes

## VK-DRV-003 — Cooperative-matrix bypass on AMD proprietary Vulkan

| Class          | Driver shader-route experiment       |
|----------------|--------------------------------------|
| Backend / arch | AMD proprietary/AMDVLK               |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High                                 |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Some dense RDNA4 shapes run faster on conventional shaders than proprietary-driver coopmat codegen.

### What / why

Source reported a dense prefill improvement when coopmat was disabled, while RADV behaved differently.

### Where to change

Vulkan coopmat route/override.

### Trigger / positive cases

- Dense R9700 prefill shapes.

### Controls / hostile cases

- MoE, RADV, gfx1100.

### Boundary sweep

- coopmat on/off by signature.

### Correctness gate

Output parity.

### Performance evidence to collect

PP, kernel time, shader ISA if available.

### Acceptance / regression rule

Conditional driver-specific route only.

### Implementation notes

## VK-ACO-001 — Audit redundant RADV/ACO scalar waits

| Class          | Mesa/RADV compiler research          |
|----------------|--------------------------------------|
| Backend / arch | RADV/ACO                             |
| Source         | R9700 experiments discussion \#21043 |
| Source status  | Recheck before port                  |
| Prerequisites  | None                                 |
| Q8_0 relevance | High research                        |

Source link: [<u>R9700 experiments discussion \#21043</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

### Hypothesis

Shader restructuring or an ACO compiler fix can remove redundant scalar waits in quantized GEMV and raise effective bandwidth.

### What / why

The R9700 study identified many redundant wait instructions in a Q4_K GEMV. Q8 shaders should be audited for the same pattern.

### Where to change

Potentially GLSL/SPIR-V structure or Mesa ACO; determine source after ISA study.

### Trigger / positive cases

- Q4_K and Q8_0 GEMV hot shaders on RADV.

### Controls / hostile cases

- Equivalent shader on proprietary driver; unaffected kernels.

### Boundary sweep

- Minimal source transformations; compiler versions.

### Correctness gate

Output parity.

### Performance evidence to collect

ISA wait count, effective bandwidth, TG.

### Acceptance / regression rule

Only port a llama.cpp shader rewrite if it reliably changes ACO codegen; otherwise file/track Mesa issue instead.

### Implementation notes

## VK-FA-001 — Decouple Vulkan FA occupancy tuning from exact shared-memory capability equality

| Class          | Vulkan robustness/tuning |
|----------------|--------------------------|
| Backend / arch | Vulkan AMD               |
| Source         | llama.cpp issue \#26163  |
| Source status  | Recheck before port      |
| Prerequisites  | None                     |
| Q8_0 relevance | High when FA uses q8 KV  |

Source link: [<u>llama.cpp issue \#26163</u>](https://github.com/ggml-org/llama.cpp/issues/26163)

### Hypothesis

An exact maxComputeSharedMemorySize==64KiB check is too fragile to identify a performance tuning path.

### What / why

Driver-reported capability changes can silently disable occupancy tuning even when hardware remains suitable.

### Where to change

Vulkan FA tuning selector.

### Trigger / positive cases

- Affected AMD driver versions where reported LDS/shared-memory limit changed.

### Controls / hostile cases

- Devices genuinely limited below shader requirement; non-AMD.

### Boundary sweep

- Explicit override and architecture-based selector; 32/64KiB reported limits.

### Correctness gate

Never launch shader exceeding legal limit.

### Performance evidence to collect

FA PP/TG before/after selector.

### Acceptance / regression rule

Separate performance occupancy heuristic from hard legality check.

### Implementation notes

## HIP-GRAPH-001 — Selective HIP graph bypass for unstable FA shapes

| Class          | ROCm graph stability     |
|----------------|--------------------------|
| Backend / arch | HIP gfx1201 long context |
| Source         | llama.cpp issue \#24961  |
| Source status  | Recheck before port      |
| Prerequisites  | None                     |
| Q8_0 relevance | High                     |

Source link: [<u>llama.cpp issue \#24961</u>](https://github.com/ggml-org/llama.cpp/issues/24961)

### Hypothesis

Disabling graph capture only for the problematic long-context FA configuration can avoid crashes without losing graph benefits globally.

### What / why

Source reports long-context R9700 FA graph failure while disabling graphs lets the kernel complete.

### Where to change

HIP graph eligibility/cache.

### Trigger / positive cases

- R9700 100K+ context FA with the reported kernel family.

### Controls / hostile cases

- Short context, other ops, graphs globally off.

### Boundary sweep

- Depth around failure threshold; graph capture/recapture events.

### Correctness gate

No crash and same output.

### Performance evidence to collect

PP/TG and graph overhead.

### Acceptance / regression rule

Use the narrowest predicate that removes failure. Prefer a recapture fix if root cause can be proven.

### Implementation notes

## HIP-GRAPH-002 — Force graph recapture when FA stream topology changes

| Class          | ROCm graph stability                |
|----------------|-------------------------------------|
| Backend / arch | HIP                                 |
| Source         | llama.cpp issue \#24961             |
| Source status  | Recheck before port                 |
| Prerequisites  | HIP-GRAPH-001 repro/instrumentation |
| Q8_0 relevance | High                                |

Source link: [<u>llama.cpp issue \#24961</u>](https://github.com/ggml-org/llama.cpp/issues/24961)

### Hypothesis

The long-context UpdateStreams failure is caused by stale captured topology and can be fixed by recapturing at the transition.

### What / why

A real root-cause fix is preferable to disabling graphs.

### Where to change

HIP graph cache/update logic.

### Trigger / positive cases

- Exact failure case; log graph shape/stream mapping.

### Controls / hostile cases

- Shapes with stable graph topology.

### Boundary sweep

- Force recapture at candidate transition points.

### Correctness gate

No failure; output parity.

### Performance evidence to collect

Recapture frequency/cost and retained graph speedup.

### Acceptance / regression rule

Promote over selective bypass if robust across repeated long runs.

### Implementation notes

## VK-SUB-001 — Architecture-aware Vulkan submission cap

| Class          | Vulkan scheduler robustness  |
|----------------|------------------------------|
| Backend / arch | Vulkan gfx1201 and older AMD |
| Source         | llama.cpp issue \#26679      |
| Source status  | Recheck before port          |
| Prerequisites  | None                         |
| Q8_0 relevance | Backend-wide                 |

Source link: [<u>llama.cpp issue \#26679</u>](https://github.com/ggml-org/llama.cpp/issues/26679)

### Hypothesis

Submission batching should cap work differently by architecture so timeout protection for older GPUs does not devastate RDNA4 throughput.

### What / why

A generalized safety heuristic reportedly caused a huge R9700 regression.

### Where to change

Vulkan command submission batching.

### Trigger / positive cases

- R9700 PP/TG; older GCN timeout-prone control.

### Controls / hostile cases

- Other vendors/architectures.

### Boundary sweep

- Submission work/FLOP thresholds.

### Correctness gate

No DeviceLost/timeout/corruption.

### Performance evidence to collect

PP/TG, submission count, GPU fault telemetry.

### Acceptance / regression rule

Choose architecture-sensitive ceiling that preserves safety and modern performance.

### Implementation notes

## VK-SUB-002 — Bound startup submission-ramp growth by learned safe ceiling

| Class          | Vulkan scheduler robustness                         |
|----------------|-----------------------------------------------------|
| Backend / arch | Vulkan AMD                                          |
| Source         | llama.cpp issue \#26679 and related submission work |
| Source status  | Recheck before port                                 |
| Prerequisites  | VK-SUB-001                                          |
| Q8_0 relevance | Backend-wide                                        |

Source link: [<u>llama.cpp issue \#26679 and related submission work</u>](https://github.com/ggml-org/llama.cpp/issues/26679)

### Hypothesis

Adaptive submission ramp must never grow beyond a known-safe architecture cap.

### What / why

A safety limit is ineffective if startup adaptation later exceeds it.

### Where to change

Submission-size ramp logic.

### Trigger / positive cases

- Repeated long runs on affected AMD GPUs.

### Controls / hostile cases

- Architectures without cap.

### Boundary sweep

- Warmup/ramp duration and ceiling.

### Correctness gate

No timeout/corruption.

### Performance evidence to collect

Stable steady-state PP/TG.

### Acceptance / regression rule

Keep if it prevents safety regressions without reducing normal throughput.

### Implementation notes

## HIPBLAS-001 — Offline hipBLASLt tuning for captured llama.cpp GEMM signatures

| Class          | ROCm library tuning                        |
|----------------|--------------------------------------------|
| Backend / arch | HIP; gfx1100/gfx1201                       |
| Source         | AMD hipBLASLt offline tuning documentation |
| Source status  | Recheck before port                        |
| Prerequisites  | None                                       |
| Q8_0 relevance | Very high                                  |

Source link: [<u>AMD hipBLASLt offline tuning documentation</u>](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html)

### Hypothesis

hipBLASLt's default heuristic is not always optimal for llama.cpp's exact GEMM signatures; offline-tuned solution files can improve selected large-M prefill operations.

### What / why

Low-cost way to test vendor-library headroom before writing kernels.

### Where to change

Benchmark/build environment; no llama.cpp code change initially.

### Trigger / positive cases

- All captured hipBLASLt signatures from Qwen3.6-27B Q8 and 35B MoE.

### Controls / hostile cases

- Native MMQ/MMVQ and untuned hipBLASLt.

### Boundary sweep

- Deduplicate M/N/K/transposes/types; tune top time×calls shapes.

### Correctness gate

GEMM output tolerance.

### Performance evidence to collect

Per-signature kernel time + end-to-end PP.

### Acceptance / regression rule

Version tuning files by GPU arch and ROCm/hipBLASLt version; use only where winner is stable.

### Implementation notes

## CK-001 — Use Composable Kernel profiler as an oracle for hot GEMM signatures

| Class          | ROCm library research  |
|----------------|------------------------|
| Backend / arch | AMD GPU                |
| Source         | ROCm Composable Kernel |
| Source status  | Recheck before port    |
| Prerequisites  | None                   |
| Q8_0 relevance | Indirect               |

Source link: [<u>ROCm Composable Kernel</u>](https://github.com/ROCm/composable_kernel)

### Hypothesis

CK can reveal better tile/algorithm choices for exact llama.cpp signatures even if BigCherry never takes CK as a runtime dependency.

### What / why

Provides a research oracle for vendor/architecture-tuned GEMM configurations.

### Where to change

Offline benchmarking tooling.

### Trigger / positive cases

- Top captured dense/MoE GEMM signatures.

### Controls / hostile cases

- BigCherry native MMQ/MMVQ/hipBLASLt winners.

### Boundary sweep

- Exact signatures ranked by time×calls.

### Correctness gate

Numerical parity.

### Performance evidence to collect

Kernel throughput and resource use.

### Acceptance / regression rule

If CK wins, either reproduce the useful specialization in ggml or evaluate a very narrow integration; do not add a broad dependency by default.

### Implementation notes

# 7. Recommended first implementation wave for BigCherry

| Priority | Experiment          | Reason                                                                                                 |
|----------|---------------------|--------------------------------------------------------------------------------------------------------|
| P0       | AMD-Q8-001          | Directly targets the Q8 tiny-M/MTP gap and fits existing MMVQ/MMQ candidate machinery.                 |
| P0       | AMD-MOE-001         | Small selector with potentially large MoE PP payoff; stress-testable with controlled routing.          |
| P0       | UP-HIP-001          | Dynamic MMVQ warps has direct R9700/XTX evidence and may already be discoverable by BigCherry's tuner. |
| P0       | UP-VK-001           | Vulkan B=8/9 cliff is easy to reproduce and low-risk to isolate.                                       |
| P0       | HIPBLAS-001         | Very low implementation risk; establishes vendor-library headroom before new kernels.                  |
| P1       | AMD-GEMM-002/004    | High-value Q8 prefill experiment but VRAM-heavy; test after crossover tooling is ready.                |
| P1       | AMD-MOE-002/003     | Compact launch after routing benchmark infrastructure.                                                 |
| P1       | AMD-FUS-001/002/003 | Small decode fusions; useful cumulative lane.                                                          |
| P1       | UP-MTP-001          | Adaptive MTP already has dual-R9700 evidence.                                                          |
| P1       | FORK-MGPU-001/002   | Topology/copy placement can dwarf kernel tweaks on non-P2P systems.                                    |
| P2       | AMD-GDN-001..004    | Potentially large but model-specific/invasive.                                                         |
| P2       | UP-HRX-001          | Strategic comparative backend lane.                                                                    |
| P2       | VK-ACO-001 / CK-001 | Research-oracle work after top bottlenecks are measured.                                               |

# 8. Model / workload qualification matrix

| Model lane                       | Purpose                               | Relevant experiments                              |
|----------------------------------|---------------------------------------|---------------------------------------------------|
| Qwen3.6-27B Q8_0 + MTP           | Dense Q8, MMQ/MMVQ, MTP, long context | AMD-Q8, GEMM crossover, MTP, FA, multi-GPU        |
| Qwen3.6-27B Q4_K/Q6_K            | Dense K-quant control                 | Quant-specific regressions and AMD MMQ techniques |
| Qwen3.6-35B-A3B Q8_0             | MoE Q8                                | MoE tile/compact launch, shared expert, Q8 decode |
| Qwen3.6-35B-A3B Q4_K_M           | MoE K-quant control                   | AMD source-comparable lane                        |
| Qwen hybrid/GDN model            | Recurrent/GDN/SSM                     | SSM layout, GDN kernel, prompt cache              |
| Llama 3.x 8B Q8_0                | Conventional dense control            | Catch Qwen-specific assumptions                   |
| Alternative MoE (Gemma/DeepSeek) | Routing/generalisation control        | Catch Qwen-only MoE selectors                     |
| DeepSeek-V4 Flash                | Vulkan/indexer/large graph stress     | Transpose and Vulkan scheduler paths              |
| Small Qwen 0.5B-4B               | Launch-overhead control               | K/V fusion and decode epilogues                   |

# 9. Common boundary sweeps

- MMQ/MMVQ: physical M / output width 1,2,3,4,8,16,24,32,48,64,128,256,512 and real captured N/K.

- Prefill: prompt lengths 128,512,1K,4K,16K,64K+ crossed with physical ubatch 128,256,512,1024,2048,4096 where memory permits.

- FlashAttention: q_rows 1,3,8,128,512,2048 crossed with KV depth 0,8K,32K,64K,128K,192K,240K.

- MTP: verify width 1,2,3,4,5,6,8,10; content classes high-acceptance code, ordinary coding, prose, low-acceptance prose.

- MoE: natural captured routing plus uniform, Zipf, mild skew, heavy skew and single-hot; record mean/p95/max/empty experts.

- Multi-GPU: 1/2/3 GPU where available; device order; layer vs tensor split; output/NextN owner; P2P capability; copy bytes/time.

- Driver: driver/ICD version is part of evidence identity. Do not compare RADV and proprietary results as one population.

# 10. Source index

[<u>AMD PR Set 1 - Tooling context</u>](https://github.com/ggml-org/llama.cpp/discussions/26333)

[<u>AMD PR Set 2 - MMQ/MoE</u>](https://github.com/ggml-org/llama.cpp/discussions/26349)

[<u>AMD PR Set 4 - Matvec/Fusion</u>](https://github.com/ggml-org/llama.cpp/discussions/26378)

[<u>AMD PR Set 5 - GEMM Prefill</u>](https://github.com/ggml-org/llama.cpp/discussions/26379)

[<u>AMD PR Set 6 - Layout/Concurrency</u>](https://github.com/ggml-org/llama.cpp/discussions/26380)

[<u>R9700 experiment corpus</u>](https://github.com/ggml-org/llama.cpp/discussions/21043)

[<u>MrLordCat fork</u>](https://github.com/MrLordCat/llama.cpp-with-GUI)

[<u>MrLordCat AGENTS</u>](https://github.com/MrLordCat/llama.cpp-with-GUI/blob/master/AGENTS.md)

[<u>hipBLASLt offline tuning</u>](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html)

[<u>Composable Kernel</u>](https://github.com/ROCm/composable_kernel)

# 11. Delivery rule for agents

Before implementing any item, recheck current upstream status and the BigCherry pin. If the exact change is already ancestral, record it as a baseline and do not create a patch. If a source PR is compound, port only one logical hypothesis at a time unless the source proves the pieces are inseparable for correctness. Update external-sources.toml with immutable identities and create the Experiment Contract before benchmarking.
