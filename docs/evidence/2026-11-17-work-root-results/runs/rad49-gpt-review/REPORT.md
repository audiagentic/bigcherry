**Flash-Next on four heterogeneous AMD GPUs — review report, 10 November 2026**

The best next steps are to improve the existing accurate RDNA2 kernel, evaluate placement by **per-layer completion time**, and extend overlap into activation transfer and expert computation. An integer-dot kernel is worth investigating, but the local codebook and E4M3 activations make it a substantial numerical change. There is no verified basis here for promising a 5× tail speedup.

This review used source inspection and online research only. No builds, GPU programs, repository edits, or pushes were performed. The session’s read-only filesystem permissions prevented creation of the requested `REPORT.md`; this response contains the full report.

Local line references describe the inspected working tree, whose HEAD was `ad44f7c65f32b7ec482ee4eea8da5fa644f3e2a9`. “Verified” below means supported by inspected code, supplied findings, or a cited primary source. Performance projections are explicitly identified as inference.

**1. What the current implementation already does**

Several frequently suggested MoE optimizations are already present:

- **Weight decoding is amortized across 16 rows.** `moe_lane_gemm.h:834–974` implements the quad-DPP kernel. Its decoded weights are reused inside the row loop at lines 944–958. Recommending this reuse as a new optimization would repeat completed work.
- **Activations are already compressed and copied sparsely.** `moe_lane_front.h:74–116` marks tokens with routes and copies only their E4M3 codes and scales. Dispatch enables this path at `moe_remote_gpu.h:644–665`.
- **Results are already reduced locally and written with vector stores.** `moe_lane_reduce.h:91–146` combines expert rows in slot order using fp32 accumulation, rounds once to bf16, and writes 16 bytes per thread.
- **CPU combine is already parallel and chunked.** `worker_combine.h:62–120` contains persistent workers; lines 218–237 wait for helper chunks, combine them, and publish completion.
- **Chunk publication already preserves explicit ordering.** `moe_lane_reduce.h:24–39` and `140–168` drain stores and publish completion. Optimizations must preserve these semantics.

The remaining pipeline limitation is concrete: within a request, activation preparation, gate/up, gated quantization, down, and reduction are queued sequentially on `w.stream`. Result chunks become available during the final reduction, **after the request’s GEMMs finish** (`moe_remote_gpu.h:1001–1034`). Existing chunking overlaps reduction with CPU combine; it does not establish a pipeline through the entire expert computation.

**2. Ranked opportunities**

All gains in this table are investigation targets or conditional bounds, not measured improvements.

| Rank | Opportunity | Gain assessment | Effort | KL risk |
|---|---|---|---|---|
| 1 | Improve the accurate fp16-dot2 kernel’s decoding and scheduling | Target 1.2–1.5× GEMM; conditional cold-TTFT reduction up to roughly 9–17% | Medium | Low if output bits remain identical |
| 2 | Place experts using per-layer device finish times | Unknown; potentially useful for correlated routing, little benefit under uniform routing | Medium–high | Medium: ownership changes reduction grouping |
| 3 | Prototype W4NL/E4M3 → integer-dot execution | Meaningful upside, but no defensible speed estimate before conversion and quality tests | High | High |
| 4 | Pipeline transfers and GEMMs; compare mapped I/O with staged DMA | Bounded by exposed transfer time; halving result-write time alone gives at most roughly 6% cold-TTFT reduction under the supplied profile | High | Low for identical arithmetic; substantial protocol risk |
| 5 | Compute selected cold host experts on CPU | Most plausible for low-row-count experts; prefill benefit uncertain | High | Medium–high |
| 6 | Continue prefill-only R9700 BAR-push evaluation | A few percent whole-prefill upside is plausible; decode benefit is negligible | Medium | Low numerically; transport correctness remains material |

**Rank 1 — Improve the accurate kernel before changing arithmetic**

**Verified locally:** the quad kernel stages activations in double-buffered LDS, uses an LDS codebook, performs ordered dot8/dot16/K64/K128 folds, disables reassociation and contraction, and rounds to bf16 once. See `moe_lane_gemm.h:839–867`, `869–889`, and `936–970`.

Two specific experiments deserve priority.

First, compare its LDS codebook lookup with the register-permute decoder already used by the local GEMV. `w4nl_to_h2x2` at `moe_lane_gemm.h:357–365` constructs the same half values with byte permutations. The quad kernel instead indexes `lut` repeatedly at lines 862–867. Reusing the register decoder, with the correct nibble ordering, could remove LDS lookup instructions without changing values.

This is an **inference about performance**, not a demonstrated bottleneck. More permutations and longer register lifetimes could cancel the saving. Inspect generated instructions, VGPR allocation, spills, LDS instruction counts, and occupancy before selecting a variant. Likewise, compare packed E4M3 widening against the scalar widening in the quad staging loop; exact widening machinery already exists nearby.

Second, drive work partitioning from **rows per expert**, rather than total request rows alone. `rad46_rows_policy.h:20–52` interpolates a gate/up crossover near 309 request rows and a down crossover near 922. Dispatch applies those choices to the whole request (`moe_remote_gpu.h:776–777`). Yet a 2048-token request may contain both nearly empty and heavily loaded experts.

Useful candidates are compact work descriptors for `(expert, row tile, output tile)`, separate treatment of partial row tiles, and splitting long expert runs across independently scheduled row tiles. Preserve each output’s existing K-fold arithmetic. Do not assume larger row blocks win: the supplied replay already rejects several plausible alternatives, and the current template explicitly limits RB to 16.

**External evidence:** vLLM’s `fused_moe.py`, especially `fused_moe_kernel_gptq_awq`, uses expert-sorted token blocks and grouped tile ordering. Triton’s grouped-GEMM tutorial demonstrates device-side scheduling across multiple GEMMs. These support borrowing scheduling techniques, while providing no gfx1030 performance guarantee. [vLLM source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L108), [Triton grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)

**Acceptance:** reproduce bit-identical results across captured requests, including sparse and partial expert runs. Then measure gate/up and down separately and repeat cold prompts, document prefill, and sustained eight-session filling. The 40-request replay is a valuable starting corpus, not sufficient coverage of every shape.

**Rank 2 — Optimize per-layer finish time, including transfers and engine work**

**Verified locally:** `worker_heat.h:46–48` models a device with a byte budget only. The objective at lines 125–149 measures the fraction of helper-owned heat assigned to the tail and constructs a heat-ranked placement. It has no compute-speed or PCIe model. Engine-local experts are fixed and excluded from candidate placement (`current == -1`, lines 62 and 140).

That explains why its “no improvement” result does not establish that a speed-aware policy is useless.

The right objective is approximately:

`minimize workload-weighted sum over layers of critical-path completion time`

Each layer’s estimate should include engine work, helper gate/up and down costs, unique activation rows, unique output rows, CPU combine, and migration cost. A route count alone is insufficient: multiple routes for one token reuse its transferred activation and produce one reduced result row.

**Inference:** globally coldest-first placement need not minimize this objective. Moving an expert off an overloaded layer can help even when it displaces a somewhat hotter expert onto the tail in a layer with available slack. Correlations between expert activations matter as well as average frequency.

Evaluate this offline on captured requests before implementing live movement. Fit device costs by stage and expert row count; the observed XTX/tail ratios differ considerably between gate/up, down, and transfers. Include both cold and ordinary workloads, retain byte budgets, and charge migration against expected reuse.

**External evidence:** HybriMoE §IV-B balances predicted CPU/GPU execution times and prioritizes work by load. DeepSeek EPLB’s `balanced_packing` and `replicate_experts` provide concrete load-balancing and replication algorithms. Neither is a ready-made optimizer for this heterogeneous PCIe topology. [HybriMoE §IV-B](https://arxiv.org/html/2504.05897v1#S4.SS2), [EPLB implementation](https://github.com/deepseek-ai/EPLB/blob/main/eplb.py)

**Limitations:** nearly uniform routing and fully occupied memory leave little placement freedom. Replication consumes additional capacity and can force useful experts into the host pool. Replicate only when saved critical-path time exceeds the resulting spill cost. Also, moving an expert between helpers changes which sums are rounded locally; identical per-expert GEMMs do not guarantee identical final rows.

**Rank 3 — Investigate integer dot4, with explicit numerical boundaries**

**Verified externally:** llama.cpp does decode IQ4_NL codebook nibbles into packed int8 values and dot them against Q8_1 activations:

- `vecdotq.cuh:1260–1277`, `vec_dot_iq4_nl_q8_1`, performs table expansion, integer dot products, and scale application. [MMVQ dot implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh#L1260)
- `mmq-load-tiles.cuh`, around lines 1350–1412, expands IQ4_NL into packed integer tiles. [MMQ tile loader](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq-load-tiles.cuh#L1350)
- `common.cuh:652–657` maps its dot4 wrapper to `__builtin_amdgcn_sdot4` on RDNA2 and `__builtin_amdgcn_sudot4` on RDNA3. [HIP integer-dot wrapper](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh#L652)
- `mmq.cu:63–68` dispatches IQ4_NL with Q8 activation precision. [MMQ dispatch](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu#L63)

**The local format is materially different.** `moe_lane_gemm.h:32–33` defines magnitudes:

`1.125, 3.25, 5.5, 8, 11, 14, 18, 22`

with both signs. Multiplying by eight gives integer magnitudes:

`9, 26, 44, 64, 88, 112, 144, 176`.

These cannot be represented exactly by one signed-int8 codebook with a common scale. A uniform affine byte representation also cannot cover that full integer span exactly. Options include approximate table conversion or multiple integer components with correction terms.

Separately, E4M3 values within a block do not generally lie exactly on one common int8 grid. E4M3-to-Q8 conversion therefore introduces another approximation unless a more elaborate representation is used.

Finally, exact integer products would still not automatically reproduce the current ordered floating-point accumulation. Preserve the distinction between exact input representation, accurate dot products, and bit-identical outputs.

**Concrete experiment:** first evaluate activation conversion and codebook approximation against captured data, without replacing the production path. Measure errors separately for gate/up and down, then full-model KL with unchanged batching. Include conversion, packing, scale handling, and gated-activation conversion in timing.

A useful fresh external result is llama.cpp **Vulkan PR #31764**, commit `d900cbf52da3a2d9e8f9df21edd420ea1811082f`, inspected as open. On an RX 6600M it reports IQ4_NL dense matmul increasing from 8.35 to 11.18 TFLOPS and approximately 20% higher prompt processing for Qwen3.5-9B. Its correctness and KL results support feasibility, not bit identity. It is a different GPU, backend, codebook, and workload; the reported throughput is not a 6900 XT MoE result. [PR, benchmark shapes and quality results](https://github.com/ggml-org/llama.cpp/pull/31764)

**Rank 4 — Extend overlap before compressing results**

For a helper touching `U` distinct tokens, the approximate payload is:

| Payload | Bytes |
|---|---:|
| E4M3 input plus fp32 scales per 128 values | `U × 2640` |
| Reduced bf16 output | `U × 5120` |

These exclude routes, control traffic, and transaction overhead. The output payload is almost twice the input payload, consistent with investigating result transfer first.

**Lossless experiments:**

1. Compare current mapped-host loads/stores with VRAM staging plus asynchronous DMA for sufficiently large contiguous chunks.
2. Partition computation so one token chunk can complete gate/up, gated quantization, down, and reduction while another chunk transfers or computes.
3. Overlap independent sessions or microbatches where possible, rather than assuming the next layer’s true routes are available early.

The tradeoff is expert reuse: slicing a request into small token chunks can make the GPU reload and decode the same expert weights repeatedly. Select chunk size using expert-run histograms, not link bandwidth alone.

**External evidence:** SGLang’s expert-parallel documentation describes two-batch overlap and shared-expert/communication overlap. MoE-Lightning §§3–4 models competing compute and transfer resources and schedules paged transfers. These are useful scheduling references; their published gains do not predict gains on this host. [SGLang overlap sections](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx#L200), [MoE-Lightning §§3–4](https://arxiv.org/html/2411.11217v1)

Sending results as fp8/int8 could approximately halve payload bytes, but introduces quantization after weighted reduction. It also changes the CPU combine interface and scale handling. Existing local evidence already shows sensitivity to combine precision: `worker_combine.h:5–7` records why fp16 combined rows replaced bf16 sums. Treat compressed outputs as a separate high-risk numerical experiment.

**Rank 5 — CPU execution for selected cold host experts**

The strongest application is an expert with very few routed rows whose weights would otherwise traverse the engine’s x4 link. It is less compelling for a heavily reused prefill expert.

**Verified externally:** Fiddler explicitly uses CPU/GPU orchestration, but its repository warns that CPU expert execution is slow without AVX512. KTransformers’ AMX documentation switches between AMX and AVX512 according to arithmetic intensity. Their reported performance should not be transferred to this i7 host. [Fiddler limitations](https://github.com/efeslab/fiddler#known-limitations), [KTransformers AMX/AVX512 design](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)

**Inference:** benchmark an AVX2-compatible implementation against actual engine host-pool service time as a function of rows per expert. Include contention with CPU combine and embedding-table access. Prefer host-pool experts initially: duplicating tail weights in RAM requires memory that is already constrained.

A dynamic CPU/GPU threshold should depend on measured completion time, not “coldness” alone. CPU execution also needs compatible rotation, quantization, activation, and reduction arithmetic.

PowerInfer supplies the broader idea of placing frequently activated work on GPU and computing cold work on CPU. Its neuron-level sparsity assumptions do not directly establish a solution for this routed-expert model. [PowerInfer](https://arxiv.org/abs/2312.12456)

**Rank 6 — BAR push remains a bounded prefill optimization**

**Verified from the supplied lab journal:** the November 10 R9700 measurements report 959.0 versus 770.4 µs serial pull/push latency at 768 rows, a 188.6 µs saving. One row slightly regresses; 16 rows saves about 6 µs. See `PROGRESS.md:306–324` under the supplied RAD50 lab directory.

The important distinction is between a faster isolated transfer/combine sequence and a shorter serving critical path. Current CPU combine already overlaps helper result publication. Replacing an overlapped pull can save much less than the serial delta.

Continue only with the production-equivalent consumer, real chunk sizes, concurrent helper traffic, and paired inference measurements. Preserve the fp16 overflow fallback and generation/lifetime rules. This does not address the 6900 XT’s gate/up bottleneck.

**3. Which other kernel systems are applicable?**

| Source | Verified technique or capability | Relevance here |
|---|---|---|
| **llama.cpp HIP MMQ/MMVQ** | Nonlinear-table expansion, Q8 activations, integer dot4, architecture-specific dispatch | Best direct reference for an RDNA2 integer-dot prototype |
| **vLLM fused MoE** | Expert-sorted blocks, grouped scheduling, quantized GEMMs | Borrow scheduling; its affine W4 handling does not implement this codebook |
| **SGLang** | Grouped-GEMM backends, overlap, placement and replication | Useful orchestration patterns; backend availability does not establish consumer-card suitability |
| **Marlin** | Offline layouts, asynchronous loading, double buffering, carefully scheduled dequantization | Useful design methods; implementation requires NVIDIA compute capability ≥8.0 |
| **ExLlamaV3** | Consumer-GPU MoE execution and CPU offload; documented CUDA requirements | Useful comparison, without verified drop-in gfx1030 support |
| **AMD CK / hipBLASLt** | Grouped and quantized GEMM infrastructure | Potential reference or comparison path after format conversion; no verified native W4NL64/E4M3 solution |
| **rocWMMA** | Matrix instructions on listed gfx11/gfx12 targets | Applicable to faster cards where appropriate; its supported RDNA list excludes gfx1030 |

The specific sources are [Marlin’s techniques and requirements](https://github.com/IST-DASLab/marlin), [ExLlamaV3’s installation and capabilities](https://github.com/turboderp-org/exllamav3), [CK changelog](https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md), [hipBLASLt grouped GEMM API](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.0.2/reference/ext-reference.html), and [rocWMMA supported architectures](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html#supported-gpu-architectures).

Calling a library “grouped GEMM” does not make it suitable for GPUs without matrix instructions. Nor does support for linear INT4, MXFP4, or FP8 imply support for this nonlinear codebook and scale placement. Expanding all resident W4 weights to fp16 would approximately quadruple their code storage and defeat the present capacity objective.

**4. Offload and prediction: useful ideas, constrained applicability**

**MoE-Infinity** uses sequence-level activation traces for prefetching and caching; §§4–6 are the relevant sections. This supports collecting request-specific locality rather than relying only on global heat. Here, most helper weights are already resident, so prediction is useful primarily for host-pool caching or migration. [MoE-Infinity §§4–6](https://arxiv.org/html/2401.14361v1)

**HybriMoE** adds score-aware caching and impact-driven prefetching. The transferable principle is to prioritize transfers by predicted execution-time saving, with failed predictions charged against available bandwidth. [HybriMoE §§IV-C–D](https://arxiv.org/html/2504.05897v1)

**Pre-gated MoE** introduces a pre-gating function as an algorithm/system co-design. It should not be treated as a transparent runtime optimization for the existing checkpoint. A predictor used only for speculative prefetch can preserve routing, provided misses fall back correctly; replacing true routing changes model behavior. [Pre-gated MoE](https://arxiv.org/abs/2308.12066)

**llama.cpp tensor overrides** provide tensor placement and CPU-MoE controls, plus a documented expert-cache option. These are useful comparisons, but tensor placement controls alone do not implement this protocol’s per-expert heterogeneous ownership. [CLI memory and placement options](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)

**DeepSpeed-MoE** distinguishes expert parallelism from expert slicing. That supports keeping complete experts on helpers where possible. Its distributed results do not contradict poor tensor-parallel performance over host memory on this machine. [DeepSpeed-MoE inference design](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/)

**5. Measurement and conclusion corrections**

**A. Reconcile model dimensions with memory occupancy.**

Using the brief’s geometry, each expert contains:

`3 × 2560 × 384 = 2,949,120 weights`

At four-bit codes plus one byte of scale per 64 weights, that is approximately **1,520,640 bytes per expert**. Across 49 layers and 512 experts, the routed-expert payload is approximately **35.53 GiB**.

That does not reconcile with 44 GiB of head experts, 14 GiB of tail experts, and another 9 GiB in the host pool, before engine-resident experts.

Possible explanations include budgets being reported as occupied weight bytes, different actual geometry, additional retained encodings, or duplicated storage. This is an accounting inconsistency in the supplied description, not proof of an allocation bug. Resolve it before deriving placement limits or declaring that every residency option is exhausted.

**B. “2.3 TFLOPS versus 46” identifies headroom, not attainable speedup.**

The useful arithmetic count is consistent with the stated geometry: approximately 5.90 MFLOP per routed expert row. But nominal fp16 throughput excludes codebook decoding, LDS access, shuffles, scale operations, padding, synchronization, and instruction dependencies.

The kernel also computes padded rows and performs work whose useful result is retained only by quad lane zero. Report useful and executed arithmetic separately. A low useful-FLOP rate does not prove a compute-unit saturation problem or imply a 20× opportunity.

**C. Apply Amdahl’s law to cold TTFT.**

If tail busy time is 73% of wall time and GEMMs are 71% of that time, GEMMs account for approximately 52% of wall time under a simplifying critical-path assumption.

That yields these conditional ceilings:

| GEMM speedup | Approximate cold-TTFT reduction |
|---:|---:|
| 1.5× | 17% |
| 2× | 26% |
| 3× | 35% |

Actual gains can be lower because busy time is not necessarily exposed critical-path time. The unchanged 27k-token workload would reach approximately 8.4 seconds under the idealized 2× case.

**D. Separate transfer bytes from reduction time.**

`lane_reduce_tokens_v` performs VRAM reads, arithmetic, mapped stores, and store draining. Its runtime is not a pure PCIe bandwidth measurement.

For example, writing all 2048 bf16 rows would transfer 10.49 MB; doing that in 1.835 ms would imply 5.71 GB/s, exceeding Gen3 x4’s approximately 3.94 GB/s encoding-level ceiling. Sparse output can resolve this: record the actual distinct-token count and bytes written for each timed request.

Similarly, the first dispatch timing stage includes activation preparation and route work (`moe_remote_gpu.h:654–734`, `1089–1095`). Keep that stage distinct from a profiler measurement of the activation-copy kernel alone.

**E. The KL evidence requires a qualified conclusion.**

The findings establish that the current helper configuration fails the stated p99.9 gate and that changing engine batch size can reproduce the sensitive-token spike. They also narrow the discrepancy toward per-token combination rather than router weights.

They do **not** establish that every helper difference is harmless or that a new quantization path should pass. Preserve the existing gate and document its batch sensitivity. Compare candidates both against the reference and against the current helper baseline, using identical batching and reference generation.

**F. “Speed-aware placement cannot help” is too strong.**

The evidence supports retaining the current default over the tested smaller-tail configuration for sustained multi-session throughput. It does not prove optimality of the current per-layer placement. The inspected live planner cannot answer that question because it optimizes tail heat share among helper-owned experts.

**G. External throughput comparisons remain unmatched.**

I did not find a reproducible, directly comparable **RX 6900 XT HIP IQ4_NL `mul_mat_id` prefill** result for these dimensions and routing distributions.

The llama.cpp HIP scoreboard’s 6900 XT result—about 1948 prompt tokens/s at commit `a972fae`—is explicitly **dense Llama-2-7B Q4_0 with Flash Attention**. It cannot validate a MoE tail-throughput prediction. [Benchmark model, commit, and table](https://github.com/ggml-org/llama.cpp/discussions/15021)

Keep the current four-card layout as the serving baseline. The most concrete first kernel experiment is to transplant the existing exact register-permute decoder into the accurate quad path, then evaluate per-expert row scheduling. In parallel, an offline placement simulation should determine whether per-layer imbalance offers enough recoverable time to justify a live-policy change.