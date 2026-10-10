---
id: MEN01
order: 1
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:00.311928+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Radiance inventory: build it on Brutus, what runs on which card, where the speed comes from

## Description

Gate for the group. Owner decision 2026-10-09: the second engine is standalone radiance (codeberg.org/StillDeadcode/radiance, Apache-2.0), not vLLM plus the radiance layer. Radiance is a C++/HIP inference server with a CMake build, an HTTP API that speaks OpenAI, Anthropic and llama-server's native protocol, and kernel libraries loaded as .so plugins through a C ABI: libr4d (gfx1200 / gfx1201 only), libref, libavx, libquant. The README says the engine starts on any AMD GPU family but another card needs a kernel library for it. Reference numbers to explain (one R9700, 4-bit Qwen3.8-27B, 2026-10-09): the vLLM + radiance container ~3,000 t/s prefill to 35K and ~2,470 at 111K; BigCherry llama.cpp on the same card 1,003-1,089 with MTP, 1,141-1,291 with no drafter, 764-857 at 111K.

## Steps

1. Clone at a recorded commit; read LICENSE, NOTICE, spec.md and the abi/ and arch/ directories; record what each kernel library covers and for which architecture.
2. Build on Brutus with the lab ROCm (/mnt/vault/tmp/bc-rocm) for gfx1201; record the exact commands and any change needed.
3. State per card what can run today: R9700 (libr4d), RX 7900 XTX and RX 6900 XT (libref only, or nothing).
4. Checkpoint formats it loads (MXFP4 from AMD Quark, FP8, bf16, int8) and which of our models exist in one of them; the container already has /models/Qwen3.8-27B-MXFP4-mtpfp8 and a DFlash2-FP8 drafter.
5. rocprofv3 kernel census of one 24K prefill, grouped like prefill-kernel-table.py, beside the llama.cpp census.
6. Write the finding: which kernel family carries the lead, and what is card-specific.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A build that starts and answers /health on Brutus; a component table with architecture coverage; one kernel census at 24K beside ours.

## Effort & Risk



## Standards



## Acceptance Criteria

Radiance builds from source on Brutus at a recorded commit, and the inventory states what runs on each of the four cards and which kernel family explains the prefill lead.

## Notes

Read-only on the container: start, measure, stop. No configuration change. Production llama-swap docker stays off.

2026-10-09 source read (no build yet). Clone on Brutus: /mnt/data/bigcherry-work/engines/radiance at 89cee7ce414486c717aa2085c99e1f522e4c0ea6 (radiance 1.3.0, 2026-10-08), Apache-2.0 with a NOTICE file. Findings from README and tree:
- Kernel libraries: libr4d (103 files of HIP kernels; built for gfx1200 / gfx1201 only), libref (reference implementations: attention, collective, GEMM, GDN, HC, MoE), libavx (CPU host backend; the README says it runs the test suite but does not serve at useful speed). No gfx1100 or gfx1030 kernel library. docs/PLUGIN.md documents writing a kernel library for another AMD card (its example names gfx1100).
- Architectures (arch/): qwen4exp_fp8 (Flash-Next), qwen35_fp8, qwen35_bf16, qwen35moe_fp8, llama_fp8, llama_dense_fp8, embgemma2.
- Flash-Next is a shipped model: qwen3.8-next-flash-fp8-iq4r-moe.rad (114 GiB; 4-bit experts, int8 trunk, MTP, vision), served with --tp 2 (also 3 and 4 ranks, docs/TP3.md) --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --kv-cache-dtype fp8 --num-speculative-tokens 3. The README's serve flags are for two 32 GB R9700 cards. Qwen3.8-27B ships as FP8 (29 GiB) and MXFP4 (19 GiB) with a DFlash2 drafter.
- Tensor parallelism is built in (--tp N, exact and lossy wht6 / tiered-int8 all-reduce wires; libr4d has 2-rank and N-rank all-reduce kernels). Every rank needs a card the kernel library covers, so on Brutus only the single R9700 qualifies: --tp 1 only, until a gfx1100 kernel library exists.
- Quantisers (libquant): AWQ, AutoRound, and a `ggml` quantiser that runs llama.cpp's own quantize_<type> and unpacks the GGUF blocks into radiance's planes (q8_0, q4_0, q4_K, q5_K, q6_K, iq4_nl, iq4_xs, iq2_xxs, ...), so codes and scales are exactly llama.cpp's. rad-convert turns a Hugging Face checkpoint into a .rad container; docs/qwen3-8-flash-next-gguf-quant-recipes.md discusses the GGUF recipes.
- Build from source: C++/HIP, CMake + Ninja, ROCm 7.2 hipcc (ROCM_PATH), ./build.sh or cmake -DRAD_GPU_TARGETS=gfx1201; ctest labels gpu / oot. Brutus has /opt/rocm-7.2.4, so the source build should be possible there without the lab's own ROCm tree.
Consequences: (1) the like-for-like Flash-Next comparison needs two RDNA4 cards; with one R9700 only --tp 1 with expert tiering to host memory can be tried. (2) The route to using the XTXs is a gfx1100 kernel library plugin, which is the natural first patch-or-plugin project for MEN07. (3) The ggml quantiser means our GGUF quant choices can be carried over without re-deriving them. Still to do: the build, the per-card run check, and the kernel census.

## Change Log

- 2026-10-08T20:36:00.311928+00:00 (created-by): Created by agent
- 2026-10-08T20:47:42.172651+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria

## Ledger-events

- chg_20261008_212949_added-a-like-for-like-single-c_9270
- 2026-10-08T21:30:02.720551+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-08T22:51:23.303074+00:00 (updated-by): Updated: section:notes
