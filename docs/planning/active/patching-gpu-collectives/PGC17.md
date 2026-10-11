---
id: PGC17
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-11T13:45:17.976093+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Cross-card transport on Brutus: peer-to-peer, PCIe atomics and RCCL version - what two community reports say and what to test

## Description

Owner, 2026-10-12: capture two community write-ups ('not the same as our setup'). Why it matters here: the cross-card sum is about 36% of each target card's prefill time on Flash-Next (0.9-1.4 ms a sum; QFP49) and most of the non-kernel time in decode; a faster transport needs no patch and pays no half-batch tax.

REPORT 1 (Reddit, 2x Radeon AI PRO R9700, Threadripper TRX50, both cards on CPU lanes, Arch, kernel 7.3-rc6, ROCm 10.1, RCCL 2.30.7, SGLang). (1) Kernel CONFIG_HSA_AMD_P2P=y: without it RCCL falls back to host-staged transfers; rocprofv3 on a TP=2 decode: median collective call about 1,001-1,020 us before, 12.7-16.2 us after (60-80x); five sixths of the decode interval had been communication; afterwards TP=2 was 15.8% faster than TP=1; logs select 'P2P/IPC' both directions; no ACS / IOMMU change needed; says nothing about remote atomics or bandwidth. (2) HIP hardware-queue serialisation (vLLM issue #58639, R9700, ROCm 7.14): a stream of small kernels pays about 17 us a dispatch when two or more other HIP streams hold pending waits on the dispatching stream, in graph mode, under the default hardware-queue limit: forward 16.9-17.0 ms with side streams against 7.18-7.23 ms with everything on the main stream; GPU_MAX_HW_QUEUES=1 or 2 removed it in their microbenchmark but not fully end to end. (3) Decode throughput on the R9700 reported bimodal per process start (ROCm #6347: about 33 or 26 tok/s, stable within a process) - bench over several fresh starts. (4) SGLang / Triton specifics, not our code: attention KV splits 16 against 64 compute units (2.14x at 256K decode in mattbucci's port); FP32 P.V in grouped decode attention costs occupancy (bf16 products with FP32 accumulation +21% at 256K); gfx1201 missing from AITER's arch table so FP8 weights are silently dequantised (closed by AMD: no plan to support the R9700). (5) Roofline: R9700 640 GB/s, 64 CUs; ceiling tok/s = bandwidth / bytes read a token.

REPORT 2 (dual R9700, one card behind the chipset, Proxmox, image stilldeadcode/vllm-radiance:0.9.3, vLLM 0.27.1, PyTorch 2.11 + ROCm 7.14, RCCL replaced by 2.27.7 from ROCm 7.1.1). Result: Qwen3.8-27B FP8, TP=2, 262,144 context, 422,964-token KV pool, about 1.6k t/s prefill, 74-143 t/s decode. A card behind the chipset cannot be given PCIe atomic operations (the chipset does not forward them); RCCL 2.30.4's kernels use atomics, so communicator init fails with 'PCIE atomic ops is not supported / rocr: unhandled cuda error'; RCCL 2.27.7 works. With no peer path the image's own peer-to-peer all-reduce must be off (RADIANCE_USE_R4D_AR=0) and TP falls back to host-staged RCCL over the chipset link; with two CPU-attached cards leave it on for a better prefill. Settings used: NCCL_PROTO=Simple, NCCL_SHM_DISABLE=0, NCCL_SOCKET_IFNAME=lo; RADIANCE_FUSE_RMS_QUANT=1, RADIANCE_USE_R4D=1, RADIANCE_USE_R4D_AR_QUANT=1; --max-num-seqs 2, --max-num-batched-tokens 4096, --kv-cache-dtype fp8, --enable-prefix-caching, --no-async-scheduling; cache directories mounted so Triton / inductor kernels are not recompiled each start. A shim exporting ncclCommDump satisfies newer torch against the older RCCL. Startup log line to look for: 'P2P access : DISABLED (RCCL fallback) 0<->1 x'. Proxmox: q35 machine, PCI-Express ticked, All Functions unticked per GPU, stop/start the VM after a hostpci change; the CPU-attached card then shows ReqEn+.

BRUTUS FACTS (checked 2026-10-12): kernel 7.0.13-070013-generic has CONFIG_HSA_AMD_P2P=y, CONFIG_PCI_P2PDMA=y, CONFIG_HSA_AMD_SVM=y, CONFIG_DMABUF_MOVE_NOTIFY=y. RCCL is 2.27.7 (ROCm 7.2.4) - the version report 2 had to go back to. Cards: 2x RX 7900 XTX on CPU x8 links (03:00.0, 06:00.0), R9700 on a x4 link (09:00.0), RX 6900 XT on the chipset (17:00.0, x4 at 8 GT/s). tools/lab/rccl/README.md, 2026-10-02: on this stock kernel 'P2P is off (RCCL uses SHM)'; on a patched kernel XTX-to-XTX direct P2P ran at only 0.9 GB/s against 13-14 GB/s host-to-GPU. PCIe AtomicOps capability per card not readable without root. The same radiance image (0.9.3) is on Brutus; whether its own all-reduce (RADIANCE_USE_R4D_AR) is on in our two-XTX runs is not recorded. Our Radiance fp8 27B on two XTXs: 48 t/s decode, about 1,185 t/s prefill - well under report 2's R9700 figures (the R9700 has native FP8 matrix instructions; the XTX uses the f16 widening path).

ALREADY CONSISTENT WITH OUR OWN MEASUREMENTS: the per-dispatch cost with a second stream active (we measured about 6 us a launch, staggering hid 84% of a sum at 2 kernels a step and 13% at 80; GPU_MAX_HW_QUEUES 1 / 2 / 8 did not help, 1 was worse) and the bimodal runs (1360 on: about 880 or 1,090 t/s with identical settings).

## Steps

1. RCCL transport per card pair on Brutus as it is: NCCL_DEBUG=INFO with the INIT / TRANSPORT / P2P / SHM subsystems on tools/lab/hip-probes/rccl_stagger_probe (queued as run p2pdbg): which transport each of XTX-XTX, XTX-R9700 is given and the reason printed. 2. Latency of a small (decode-sized) and a large (prefill-sized) sum per pair and per transport, against report 1's 13-16 us. 3. Why P2P is off on the stock kernel although the config flag is set: peer-access check (tools/lab/rccl p2p-check), PCIe AtomicOps capability per card (needs root: owner), resizable BAR, IOMMU / ACS state, and whether the 0.9 GB/s of the patched kernel was a BAR-mapping artefact. 4. Two-XTX-only test (both CPU-attached, same generation) before anything involving the R9700 or the chipset card. 5. Radiance: read the startup log for the 'P2P access' line on our two-XTX runs and compare prefill with RADIANCE_USE_R4D_AR on and off. 6. Bring the owner the specific host change if one is needed (slot move, BIOS setting, kernel, ROCm version): host changes are the owner's.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A table: transport chosen and sum latency (small and large) for each card pair, before and after any change; a prefill and decode A/B on the production profile if a faster transport is obtained; bench over several fresh server starts (bimodality).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-11T13:45:17.976093+00:00 (created-by): Created by agent
