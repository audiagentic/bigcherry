---
id: VTP31
order: 31
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-12T07:38:50.011623+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# TP3 all-reduce: latency, compute/communication overlap and swappable methods

## Description

Native TP3 (XTX, XTX, R9700; VTP14) loses to TP2 only on the all-reduce. Every collective is host-staged, and per call TP3's costs about twice TP2's. This item makes the all-reduce cheaper and overlaps it with compute, and makes the method selectable per run so alternatives can be A/B'd from the manifest.

## Results so far (2026-11-12, MXFP4 6/6/12, exact wire, DFlash2 depth 3)

**Per-op profile** (`--profile-ops`, logs `vtp14/logs/profwhere-0723`). TP2's figure includes its fused `ar_rmsnorm_quant_fp8`. The fused AR norms refuse world != 2, so TP3 runs `all_reduce` and then `rmsnorm_quant_fp8`.

| workload | TP2 all-reduce | TP3 all-reduce | TP3 compute per rank (TP2) |
|---|---|---|---|
| 8-stream decode | 2.24 s, 0.32 ms a call | 3.9-4.8 s, 0.55-0.67 ms a call | 2.8-3.2 s (~4.2 s) |
| four 4096-token prompts | 5.36 s, 3.3 ms a call | 8.8-9.5 s, 5.8 ms a call | 3.4-3.7 s (~6 s) |

- A decode message is ~320 KB, which Gen4 x8 moves in ~40 us. So ~0.6 ms a call is latency (per-block flag round trips over host memory) and waiting for the slowest rank of each layer, not bandwidth.
- There are 129 all-reduces per decode step.

**Matched N7 ABBA x3** (unprofiled), TP3 vs TP2 aggregate tok/s with speculation:

| streams | change |
|---|---|
| 1 | +28.8% |
| 4 | -1.1% |
| 6 | +1.6% |
| 8 | -4.4% |

The decode step is faster at every concurrency. Prefill TTFT is +11 to +29%. Without speculation (N0), TP3 was +13-17% at 4/6/8.

**Already in place** (VTP14):
- rad0020: the pipelined prefill, one sequence of 1024+ rows only; worth 15-19% of TTFT.
- rad0021: link-weighted three-rank two-shot (weights 4,4,1); TTFT -11% on a 4096-token prompt.
- Lowering the pipelining floor to 512 rows and using 4096-token chunks: no effect. `--tp-attn-split rows` is unsupported by qwen35_fp8.
- The three-rank lossy tiered-int8 wire is 30% slower than exact at 8 streams (no fault in that run; still refused by default).

**Platform.** No GPU peer-to-peer: the driver reports none and KFD has no p2p links, although Resizable BAR is on (32 GiB BAR0), the IOMMU is `iommu=pt` and amdgpu `pcie_p2p=Y`. Alder Lake i7-12700KF, host bridge 8086:4668; each card on its own CPU root port: XTX x8, XTX x8, R9700 x4. Linux allows P2P DMA between root ports only through allow-listed host bridges. This is a host kernel matter; host staging stays.

**Survey of other engines:**
- vLLM custom all-reduce: one-shot/two-shot by size, 16-byte vectors.
- AMD QuickReduce: quantized two-shot; slower than custom below ~512 KB.
- NCCL LL/LL128: flag inside the data word, no separate fence and flag round trip. LL128 needs 128-byte store atomicity, which PCIe lacks; LL is safe.
- TokenWeave and NanoFlow: split the tokens in two and overlap one half's all-reduce with the other half's compute.
- Collective matmul / FLUX: chunked or fused GEMM and communication.
- Kraken: the all-reduce on a dedicated stream beside the QKV GEMM.

**GPT deep analysis** of where the time goes and which options to try: request `req_vtp14-n1-rank-code-eligi_cb7d41d06315` (gateway); results go into Notes.

## Steps

1. Add a typed per-run selector for the all-reduce method (e.g. `tuning.allreduce_method`; contract amendment; engine option in a managed patch) so every method below is an A/B from the manifest.
2. Quantify the latency and per-layer wait split from the profile's wait column. Read GPT's analysis and pick the first two experiments.
3. Overlap in decode and multi-sequence prefill: generalise rad0020's two-piece pipeline (split the step's rows in two; one piece's all-reduce on lane 1 beside the other's compute), as TokenWeave and NanoFlow do.
4. A three-rank fused all_reduce + RMSNorm + fp8 quant, as the two-rank one, so TP3 drops the extra pass.
5. A low-latency handshake for small decode messages: flag-in-data (LL-style, 8-byte data+flag units) and fewer blocks, so a message costs one host round trip.
6. A CPU-assisted exact reduce: each GPU writes its partial once to pinned host memory, CPU threads sum in fp32 ascending rank order (reuse the helper-row combine pattern from moe-remote), and each GPU reads one plane. Use SDMA copies so transfers run beside compute.
7. Push from the producing GEMM's epilogue into host-mapped scratch (FLUX-style), removing the separate push pass.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Per-method ABBA at 1/4/6/8 streams and 512/2048/8192-token prompts (n7-abba.sh pattern).
- `--profile-ops` all-reduce ms per call before and after.
- Greedy identity on every arm.
- KLD against BF16 for any lossy option.

## Effort & Risk



## Standards



## Acceptance Criteria

- Each method is selectable per run from the manifest, and its default leaves today's selection unchanged.
- Every exact method gives bit-identical greedy output against the current exact wire.
- A matched ABBA (at least 3 repeats) shows TP3 at or above TP2 aggregate tok/s at 4, 6 and 8 streams with speculation on, and prefill TTFT gap reduced, with the change attributed per method.

## Notes

Evidence and scripts: ${BC_TASK_TMP}/vtp14/ (n7-abba.sh, prof-where.sh, ts3w-ab.sh, lossy3.sh, prefill-knobs.sh, notes-allreduce-analysis.md, p2p-finding.md). Code: libr4d r4d_ar_entry.hip, r4d_ar_oneshot_Nrank_exact.hip, r4d_ar_twoshot_Nrank_exact.hip, r4d_ar_seq.h, r4d_rows.cpp (selection rows and host twins); BigCherry patches rad0016-rad0021.

## Change Log

- 2026-11-12T07:38:50.011623+00:00 (created-by): Created by agent
- 2026-11-12T07:39:30.275584+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
