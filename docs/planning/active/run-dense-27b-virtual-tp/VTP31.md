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

## Experiment 1 result: launch geometry (2026-11-12, build f8406b8e, logs vtp14/logs/geo-0757)

`RADIANCE_TUNE=all_reduce:nblocks=N,nthreads=M` on the existing TP3 exact all-reduce (no source change), MXFP4, N7
split, depth 3. Six arms, run forward then reversed; mean of the two passes. Prefill is the worst TTFT of four
4096-token prompts.

| grid | 1 stream | 4 streams | 8 streams | prefill TTFT | vs default (4 / 8 / prefill) |
|---|---|---|---|---|---|
| default (scaled, 1024 threads) | 106.0 | 178.5 | 188.8 | 11.14 s | |
| 2 x 256 | 107.6 | 185.2 | 198.6 | 10.51 s | +3.7% / +5.1% / -5.6% |
| **4 x 256** | 107.7 | 189.9 | 197.6 | 10.52 s | **+6.4% / +4.6% / -5.6%** |
| 8 x 256 | 106.7 | 181.8 | 195.9 | 10.95 s | +1.8% / +3.7% / -1.7% |
| 8 x 1024 | 106.5 | 181.3 | 193.7 | 11.21 s | +1.6% / +2.6% / +0.6% |
| 16 x 1024 | 105.7 | 178.1 | 192.4 | 11.13 s | -0.3% / +1.9% / -0.2% |

Greedy output identical on every arm. Fewer, narrower blocks win at every size: the cost is polling and occupancy,
not payload (GPT's first hypothesis holds). The default grid is tuned for the two- and four-rank cases. Next: a 3x ABBA
of TP2, TP3 default and TP3 4x256 (`n8-abba.sh`), then make the grid a typed per-run setting (step 1) rather than an
environment pin, and try a three-rank narrow default in the selection row.

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

### GPT analysis (2026-11-12, from the six profwhere logs and the patched source)

Corrected accounting: `prof_split.py` counts TP2's fused `ar_rmsnorm_quant_fp8` as compute; counted as a collective,
the per-rank operation totals (ms, cumulative device events, not wall time) are:

| workload | rank | collectives | other | lane idle |
|---|---|---|---|---|
| TP2 dec8 | XTX 0 / 1 | 2239 / 2176 | 4820 / 4855 | 2288 / 2320 |
| TP3 dec8 | XTX 0 / XTX 1 / R9700 | 3941 / 3990 / 4781 | 3236 / 3205 / 2767 | 2801 / 2786 / 2368 |
| TP2 pre4 | XTX 0 / 1 | 5357 / 5281 | 5917 / 5945 | 316 / 379 |
| TP3 pre4 | XTX 0 / XTX 1 / R9700 | 8774 / 8904 / 9487 | 3727 / 3611 / 3399 | 599 / 588 / 194 |

- TP3 pre4 is 74% collective on the R9700 (TP2 pre4: 48%). The profile's `wait us` is lane idle before an op starts;
  waiting for a peer inside an all-reduce is part of that kernel's time, so the profile cannot separate peer wait,
  flag polling and transfer.
- Overlap reaches only some steps: TP3 pre1 has 1280 of 1738 ARs at `overlap=1`, pre4 only 256 of 1624, because
  `step_pipelined` refuses `n_seq != 1`. Multi-sequence prefill is unserved by rad0020.
- The three-rank selection has no narrow lane row (two- and four-rank have `tArLane`), so the TP3 pipeline's lane-1 AR
  uses the full-width grid beside the compute.
- Link floors: a 32-row decode AR is 320 KiB; one-shot puts 2S each way on the R9700's x4 (about 83 us per direction
  at 7.88 GB/s) against a measured 440-570 us per call, so most of a decode AR is not the payload. A 2048-row prefill
  AR (20 MiB, weighted two-shot 4,4,1) moves 22 MiB on the R9700 (about 3.0 ms floor) and 29 MiB per XTX (about
  1.9 ms) against 5.7-6.3 ms measured: bandwidth is a large part.
- Per-kind GEMM time: decode XTXs carry more attention and FFN time than the R9700 (the R9700 is not the slow compute
  rank); pre1 the R9700 is the slowest (FFN 2578 ms); pre4 FFN is heavier on the XTXs and delta-net on the R9700.
  The kinds pull different ways, so the split stays per kind (VTP32); expected gain from more rebalancing is 0-3%.

Ranked proposals (gains are hypotheses):

1. Launch geometry of the existing TP3 AR through `RADIANCE_TUNE=all_reduce:nblocks=N,nthreads=M` (no source change;
   exact by construction). 0-8% decode.
2. A tunable three-rank one-shot/two-shot crossover (today compiled at 524288 elements). 0-10% on mid-size messages.
3. Multi-sequence pipelined prefill (generalise `ArPieces` and `step_pipelined`; every rank must decide the same).
   5-15% pre4 TTFT, speculative.
4. A three-rank fused AR + RMSNorm + fp8 quant. 0-3%: the separate norm kernel is about 106 ms of 12.5 s in pre4.
5. GEMM epilogue publishing tiles to host staging (FLUX-style). 0-12% prefill, high regression risk.
6. CPU-assisted exact reduce for short messages: halves the R9700's one-shot traffic (2S to S) but adds a CPU
   wake/sum/publish boundary and must reproduce the GPU's fp32 order and bf16 rounding. 0-20%, highly uncertain.
7. An LL-style flag protocol only after phase timings show polling dominates; never drop release/acquire.
8. Per-kind ownership refinement (VTP32).

Not prioritised: decode token splitting (8 rows into 2x4 doubles the AR boundaries), algebraic AR removal, three-rank
lossy wire. Method selection belongs in a typed `tuning.allreduce_method` (`auto` keeps Radiance's selection; explicit
methods fail closed), with the selected row recorded per size band in the run receipt.

First two experiments: (1) the launch-geometry sweep above; (2) optional per-phase GPU timestamps in the exact one-shot
and two-shot (push done, peer flags seen, local reduce done, phase-two arrival, assembly) over 8-64 row, crossover-size
and 1024/2048-row messages, to decide which term (peer wait, polling, payload, reduce, idle CUs) the rewrites must beat.

Evidence and scripts: ${BC_TASK_TMP}/vtp14/ (n7-abba.sh, prof-where.sh, ts3w-ab.sh, lossy3.sh, prefill-knobs.sh, notes-allreduce-analysis.md, p2p-finding.md). Code: libr4d r4d_ar_entry.hip, r4d_ar_oneshot_Nrank_exact.hip, r4d_ar_twoshot_Nrank_exact.hip, r4d_ar_seq.h, r4d_rows.cpp (selection rows and host twins); BigCherry patches rad0016-rad0021.

## Change Log

- 2026-11-12T07:38:50.011623+00:00 (created-by): Created by agent
- 2026-11-12T07:39:30.275584+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
