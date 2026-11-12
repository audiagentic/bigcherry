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
not payload (GPT's first hypothesis holds). The default grid is tuned for the two- and four-rank cases.

### Confirmed: 3x ABBA, TP2 against TP3 default and TP3 4x256 (2026-11-12, logs vtp14/logs/n8-0818)

Same build and settings; per repeat TP2, TP3 4x256, TP3 default, TP3 default, TP3 4x256, TP2 (six runs an arm). Aggregate
tok/s at 512-token prompts, 128 new; TTFT is one stream.

| | 1 stream | 4 streams | 6 streams | 8 streams | TTFT 512 / 2048 / 8192 |
|---|---|---|---|---|---|
| TP2 (two XTX) | 83.0 | 182.0 | 176.9 | 196.1 | 320 / 859 / 2880 ms |
| TP3 default grid | 105.9 | 176.4 | 177.4 | 188.5 | 359 / 1156 / 3413 ms |
| TP3 4x256 | 107.8 | 188.2 | 183.6 | 199.3 | 347 / 1068 / 3023 ms |
| 4x256 vs TP2 | +29.9% | +3.4% | +3.8% | +1.7% | +8.6% / +24.3% / +5.0% |
| 4x256 vs default | +1.7% | +6.7% | +3.5% | +5.7% | -3.5% / -7.6% / -11.4% |

Run-to-run spread is 1-2 tok/s. **TP3 is now above TP2 at 1, 4, 6 and 8 streams.** Prefill TTFT still trails TP2
(the gap at 8192 tokens fell from +17% to +5%).

The grid is a manifest setting: rad0022 adds `--tune-pin SPEC` (radiance's `RADIANCE_TUNE` text and checks), the
contract field `tuning.tune_pin` (amendment 20), and the `qwen27b-tp3` layout sets
`tuning.tune_pin = "all_reduce:nblocks=4,nthreads=256"`. The int8 layout is not measured with it yet. Next: a three-rank
default in the selection row is not needed while the layout carries it; chained launches for the three-rank rows (the
four-rank rows have them, r4d_chain.h) and the hierarchical exchange around the x4 link.

### int8 (w8a8) TP3 with the grid (2026-11-12, build 2cb05432, logs vtp14/logs/n9-0912)

2x ABBA, `qwen27b-tp3-int8` split, `--tune-pin all_reduce:nblocks=4,nthreads=256` against the default grid:

| | 1 stream | 4 streams | 6 streams | 8 streams | TTFT 512 / 2048 / 8192 |
|---|---|---|---|---|---|
| default grid | 82.8 | 176.7 | 170.1 | 195.2 | 342 / 1309 / 4053 ms |
| 4x256 | 83.8 | 180.6 | 172.2 | 205.0 | 329 / 1274 / 3764 ms |
| change | +1.1% | +2.2% | +1.2% | +5.1% | -3.9% / -2.7% / -7.1% |

The int8 layout now sets it too.

### Around 4x256: no further gain (logs vtp14/logs/geo2-0927)

Against 4x256 (mean of a forward and a reversed pass): 4x512 -1.7% / -2.3% at 4/8 streams, +4.2% prefill TTFT; 2x512
-0.7% / -0.7%; `drain=1` -0.8% / -0.6%; `acq=1` +2.7% / -0.1%. All within run-to-run spread; output identical. The grid
and fence axes are exhausted; the next step is the phase breakdown (rad0023, experiment 2).

### Corrections to the kernel plan

- **Chained launches:** not for this model. Radiance measured no gain from them for Qwen3.8-27B with DFlash2 (4x
  R9700), so its plugin leaves them off (`chained launches: off` in our logs), and the host-mailbox `_host` rows TP3
  runs are not chainable.
- **Routing around the x4 link:** prefill already moves the minimum over it (rad0021: the R9700 owns no two-shot shard,
  N each way). The decode one-shot sends 2N each way, but a 32-row message is ~320 KiB, about 83 us at x4, against
  440-570 us a call: decode is synchronisation-bound, not bandwidth-bound.
- **Write once for the host mailbox:** in host mode every rank's slots are host memory, yet the one-shot pushes the
  same message to each peer's slot ((ws-1) writes). Peers could read one copy. Worth it only if the timeline shows the
  push matters.

## Experiment 2 result: where the one-shot's time goes (2026-11-12, rad0023, build 9e3cf8c7, logs tl-1011)

MXFP4 6/6/12, exact wire, one server per arm, 512-token prompts and 256 new tokens. Per rank and phase, median us of the
slowest block (r0 and r1 the XTX, r2 the R9700 on the x4 link). Decode at 8 streams is the 320 KiB message (32 rows);
at 1 stream, 40 KiB.

| arm | message | rank | push | wait | fold | total |
|---|---|---|---|---|---|---|
| 8 streams, 4x256 | <= 512 KiB | XTX r0 / r1 | 46 / 46 | 26 / 2 | 49 / 72 | 120 / 121 |
| | | R9700 r2 | 1 | 99 | 120 | 218 |
| 8 streams, default | <= 512 KiB | XTX r0 / r1 | 51 / 51 | 45 / 26 | 55 / 82 | 144 / 147 |
| | | R9700 r2 | 1 | 96 | 150 | 247 |
| 1 stream, 4x256 | <= 64 KiB | XTX r0 / r1 | 6 / 6 | 3 / 2 | 7 / 9 | 15 / 16 |
| | | R9700 r2 | 1 | 22 | 15 | 37 |

Across ranks (launches matched by size), the flag round trip, the smallest wait of a launch, is 2 us at 8 streams with
4x256 (25 us with the default grid) and 1.6 us at 1 stream. Synchronisation is not the cost.

What it shows:

- **The R9700's fold is the long phase.** It reads both peers' messages over its x4 link: 640 KiB in 120 us (about
  5.3 GB/s) against 49-72 us on the XTX. Its own push is posted and costs nothing on its clock.
- **The R9700 arrives first and waits** about 95 us for the XTX pushes (2 x 320 KiB each, 46 us), then folds. Its
  critical chain per all-reduce is the XTX push plus its own fold, about 170 us of its 218 us.
- **4x256 helps through the XTX push and fold** (51 to 46 us, 55 to 49 us) and the R9700 fold (150 to 120 us): fewer
  blocks contend less on the host link. That is the 4-5% it gave at 8 streams.
- At 1 stream the all-reduce is 15-37 us a call, so decode there is not all-reduce bound.

Next (experiment 3): cut the bytes the R9700 reads at decode sizes. The three-rank weighted two-shot (rad0021) gives it
the smallest shard, so it reads about one message instead of two, for one more handshake (2 us measured). Upstream's
four-rank one-shot already hands over to its two-shot from the row's `ts_min`; a patch can do the same for the
three-rank host one-shot with the settled link weights, pinned on and off by `--tune-pin all_reduce:ts_min=N` for an
A/B. Also worth a look: a push that writes each message once to host memory for both peers, which halves the XTX push.

## Experiment 3 result: decode takes the weighted two-shot (2026-11-12, rad0024, build a08d17ff, logs e3-1056)

rad0024 hands the three-rank host one-shot to the link-weighted exact two-shot from `ts_min` elements (row default
32768), so at decode sizes the R9700 owns the smallest shard and reads about one message instead of two. 2x ABBA on
MXFP4 TP3 6/6/12, both arms pinned `nblocks=4,nthreads=256`; "off" adds `ts_min=0`. Mean tok/s of four runs a mode:

| | 1 stream | 4 streams | 6 streams | 8 streams | TTFT 512 / 2048 / 8192 |
|---|---|---|---|---|---|
| off (one-shot) | 107.7 | 187.8 | 184.3 | 200.0 | 346 / 1057 / 3966 ms |
| on (rad0024) | 107.6* | 197.2 | 190.8 | 212.2 | 349 / 1072 / 3977 ms |
| change | 0 | +5.0% | +3.5% | +6.1% | +0.8% / +1.4% / +0.3% |

\* one of the four "on" runs measured 87.4 at one stream; the other three 107.5-107.7. One stream is 4 rows (20480
elements), below `ts_min`, so it takes the one-shot either way; the outlier is left out of the mean.

Every run gave the same greedy answer (same hash in all eight), as the same fold order requires. The timeline arms
confirm the path: with the handover off the one-shot logs 10920 launch-ranks of 256-512 KiB at eight streams; with it
on, none above 64 KiB (the decode messages left the one-shot). Prefill is unchanged (its messages were already above
the two-shot's floor). Against TP2 (N8: 182.0 / 176.9 / 196.1 at 4/6/8 streams) TP3 is now +8.4% / +7.9% / +8.2%.
## Experiment 4 result: decode alone scales; prefill is the gap (2026-11-12, build a08d17ff, logs e4-1107)

The standard bench (512-token prompts, 128 new tokens) mixes prefill into its tok/s. With 128-token prompts and 1024
new tokens, so decode dominates, ABBA TP2 against TP3 (6/6/12, 4x256, rad0024), mean of two runs each:

| | 1 stream | 4 streams | 8 streams |
|---|---|---|---|
| TP2 | 96.3 | 284.6 | 360.9 |
| TP3 | 137.6 | 392.9 | 456.8 |
| TP3 over TP2 | +42.8% | +38.1% | +26.6% |

The third card's bandwidth is a third more than two XTX; TP3 gains more than that at 1 and 4 streams (TP2 pays its
own host-staged all-reduce on fewer, larger shards). Decode is therefore not where TP3 falls short. The standard
bench's +8% at 4-8 streams comes from prefill, where TP3 is 8-24% slower to the first token than TP2. The next lever
is prefill: overlapping the prefill all-reduce with compute across sequences (multi-sequence pipelined prefill),
a fused three-rank all-reduce with the following norm, and upstream's prefill interleave when v1.18.0 ships.

## Experiments 5-7: prefill (2026-11-12, build a08d17ff, logs e5-1116, e6-1120, e7-1125)

**Experiment 5, one-prompt prefill.** Cold time to first token, one stream (first round of two; the second hits the
prefix cache):

| | 512 | 2048 | 8192 tokens |
|---|---|---|---|
| TP3 (6/6/12, 4x256, rad0024) | 311-315 ms | 944 ms | 3845 ms |
| TP2 | 317 ms | 858 ms | 3791 ms |
| TP3 `--pipelined-prefill off` | 314 ms | 1208 ms | 4915 ms |

One prompt now prefills about as fast on TP3 as on TP2 except at 2048 tokens (+10%); the earlier gap (N8) averaged the
cold and cached rounds. The one-sequence pipelined prefill (rad0020) is worth 22-28%. `--max-num-batched-tokens 4096`
changes nothing (prefill chunks stay at the 2048-token checkpoint interval) and `--tp-attn-split rows` is refused for
this model.

**Experiment 6, multi-sequence prefill.** No scheduler option makes the 4-8 stream bench pipeline its prefill: its
steps hold several sequences and `pipe_ready` refuses them ("runs serially: more than one sequence"). Options tried:
`--busy-prefill-chunk off`, step caps of 512 and 1024 tokens, a 256-row pipeline floor.

**Experiment 7, the pipeline floor.** 2x ABBA, `--pipelined-prefill-min-rows 256` against the plugin's 1024 (mean of
four runs each): 117.1 / 207.2 / 221.9 against 108.8 / 201.4 / 218.4 tok/s at 1/4/8 streams, +7.6% / +2.9% / +1.6%,
the same answer throughout; 128 measured the same as 256. A 512-token prompt now pipelines too. The TP3 layout sets
`tuning.pipelined_prefill_min_rows = 256`; the int8 layout is not measured yet.

### Where device time goes (profile of an earlier build, before rad0021/22/24; `tools/lab/radiance/tp-profile.sh`)

| workload | rank | trunk | drafter | collectives |
|---|---|---|---|---|
| TP3, 8 streams | XTX | 41% | 3.9% | 55% |
| TP3, 8 streams | R9700 | 35% | 1.6% | 63% |
| TP2, 8 streams | XTX | 63% | 4.9% | 31% |
| TP3, four 4096-token prompts | each | 26-29% | 0.4-0.8% | 70-74% |

Profiled collectives include waiting for peers. Two conclusions: the drafter is 2-5% of device time, so moving
DFlash2 to a separate card (the R9700 alone, or the RX 6900 XT) cannot pay for shipping the trunk's hidden states to
it and its drafts back every step, and the next draft waits on the verify, so there is little to overlap. Unlike the
llama.cpp setup, the drafter here is a small head on the trunk's own features, not a separate model. And a
multi-sequence prefill step spends about 70% of its time in collectives, because it runs serially.

### Next: pipelined prefill over several sequences

`pipe_ready` pipelines only a one-sequence prefill step, split inside the sequence (attention reads the first piece's
keys from the cache, the delta-net hands its state on). A step of several whole sequences can be split between
sequences instead: the two pieces then share no attention or recurrent state, and each block's serial path already
runs over a contiguous run of sequences (`[D, S)` in the delta-net). The change, in the `qwen35_fp8` plugin as a
managed patch:

1. `pipe_ready`: accept a pure prefill step (no decode rows) of two or more sequences, the cut at the sequence boundary
   nearest half the rows; the piece table gains the per-piece sequence range.
2. Attention and delta-net piece paths: for a sequence-boundary cut, run each piece over its own sequences' slice of
   the step's per-sequence arrays (query bounds, seqused, block table rows, slots, positions; conv and state indices).
3. The MLP bodies and boundaries are row ranges already and need no change.
4. Same bits as the serial step (every row's arithmetic is unchanged); measure at 4/6/8 streams and with four 4096-token
   prompts, with a profile arm. Later: steps that mix decode rows with prefill.

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
