---
id: VTP04
order: 3
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Fast virtual TP4 experiments: local R9700 and physical three-way exchange

## Objective / why small harnesses first

Answer the two least expensive physical questions before building a Q6/Q8
model or modifying Radiance graphs:

1. Can the R9700 execute its TWO logical slices with less exposed latency
   using one fused physical kernel, one stream sequentially, or two HIP
   streams with event dependencies?
2. Does a three-physical-GPU **exact F32**, no-P2P host reduction benefit
   from GPU-local R9700 combine and ONE host packet rather than two?

These are **microarchitectural feasibility** experiments. They cannot
predict real quantized 27B FFN GEMM performance, quality, graph scheduling,
KV residency or end-to-end token/s. Do not conflate Q6_K GGUF, Radiance
IQ6R and Radiance W8A8 encodings. VTP02 owns memory/quality envelopes;
XDP03 owns Q6/Q8 kernel compatibility and conversion; KAB01/rad-kbench
owns native 27B-shape kernel throughput at M1/4/8/64/128/512/2048.

## Implemented experimental code (no vendor edits)

- tools/lab/radiance/xtx27_vtp_local.hip: one gfx1201 physical GPU,
  two synthetic UINT32 device weight buffers. Mode A fused one kernel;
  mode B two sequential scans plus a local add; mode C two independent
  nonblocking HIP streams with event joined GPU local add. Every mode
  scans the **same bytes** and returns identical per-lane integer sums.
  Independent CPU samples check values (including UINT32 wrapping);
  all lanes compare across modes. Time both GPU events and host wall.
  Sweep 4/16/64 MiB **per virtual shard**, default 32768 lanes. Not a
  native Q6/Q8 GEMM. Do not infer HBM-saturation or BF16 correctness.
- tools/lab/radiance/xtx27_vtp3_host_collective.hip: discover exactly
  two gfx1100 + one gfx1201 from HIP arch, ignoring gfx1030. Exact
  F32 input codes are powers-of-two fractions, fully checked against
  an independent oracle on ALL three physical GPUs. Local R9700
  add produces a single D2H packet versus two separate R9700 packets;
  CPU-root sum and H2D broadcast are timed together and by phase.
  Three device streams and HIP events, no P2P, no GPU-side spin waits.
  Sweep M=1/4/64/512 (optionally 2048) at width=5120.
- tools/lab/radiance/xtx27_vtp_micro.py: one CLI for dry-run, GPU-free
  HIP compilation and exclusive-lease GPU campaign. Compiler+source
  digest manifests and per-case logs under BC_WORK_ROOT scratch.
  No GPU interaction unless --phase gpu --run explicitly requests
  agent-lease acquire; exits without running if the GPU is leased.
  No generated artifact or model download inside the source worktree.
- tools/tests/lab/test_xtx27_vtp_micro.py: bounds, no implicit GPU
  execution, per-kernel source provenance, host/CPU oracle presence.

## Run contract

From the existing experimental worktree:

    python3 tools/lab/radiance/xtx27_vtp_micro.py

    python3 tools/lab/radiance/xtx27_vtp_micro.py --phase compile --run

    python3 tools/lab/radiance/xtx27_vtp_micro.py \
      --phase gpu --run --local-mib 4,16,64 --rows 1,4,64,512 \
      --lanes 32768 --iterations 9

The first command only prints a bounded plan. Second only compiles two
small HIP programs with gfx1100/gfx1201 offload objects. Third obtains
the global exclusive GPU lease, with no P2P/peer atomics or live model
work. Each case writes stdout, stderr and a JSON receipt in isolated
scratch. Do not run under other agents' leases.

## Acceptance / next decision

- Confirm all local modes and both host reduction modes pass output
  oracle, repeat across three fresh starts and rotate order to reject
  warmup/clock and HIP queue/order artifacts. Track p50/p95 host,
  GPU and per-phase times, actual PCIe link saturation and clocks.
- For each message band determine measured local-combine savings versus
  extra kernel+sync cost. A tiny-message combine may cost MORE than
  sending two packets: preserve both options and require a crossover.
- Compare R9700 fused and two-stream performance under same bytes,
  and test effect of increased lanes or resident competing load
  before believing one stream is better. The winner is a kernel and
  occupancy result, not a quantized decode speed claim.
- Next small step: use **real current-pin rad-kbench fixtures** for
  Q8 and a supported Q6 representation at genuine N/K/M. Retain
  the model's FFN tile rounding, attention/KV head geometry.
- A smaller similar model may cheaply expose scheduler/TP correctness
  but DOES NOT transfer realistic 27B quant-specific weight bandwidth,
  K dimensions, full-attention KV or 4+ stream memory headroom.
  Do not select a smaller model unless its exact architecture, head
  count, quant format and device dispatch are proved; final gate is
  the actual Qwen3.8-27B model and a future 1M-capable model separately.

## Status log

- 2026-11-10: Existing worktree rebased onto origin/main at
  2602d6970; already-merged XDP01-03 commits recognised, catalog
  conflict resolved without dropping upstream entries. XDP09 is
  tracked in a separate commit.
- 2026-11-10: New HIP source prototypes and dry-run CLI written.
  Both HIP source programs **compiled successfully** under the
  available ROCm 7.2 toolchain with gfx1100 + gfx1201 targets.
  Card-free test and lint evidence must be captured before commit.
- Hardware status updated below: early leased canary and partial sweep passed numerical oracles; no whole-model qualification.

## Initial exclusive-lease GPU evidence (2026-11-10)

- Both HIP programs compiled against ROCm 7.2.4, gfx1100 and gfx1201.
- The 4 MiB R9700 canary and 1-row three-physical-device host collective passed output oracles. The larger 9-case sweep finished **7 of 9** cases: local 4/16/64/128 MiB and host 1/4/64 rows; all completed cases PASS. The shell deadline interrupted the sweep before 512/2048 rows. Its scratch summary is correctly **incomplete**.
- 64 MiB R9700 per shard GPU medians (microseconds): fused 294.60, sequential 358.12, two_stream 282.48. At 64 collective rows, local-combine wall median 643.77 versus 870.87 us for two R9700 packets. These are preliminary synthetic numbers, NOT real Q6/Q8 GEMM or 27B speedups.
- The 4 MiB fused median shifted from 113 to 71 us across starts; order/clock sensitivity demands paired reverse-order runs. Invoke the same GPU command with --reverse-order for that control.
- Scratch receipts: ${BC_WORK_ROOT}/xtx27-prefill/virtual-tp4/runs/20261110T020409Z-3bddb279d0 and 20261110T020459Z-3bddb279d0.
- The interrupted shell left an orphan lease with no KFD processes. It was explicitly released; status returned no leases. All further GPU runs need a fresh exclusive lease.

### Reversed-order discriminating run

Fresh ROCm 7.2.4 HIP rebuild (source digest 3541c5ee06) added
--reverse-order with full GPU/CPU oracles preserved. Under a fresh
exclusive lease at 64 MiB per R9700 virtual shard, GPU medians were:
**two-stream 306.33 us, sequential 357.93 us, fused 295.89 us**.
This reverses the small two-stream advantage of the first run; the
data support a clear sequential penalty, but **NO two-stream win**.
At 64 collective rows, sending two R9700 packets was 888.77 us
versus 649.42 us for R9700-local combine and one packet, both PASS.
Receipt: ${BC_WORK_ROOT}/xtx27-prefill/virtual-tp4/runs/20261110T021037926582Z-3541c5ee06.
A full forward/reverse balanced series, actual Q8/Q6 kernels and
whole-model performance remain future gated measurements. Lease released.
