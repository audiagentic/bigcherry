# Dual-XTX P2P diagnostics (GP11, 2026-09-04)

Standalone HIP programs used to establish why every GPU-to-GPU P2P design
fails on this box, after `hipDeviceCanAccessPeer` started reporting 1 for the
GPU0<->GPU1 (RX 7900 XTX) pair. Build each with:

```
hipcc -O2 -std=c++17 -o <name> <name>.cpp
```

Run order and what each one settles:

| Program | Question it answers | Result |
|---|---|---|
| `p2p_diag.cpp` | Which peer operations work? Includes a **local-read control arm** proving the test harness itself is sound. | Local control OK. Kernel peer *reads* return 0 both directions. Kernel peer *write* works. `hipMemcpyPeer` returns 0. |
| `p2p_write.cpp` | Do peer writes hold up at scale, scalar and vectorized? | Correct at every size to 67 MB, scalar and `float4`. Apparent >PCIe-rate bandwidth at small sizes is posted writes buffering, not real throughput; 67 MB shows the true ~6.8 GB/s. |
| `p2p_coh.cpp` | Is a peer write visible to a kernel on the destination device? | Yes, for a kernel launched *after* the writer completed — with any load flavour. |
| `p2p_spin.cpp` | Is a peer write visible to a **concurrently running** kernel? **The decisive test.** | No. 2,000,000 iterations each of sys-acquire load, system-scope atomic RMW, non-temporal load, volatile load, and fence+load all fail to observe it. |
| `p2p_d2d.cpp` | Does `hipMemcpyPeer` / D2D corrupt at realistic sizes? | `hipMemcpyPeer` corrupt both directions. `hipMemcpy` D2D/Default correct when the current device is the **source**. |
| `p2p_bwval.cpp` | Was the earlier "10.4 GB/s P2P" measurement real? | No. Replicating that benchmark's exact configuration with validation added gives 262144/262144 elements wrong (all zeros). |

## Round 2 (2026-09-04): exhausting the design space

| Program | Question | Result |
|---|---|---|
| `asyncprobe.cpp` | Which ASYNC push variants are legal *and* correct? | Exactly one works: `hipMemcpyAsync` + `hipMemcpyDeviceToDevice` + **explicit non-null stream** + current device = **source**. `hipMemcpyDefault`, the null stream, and `hipMemcpyPeerAsync` all silently corrupt. |
| `dmabench.cpp` | Is a DMA push + kernel-boundary reduce better than host staging? And can the transfer be HIDDEN behind compute? | No, and yes-but-irrelevant. See table below. |
| `dispatch_overhead.cpp` | What does a kernel dispatch actually cost, and do HIP graphs help? | 3.463 us/kernel stream launch vs 2.848 us via graph replay (1.22x, 0.615 us saved). |
| `analyze_decode_trace.py` | Where does decode time actually go? (union-of-spans, never summed durations) | MMVQ 50.3%, AllReduce 6.6%, **GPU idle 30.2%**. |

DMA-push (arm C) and DMA-push-overlapped-with-compute (arm D) vs the
production host-staged design (arm A), all correctness-validated:

| size | A host | C dma | C vs A | D overlap | hidden |
|---|---|---|---|---|---|
| 30 KB | 25.48 us | 112.34 | +341% | 154.58 | 44.8 us |
| 60 KB | 29.52 | 106.10 | +259% | 135.35 | 47.7 |
| 120 KB | 42.84 | 115.86 | +170% | 142.29 | 46.2 |
| 240 KB | 63.36 | 130.46 | +106% | 159.72 | 49.9 |
| 480 KB | 105.44 | 174.02 | +65% | 199.05 | 40.4 |
| 1 MB | 302.40 | 298.30 | -1.4% | 318.57 | 44.2 |

The DMA-push design carries a ~100us FIXED floor (two stream syncs plus a
kernel boundary) against host staging's ~25us total at decode sizes. Overlap
really does work -- a consistent ~44us of transfer hides behind compute -- but
cannot cover that deficit. (The 1MB row's -1.4% here looked like a crossover
but was noise -- see Round 3, which re-measures it at +6.2% against a properly
measured baseline. There is no crossover at any size.)

**The governing constraint is that at decode sizes the ENTIRE collective costs
~25us, while a single kernel boundary plus stream sync costs tens of us.** Any
design needing an extra dispatch boundary loses before it transfers a byte.
That is why the production single-kernel, in-kernel-handshake, pinned-host
design wins, and it is now measured rather than assumed.


### Round 3: large/prefill sizes -- still no crossover

The decode-size table above showed DMA-push converging toward parity with
size, and a -1.4% reading at 1MB suggested a possible crossover. Re-measured
against PROPERLY MEASURED host-staged baselines (the earlier large-size
baselines were estimates, and the -1.4% was noise):

| size | A host (measured) | C DMA push | verdict |
|---|---|---|---|
| 1 MB | 304.48 us | 323.27 | +6.2% slower |
| 2 MB | 403.67 | 521.98 | +29.3% slower |
| 4 MB | 780.91 | 916.25 | +17.3% slower |
| 10 MB (prefill AllReduce size) | 1919.01 | 2362.52 | +23.1% slower |
| 20 MB | 3816.31 | 4538.95 | +18.9% slower |

**There is no size at which P2P wins on this machine.**

Why, and this is the structural point: host staging is NOT serialized. Each
GPU writes to host memory over ITS OWN dedicated Gen4 x8 root-port link, in
parallel with the other. P2P instead funnels everything through a single
hairpin between two separately-bifurcated CPU root ports. So P2P trades two
parallel links for one shared path that is no faster per byte -- losing on
both bandwidth and latency. The "avoid the double hop" intuition is wrong
here because the double hop was never the bottleneck.

Generalization: P2P would only pay off on a topology where both GPUs sit
under a COMMON PCIe switch (traffic stays in-hierarchy and never reaches the
root complex), or over a real interconnect (XGMI/NVLink). If these cards are
ever moved behind a PLX-style switch, the broken pull path may also start
working and this analysis should be redone. On the current board it is closed.

## Conclusion

Two independent blockers, either one fatal:

1. **Pull operations silently return zeros.** Kernel peer reads, `hipMemcpyPeer`
   (both directions), and `hipMemcpy` with the current device set to the
   destination all produce zeros with no error and no fault. Only *pushes*
   (current device = source, or kernel-side peer stores) move real data. This
   is the standard PCIe posted-write/non-posted-read split, and it is why
   NCCL/RCCL push into remote buffers rather than pulling from them.

2. **Peer writes do not drain until the writing kernel retires.**
   `__threadfence_system()` does not flush them. The atomic-RMW arm of
   `p2p_spin.cpp` is decisive: an RMW cannot be satisfied by a stale cache
   line, so the data is genuinely not yet in the destination's memory. This is
   a writer-side flush problem, not a reader-side coherency problem.

Together these rule out any single-kernel, fine-grained handshake collective
between these two GPUs. P2P could only be used as
`push kernel -> kernel boundary -> reduce kernel`, and that launch plus
synchronisation cost exceeds the entire theoretical saving at decode message
sizes. The production pinned-host-memory design works precisely because host
memory writes are immediately visible to the peer.

## Safety note

The silent-zero behaviour on pulls is a real data-corruption hazard on this
machine: any device-to-device copy between GPUs 0 and 1 issued with the
destination device current returns zeros and reports success. RCCL currently
selects SHM transport for this pair, so production paths are unaffected today,
but any future change that enables a D2D or `hipMemcpyPeer` path here would
corrupt silently.
