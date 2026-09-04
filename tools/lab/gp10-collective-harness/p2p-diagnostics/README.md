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
