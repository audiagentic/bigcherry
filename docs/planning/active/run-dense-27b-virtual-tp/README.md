# Dense 27B — current Radiance-native physical TP work

**Current Planning state (2026-11-22):** The earlier VTP01–VTP14 logical-rank, model-geometry, IQ6R, W8A8 and initial TP3 investigation series is **completed historical evidence**, not fourteen active implementation tasks. VTP31/33/35/39/41/43/44 also completed. The earlier `exp/xtx27-prefill-opt` worktree and archive tag are historical; **do not use them as the current execution branch**. Check the current canonical plan owner and the pinned engine before changing code.

## Active owners and acceptance

| Active plan | Single responsible scope |
| --- | --- |
| [VTP32](VTP32.md) | Per-op-kind physical-card split costs and measured `tp_ownership` decisions |
| [VTP36](VTP36.md) | Physical TP3 prefill/decode performance and mixed-device kernel, collective and split evidence |
| [VTP37](VTP37.md) | INT8 W8A8 TP3 decode/prefill route and exact fused collective/normalisation admission |
| [VTP38](VTP38.md) | Production physical TP3 host transport and optional ordered reduction |
| [VTP40](VTP40.md) | Measured speculative-window decisions under load |
| [VTP42](VTP42.md) | Prefill governor, bounded reservation and backfill; source and A/B proof |
| [VTP45](VTP45.md) | Outstanding VTP44 follow-ups: routing determinism, selftest geometry, R9700-only and KLD |

## Completed source/evidence, not pending dependencies

- [VTP34](../../completed/run-dense-27b-virtual-tp/VTP34.md) superseded: C01–C24 audit retained; native configuration and model geometry work transferred to RSC11/RCS12 and existing owners.
- [VTP56](../../completed/run-dense-27b-virtual-tp/VTP56.md) superseded: shipped IQR tiles and negative results retained; remaining work in VTP57/VTP58.

- [VTP14](../../completed/run-dense-27b-virtual-tp/VTP14.md) holds the native Radiance integration and physical TP3 foundation. Avoid a second virtual-rank scheduler.
- [VTP01](../../completed/run-dense-27b-virtual-tp/VTP01.md)–[VTP13](../../completed/run-dense-27b-virtual-tp/VTP13.md) preserve the original Q6/Q8 geometry, capacity, per-GPU kernels, host collective and model-quality decisions.
- [VTP31](../../completed/run-dense-27b-virtual-tp/VTP31.md), [VTP33](../../completed/run-dense-27b-virtual-tp/VTP33.md), [VTP35](../../completed/run-dense-27b-virtual-tp/VTP35.md), [VTP39](../../completed/run-dense-27b-virtual-tp/VTP39.md) and [VTP41](../../completed/run-dense-27b-virtual-tp/VTP41.md) record collective, capacity, prefill-order and concurrency acceptance already delivered.
- [VTP43](../../completed/run-dense-27b-virtual-tp/VTP43.md) and [VTP44](../../completed/run-dense-27b-virtual-tp/VTP44.md) record the packed-KV and generic layout acceptance; follow-ups belong only to VTP45, not a reopened VTP44.

The target remains native dense 27B **Q6/Q8**, 4+ concurrent streams and resident KV on two gfx1100 RX 7900 XTX cards plus a gfx1201 R9700 with no assumed P2P. Benchmark results must be tied to actual model quantisation, physical device ownership, full numerical acceptance, pinned source and the current native release. A microkernel result or archived experimental data is **not** a new production TP3 performance or capacity claim.

Engine-native serving/collective lifetime and ABI fixes belong to their existing Radiance RUM/serving owners; VTP plans implement and measure **dense-model-specific** requirements only. No alternate launcher, shared dispatcher, source pin or physical-rank resource allocator should be added here.
