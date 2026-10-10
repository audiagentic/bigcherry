---
id: PNRO18
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-25T23:16:44.258298+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# 1252 P2P probe hardening — terminal after provider rejection

## Description

**Closed without implementation.** PNRO18 was a correctness-hardening follow-up to PNRO03/1252, not a separate optimisation. The parent provider was rejected on 2026-10-08 at b11474 following a 2026-10-07 lab fault. Do not reactivate the rejected provider merely to finish probe work.

## Implementation-level finding

In `engines/llamacpp/patches/1252_nro03_allreduce_p2p_provider/patch.py`, `ggml_cuda_ar_p2p_probe()` tests fresh `cudaMalloc` source/destination buffers with source-current `cudaMemcpyPeerAsync`, stream synchronization and byte readback at 4 KiB, 64 KiB+256, 1 MiB and 4 MiB. Actual `ggml_cuda_ar_allreduce_p2p_impl()` uses `p->dev_tmp`, cross-device `p->ev_pool`/`p->p2p_done` event slots, compute-stream waits and asynchronous reduction. The probe does not cover production buffer ownership, slot reuse or dispatch thresholds. **This is a proven coverage mismatch, not a proven root cause of the fault.**

The permanent `tools/tests/hardware/test_p2p_copy_correctness.py` uses `hipMemcpyDefault`, not the production peer-async path. `evidence/validation.json` reports full-vocabulary bit identity **but activation=fail and contract passed=false**. Fallback output parity cannot qualify P2P correctness or speed. `docs/evidence/lab-run-summaries/ab-27b-p2p.json` is historical and not activation-qualified.

## Ownership / consolidation

PNRO03 owns rejected 1252; PNRO18 has no surviving independent implementation. PGC09/PGC12 own production exact-F32 host/RCCL routing. Existing hardware tests own generic peer-copy diagnostics. Do not create another transport, probe, scheduler, selector, allocator, queue or configuration surface.

## Upstream / external decision

llama.cpp #27825 already merged native HIP internal AllReduce; it does not prove direct-P2P safety. llama.cpp #21648 documents corruption without peer access. vLLM ROCm custom AllReduce and external dual-R9700 direct-P2P results are topology-specific and not performance evidence on BigCherry's no-P2P gfx1100/gfx1201/gfx1030 setup. **Wait; no port or adoption.**

## Conditional re-entry gate (not queued)

Only a materially new peer-capable topology/driver and supported workload may reopen PNRO03. Verify directed capability and enablement, actual production `dev_tmp`, source-owned streams, cross-device event-slot reuse, all runtime `copy_bytes`/chunk thresholds ± one element, repeated graph replay, multi-request same-process, positive provider-completion trace, bit-identical full-vocab logits/tokens and crash freedom. Then compare same-binary P2P versus host/RCCL for decode, MTP and prefill. Reject on any missing activation, mismatch, hang or no repeatable E2E benefit. Keep 1252 rejected on current hardware.

## Validation

Audit-only: ten pinned source/record static assertions and sixteen disposable fail-closed admission fixtures passed. No repository pytest, HIP compilation, GPU execution or new benchmark.

## Terminal acceptance

PNRO18/PNRO03 state done, 1252 rejected/default-off, negative evidence preserved, no new experiment queued.

## Change Log

- 2026-09-25: Created as P2P probe-hardening follow-up.
- 2026-10-09: Closed after parent rejection; clarified activation and production-path evidence.

## Ledger-events

- chg_20260925_232233_correctness-checks-for-approxi_1217
- 2026-09-25T23:22:45.229638+00:00 (updated-by): Updated: section:ledger-events
