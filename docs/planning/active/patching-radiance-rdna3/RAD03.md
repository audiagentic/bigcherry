---
id: RAD03
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-13T10:31:20.999426+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Redesign the R9700 expert helper as GPU lanes: direct-to-VRAM load, GPU-signalled mailbox exchange, one R9700 stream per XTX

## Description

Owner direction 2026-10-13: the slice-2b helper is CPU-mediated (mmaps the 122 GB container and relayouts on host; per-exchange hipMemcpy D2H into shm, futex wake, H2D, compute, D2H, engine-thread wake; per-layer routing D2H + stream sync; one serial queue for both ranks). Measured: decode 26.5 -> 34.3 tok/s but prefill -19% (-11% from helper presence alone). Redesign so each operation runs in its own GPU lane with no CPU in the data path.

1. Load: read only owned experts in chunks through a small pinned bounce buffer; relayout on the R9700; host footprint a few MB.
2. Exchange: pinned host mailbox per producer rank (no P2P on dev). XTX MoE kernel writes owned-route rows + routes straight into it and raises a flag from its stream (hipStreamWriteValue); the R9700 runs one persistent stream per XTX that waits on the flag on-GPU (hipStreamWaitValue), reads the mailbox in place, computes, writes results + done flag; the XTX stream waits on-GPU and adds. CPU only sets up.
3. Overlap: XTX continues resident experts + shared expert while its R9700 lane runs; join only at the merge.
4. Routing on GPU: a select kernel splits owned host-pool routes on the XTX; no per-layer D2H/sync, gather fusion kept for prefill.
Patch 0002 changes from CPU callbacks (submit/wait) to stream-level operations. The same mailbox exchange is the transport RAD02 (inverted layout) needs. The CPU-mediated 2b version stays as the evaluated baseline.

Validation: fn-helper-gate.sh (helper used on both ranks, decode above helper-off, prefill within 5% of helper-off) before any full on/off comparison; KL quality check with the helper on; host RSS of the helper < 1 GiB.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-10-14 quality finding: the helper's KLD 0.0616 / top-1 88.8% (4-doc corpus, 64-token steps) is the model's own numerical-sensitivity floor, not a helper defect. Evidence: (1) per-layer row dumps (all 44 layers, early and late requests) show helper rows == engine rows except 8.4% non-bit-identical rows, median 1 element at 1 ulp; add kernel reads exactly the buffer. (2) Mask-only split: helper rows only where bit-identical -> KLD 0.000000; only differing -> 0.0616; only 1-8-element diffs -> 0.0556; only >8 -> 0.0608. (3) Noise floor without the R9700: stock XTX path re-chunked at 128 / 256 tokens -> KLD 0.0624 / 0.0582, top-1 88.4%. Quality bar for the helper (and RAD02) is therefore 'within the XTX re-chunking noise floor', not KLD<0.002 vs a bit-exact reference.

2026-10-14 speed A/B (Flash-Next fp8-iq4r, flashnext-2xtx profile, 7306-token prompt + 256 greedy tokens, 8 runs off/on/on/off/off/on/on/off, build from feat/rad03-gpu-lanes 8a7ed66e): decode off 27.0/27.2/27.1/27.2 vs on 48.6/48.6/47.3/48.6 tok/s (mean 27.1 -> 48.3, +78%); prefill off 1185.5/1170.8/1191.1/1179.9 vs on 1257.6/1226.4/1257.4/1253.4 (mean 1181.8 -> 1248.7, +5.7%). Complete separation on both metrics. Greedy output is identical within each arm across all runs (off 349546fd x4, on 7fdbdc2c x4) and differs between arms, consistent with the KL noise-floor finding. Run: ${BIGCHERRY_ROOT}/runs/perf19.

## Change Log

- 2026-10-13T10:31:20.999426+00:00 (created-by): Created by agent
- 2026-10-14T08:39:36.398548+00:00 (updated-by): Updated: section:notes
- 2026-10-14T09:10:27.637907+00:00 (updated-by): Updated: section:notes

## Ledger-events


- chg_20261014_091039_the-r9700-expert-helper-for-fl_9807
- 2026-10-14T09:10:43.118131+00:00 (updated-by): Updated: section:ledger-events
- chg_20261014_160331_flash-next-can-now-run-with-th_1934
- 2026-10-14T16:03:38.933059+00:00 (updated-by): Updated: section:ledger-events
