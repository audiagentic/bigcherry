# RAD50 Gate 4 — cross-process GPU-already-waiting readiness

Date: 2026-11-09. Hardware: RX 7900 XTX (gfx1100), Brutus. **Outcome: PARTIAL PASS for 1–256 rows; FAIL at 768 rows. Gate 4 is not approved.**

## Tested protocol

The persistent independently spawned CPU-only helper receives a GPU-local HSA VMM DMA-BUF FD via SCM_RIGHTS and maps it with Linux mmap. It calls the **production combine_row_f16** (AVX2/F16C, two BF16 helper rows of width 2,560), writing directly to GPU-local VRAM. Four deterministic BF16 input profiles rotate by row and generation: 1+2, helper mask, overflow fallback sentinel and canonical NaN. The GPU checks **every FP16 word** against an independent oracle.

The parent also shares a memfd control region with the helper, registered with hipHostRegisterMapped and mapped to the GPU. The GPU kernel is launched **before** the helper receives its command and publishes armed=ticket. The parent waits for armed, then commands the helper to combine. The helper performs DMA_BUF_IOCTL_SYNC START, writes the payload, performs DMA_BUF_IOCTL_SYNC END, and **only then** stores ready=ticket with CPU release semantics and sends its socket ACK. The already-running GPU performs system-scope acquire of the ready flag and checks every FP16 word, then publishes errors and done=ticket. The parent requires exact done, zero errors and GPU synchronization before buffer reuse. The previous ready ticket is deliberately not cleared: stale generations cannot release the next waiter.

The live kernel and host have bounded waits. A timeout is **a correctness failure**, not a pass. This is hardware evidence, not a guarantee under the HIP memory model or at serving scale.

## First hardware runs — 256 threads polling host readiness

| GPU | Rows | Generations | Result | Observed end-to-end p50 |
| --- | ---: | ---: | --- | ---: |
| XTX0 | 1 | 10 | PASS | approximately 56–57 µs |
| XTX1 | 1 | 10 | PASS | 59.27 µs |
| XTX0 | 16 | 10 | PASS | 274.73 µs |
| XTX1 | 16 | 10 | PASS | 281.88 µs |
| XTX0 | 256 | 10 | PASS | 2,542.28 µs |
| XTX1 | 256 | 10 | PASS | 2,524.46 µs |
| XTX0 | 768 | first generation | **FAIL, GPU readiness timeout** | no valid timing |

The first 768-row failure had errors=4294967295 (GPU readiness timeout sentinel). It did NOT report an FP16 mismatch. The large 16/256-row timing includes a diagnostic GPU reader with 256 host-memory polling lanes and must not be compared with the earlier serial A/B, fused Radiance consumer or inference throughput.

## Single-lane polling correction — still FAILED

We changed the GPU reader so only **one GPU thread** polls the host flag and then releases the 256 verification threads through a block barrier. The new kernel also reports the GPU-observed ready generation and spin loop count. One fresh-lease 768-row diagnostic on XTX0 again failed on its first generation:

- Expected generation: 868001; CPU observed host ready=868001 after successful DMA-BUF write and SYNC_END.
- GPU observed gpu_seen_ready=0, wait_loops=526, errors=4294967295 (timeout).
- CPU helper ACK was received 872.71 µs after arming.
- No generation was verified as correct and no completion marker was emitted.
- Reducing polling contention was **not** a fix. The failure is consistent with unsuitable host-control coherence or visibility, but the precise cause is not established. Do not make the timeout longer or weaken verification to get a PASS.

The initial failing raw log and the one-poller diagnostic remain in the evidence directory alongside every earlier successful run (rad50-live-xtx*.log). Files include original stdout, stderr, exit code and truncation/timeout status; only machine-specific command-header paths are sanitized. Use live_analyze.py --provisional to inspect failures and partial data; its strict mode refuses to turn incomplete runs into final results.

## Next candidate — extended fine-grained registered control

**Prepared and compiled; NOT HARDWARE TESTED.** The existing single allowlisted rad50-tester now supports the fixed dmabuf-live-fg mode, using hipHostRegisterMapped | hipExtHostRegisterUncached for the memfd control, while leaving the DMA-BUF payload path, independent helper, GPU waiter and correctness oracle unchanged.

The installed ROCm HIP header documents the extra flag as mapping onto the extended fine-grained host-memory pool, AMD Linux **except Navi4X**. It may be appropriate for RX 7900 XTX/gfx1100 but must not be presumed to work on R9700/gfx1201. If unsupported, record the returned HIP error and STOP. This is a test candidate, not a demonstrated resolution.

On a free host after acquiring the exclusive GPU lease, suggested bounded first runs (stop on any failure):

    rad50-tester --test dmabuf-live-fg --device 0 --iterations 10 --rows 1 --lease-confirmed
    rad50-tester --test dmabuf-live-fg --device 0 --iterations 10 --rows 768 --lease-confirmed

If both pass, repeat on XTX1, expand generation count, test helper death/restart, stale/duplicate tickets, buffer lifetime and real overlap. If fine-grained registration still fails, a new design is needed.

## Two additional unimplemented control-plane alternatives

**Coherent parent relay:** allocate a separate hipHostMallocCoherent|hipHostMallocMapped flag in the GPU parent, which publishes the ready ticket after receiving the CPU-only helper's post-SYNC_END socket ACK. This adds parent CPU relay and needs full critical-path measurement. The CPU helper cannot directly access an unrelated HIP allocation without a proven sharing mechanism.

**Device-local ready flag:** place the ready word in the exported GPU-local DMA-BUF, with carefully ordered CPU stores, DMA_BUF_IOCTL_SYNC and GPU-local polling. The GPU cache hierarchy and CPU posted-write ordering require independent hardware proof; serial GPU verification is insufficient.

Official HIP documentation distinguishes fine- versus coarse-grained memory; Linux DMA-BUF documentation states DMA_BUF_IOCTL_SYNC provides cache coherency but does not itself coordinate GPU and application access:
- https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/coherence_control.html
- https://kernel.org/doc/html/next/driver-api/dma-buf.html

## Safety and status

Gate 4 **not passed**; the 768-row GPU-live publication failed twice on XTX0. No XTX1 768-row run or R9700 Gate 4 run was launched. No inference or serving speedup is established. Both RAD50 GPU leases were released after tests, and the host was idle/no KFD PIDs. RAD49 subsequently reacquired the exclusive GPU lease, so all further GPU work was stopped.

The unified tester remains one executable, the existing worktree only. Lab-only code, no production/vendor changes, no push or PR. Card-free: 72 Python unit tests passed; 60 CLI parser checks passed.

## Subsequent corrective hardware result — GFX11 wall-clock deadline (2026-11-09)

**This section supersedes the earlier "Gate 4 not passed at 768 rows" diagnosis above; the old failures are retained as essential historical evidence.** Official HIP documentation states that `clock64()` is not reliable on GFX11/RDNA3. Device-local timeout records had exit_reason=2 after only ~528 host-control polls, with implausible clock deltas. Replacing the watchdog with `wall_clock64()`, scaling its 3-second deadline by `hipDeviceAttributeWallClockRate`, and retaining independent host-side deadlines resolved the observed 768-row premature timeouts. XTX0 768-row tests subsequently pass 10 generations with the original mapped-host control and extended fine-grained registration; they are still very slow (~7.38 ms p50) with the old one-block verifier. This establishes a broken watchdog in the test, **not** a proven problem with FP16 payloads or ready visibility.

A new 256-block parallel GPU verifier, preceded by a one-thread waiter and followed by a separate done-publication kernel, was tested with both direct CPU-child publication and a coherent parent relay. Both methods pass at 768 rows on both XTXs, including **100 verified generations**, old ticket retention, GPU first-armed ordering, SYNC_END before ready and exact FP16 full-word checking. Direct-publish 768-row p50/p95: XTX0 463.87/596.41 us, XTX1 463.71/590.44 us; coherent relay: XTX0 469.06/607.11 us, XTX1 471.06/532.53 us. Difference in p50 is only ~5–7 us from sequential experiments, not a proven causal speedup. Every generation contributed to the summary, while compact stdout contains first three, each tenth and final plus all failure records. Earlier failed logs and all newer successes are in `evidence/2026-11-09/rad50-slices-*.log`. Use `slice_analyze.py` for strict complete-run validation, or `--provisional` for partial logs. See `SLICE_OPTIMIZATION.md` and `docs/planning/active/patching-radiance-rdna3/RAD50_SLICE_OPTIMISATION.md` for the per-slice optimization agenda. **Gate 4 lab correctness is materially advanced, but production readiness remains blocked** on helper crash/restart, FD lifetime, real fused consumer semantics, TP2 rank exact-once, genuine PCIe contention and measured end-to-end inference. The historical negatives above are not current status after the clock correction.

## 2026-11-09 later correction — earlier 768-row failure was a false device deadline

The original `clock64()` GPU watchdog was unsuitable on the tested gfx1100 XTXs. Device-local diagnostics showed an expired clock deadline with only hundreds of wait loops; replacing it with `wall_clock64()` scaled by HIP's reported wall-clock rate restored 768-row readiness. This **supersedes the earlier Gate 4 failure conclusion above**; original failure logs remain useful and preserved. Both XTXs now pass 100 repeated generations of 768 rows in the direct GPU-ready variant with a parallel 256-block FP16 verifier; coherent parent relay also passes. Full p50 and stage diagnostics are in `SLICE_OPTIMIZATION.md` and `rad50-slices-validated-summary.txt`. Further fault/restart, cross-device memory-order review and full Radiance inference A/B remain required. These are lab-only results, not production enablement.
