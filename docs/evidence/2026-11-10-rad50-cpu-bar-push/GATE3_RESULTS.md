# RAD50 Gate 3 — first measured GPU-pull versus CPU BAR-push A/B

Date: 2026-11-09 (Australia/Melbourne). Worktree: exp/rad42-cpu-bar-push. Source HEAD at capture: 6398e999407b86e850bd0c22a0eb1f747b2807d5 plus subsequently documented lab-only analyzer completion marker modifications. ROCm HIP 7.2.53211 / ROCm LLVM 7.2.4. Tested gfx1100 RX 7900 XTX GPU0 PCI 0000:03:00.0 and GPU1 0000:06:00.0.

**Scope:** uncontended, synthetic, SERIAL, persistent CPU-only helper process, HSA VMM GPU-local DMA-BUF mapped by CPU, DMA_BUF_IOCTL_SYNC START/END per payload; baseline directly GPU-reads HIP-registered memfd shared CPU memory. Same GPU full-word pattern-verification kernel on both paths, alternating order per sample. Both paths ran 30 measured iterations plus 3 warmups per payload per method. Every word GPU-checked on every iteration. All test runs returned 0 with RAD50_AB_COMPLETE status=0. Results are NOT production inference speedups, concurrent publication proofs, or models of actual FP16 MoE combine.

## Direct synthetic CPU-compute-to-destination results

| Bytes | XTX0 pull p50 (us) | XTX0 CPU-push p50 (us) | XTX0 speedup | XTX1 pull p50 (us) | XTX1 CPU-push p50 (us) | XTX1 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 73.40 | 74.97 | 0.979x | 73.47 | 74.88 | 0.981x |
| 64 KiB | 82.26 | 77.22 | 1.065x | 83.29 | 78.91 | 1.056x |
| 256 KiB | 110.93 | 95.09 | 1.167x | 109.11 | 92.89 | 1.175x |
| 1 MiB | 276.00 | 216.49 | 1.275x | 272.45 | 212.07 | 1.285x |
| 4 MiB | 1075.16 | 760.30 | 1.414x | 1139.85 | 836.90 | 1.362x |

At 4 MiB the direct 32-bit CPU write costs approximately the same whether writing host RAM or GPU-local DMA-BUF: XTX0 host 539.99 us vs GPU VRAM 539.64 us; XTX1 host 554.93 us vs GPU VRAM 558.71 us. The GPU consumer's kernel time drops from ~331–333 us when reading host RAM to ~25–27 us from local VRAM. Median DMA-BUF SYNC START+END is ~0.50 us on XTX0, ~0.67 us on XTX1. These are median component timings, not additive percentiles. Stage overhead and host GPU completion are included in end-to-end serial p50.

At 4 KiB the extra handoff/SYNC makes CPU-push approximately 1–2% slower. The apparent break-even lies between 4 KiB and 64 KiB for this synthetic bench, not necessarily real MoE chunks.

4 MiB direct p95: XTX0 pull 1113.72 us vs push 827.82 us; XTX1 pull 1268.57 us vs push 926.41 us. Tail latency is variable, and these 30-sample p95/p99 summaries are not sufficient for production SLO decisions.

## Staged CPU prepare + memcpy results

| Bytes | XTX0 speedup CPU-push over GPU-pull | XTX1 speedup CPU-push over GPU-pull |
| ---: | ---: | ---: |
| 4 KiB | 0.980x | 0.991x |
| 64 KiB | 1.041x | 1.013x |
| 256 KiB | 1.040x | 1.041x |
| 1 MiB | 1.075x | 1.069x |
| 4 MiB | 1.111x | 1.108x |

At 4 MiB staged write: CPU staging preparation ~411–413 us; host-memory memcpy ~239–245 us, GPU-local DMA-BUF memcpy ~385–386 us. The additional post-combine copy sharply reduces the benefit versus direct CPU stores. At 4 MiB the serial DMA-BUF CPU-push path is ~760 us (direct) on XTX0 vs ~1096 us (staged), and ~837 us (direct) on XTX1 vs ~1093 us (staged). The direct CPU-write variant is the better candidate to implement in actual MoE combine, subject to equivalence and throughput validation.

## Raw evidence and reproducibility

- Complete direct sweep XTX0: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-xtx0-sweep-direct-30.log
- Complete direct sweep XTX1: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-xtx1-sweep-direct-30.log
- Complete staged sweep XTX0: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-xtx0-sweep-staged-30.log
- Complete staged sweep XTX1: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-xtx1-sweep-staged-30.log
- Earlier per-payload 10- and 30-iteration runs also preserved under ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-ab-xtx*.log; these are useful intermediate measurements and not discarded.
- Analyzer output: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-gate3-o2-analyzed.txt
- Benchmark command: ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-tester --test dmabuf-ab --device {0,1} --iterations 30 --mode {direct,staged} --lease-confirmed
- The performance-relevant tester.o and dmabuf-ab.o were compiled -O2, --offload-arch=gfx1100 --offload-arch=gfx1201, then linked -O2 with previously built inactive probe objects. Executable SHA256 at capture: 6b72dc31fb76e7992bd0e305d333733b4cc7c660304b9c1d7faa398201915014.
- All runs were under an acquired exclusive GPU lease, initially no KFD PIDs or GPU allocations. Released lease and confirmed no leases after testing; four GPUs idle/empty.

## Conclusions, boundaries, and next steps

1. **Positive synthetic serial transport signal** for FP16-like large payloads. The 4 MiB direct path saves ~303–315 us in the complete full-size sweeps and about ~0.3–0.4 ms in separate repeated single-size trials. Do not extrapolate this to per-layer latency or inference throughput.
2. Actual RAD42 combines FP16 routed rows with CPU work overlapping engine GPU pulls; raw isolated serial speedups do not establish any overlap-safe engine improvement. Integrating a *second* full host-to-GPU copy after combine risks losing much of the saving. Test **direct CPU combine into GPU-local DMA-BUF** first, with FP16 accumulation/NaN/overflow exactness.
3. Complete Gate 2b durability (multi-generation slot reuse, helper crash/restart and FD lifetime), Gate 4 concurrent GPU waiter/readiness, and consumer rank once-only semantics. Investigate required CPU/GPU memory ordering beyond DMA-BUF SYNC ioctl and avoid unsafe spin readiness claims.
4. Run controlled PCIe contention A/B on XTX0/XTX1 while respecting exclusive lease, then synthetic captured real FP16 combine and full TP2 prefill/decode/KL checks. End-to-end confidence and p99 can only come from paired real serving trials.
5. The standalone verifier kernel does extra full-word checking and is not Radiance's production fused gather or mailbox consumer. The measured GPU read latency difference depends on test kernel/cache effects; reproduce with a realistic fused consumer before committing to RAD51.
6. Preserve **partial and failure logs** as provisional diagnostic evidence. The strict analyzer requires an exit-0 completion marker for final completed-run decisions, but `ab_analyze.py --provisional <log>` now exposes any observed per-path rows, unpaired samples, and failure/timeout diagnostics without silently discarding them.

**No production/vendor edits, branch pushes, new worktrees or engine integration.**

**Durable evidence snapshots:** the same 14 raw first-run, single-size and all-size logs plus analyzed summary are copied into tools/lab/rad42-bar-push/evidence/2026-11-09/ under this branch, not only the transient ${BC_WORK_ROOT}/builds directory. This preserves the user-requested partial/intermediate evidence across build cleanup.

In the checked-in .log copies, only the machine-specific absolute executable path in the command header is expressed as ${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-tester to satisfy the repository's path policy. Captured benchmark stdout, stderr, exit status, timing rows and completion markers remain unchanged. The exact original raw logs remain in the project build-root directory.
