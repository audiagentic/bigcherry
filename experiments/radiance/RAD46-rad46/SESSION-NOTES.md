### Preliminary rows-kernel finding (2026-11-03)
A captured 6900 XT layer-6 request (122 routes, 119 tokens) measured relative L2 0.00679 / 0.00899 for the two split ranks against the CPU reference. The replay reference was corrected to consume captured, already quantised activations and to require two ranks. With unscaled half-precision operands and float scales, errors were 0.00160 / 0.000648; the XTX grouped path measured 0.000810 / 0.000463. The capture is from a different prompt, so it is not proof of the KLD outlier routes.

A first attested MEN26 configuration capture at candidate commit 383c17c1b gave p99.9: helper-off 2.79229, two XTX 3.19805, three-helper tail 3.71049. The first two match the preceding capture; the old tail was 4.36970. The candidate reduces the tail metric but fails the required +0.5 increase over control (+0.91820). See machine-local artifacts/RAD46-diagnostic/capture-attested-1/ and replay logs in artifacts/logs/manual/. An alternate grouped-GEMM quality capture is in progress. The item remains pending.

### Alternate-path diagnostic (2026-11-03)
The same three-arm KLD configuration with R3_MOE_WORKER_ROWS=0 passed all six limits. Tail p99.9 was 1.9001 against helper-off 2.79229; the two-XTX value was 3.19805 and the controls matched the prior capture. The candidate rows path remained at 3.71049, above the gate. This localizes the remaining difference to the rows path or its interactions, but does not yet identify a particular outlier expert. The alternate path is a diagnostic, not an accepted fix: end-to-end decode and prefill throughput still need comparison. Machine-local results: artifacts/RAD46-diagnostic/capture-gemm/.

### Stage comparison and grouped-fold prototype (2026-11-03)
On the 122-route layer-6 capture, canonical token/slot stage snapshots found 3 gate and 5 up BF16 mismatches on rank 0 and 5 gate and 6 up mismatches on rank 1 when comparing the first rows candidate to grouped GEMM. Six quantisation groups changed on each rank; no changed quantisation group had identical gate/up inputs. This is a different prompt from the KLD outliers.

A prototype that folds the raw 64-K groups and their scales in GEMM order matched the grouped GEMM final output hashes for both ranks of that replay. A short profiler trace over three captured requests, however, measured gate/up 4445 us versus 740 us and down 2721 us versus 519 us for the original rows kernel. These are exploratory kernel averages, not an end-to-end throughput result. The prototype is not an accepted fix; matching the arithmetic with far fewer cross-lane operations remains necessary. Machine-local details: worktrees/codex/rad46-groupfold/artifacts/logs/manual/ and worktrees/codex/rad46-stageprobe/artifacts/logs/manual/.

### Mapped outlier routes and output-lane experiment (2026-11-03)
A separate diagnostic build reproduced the candidate tail p99.9 of 3.71049. Scheduler step, ticket, and helper-side request records joined 5318 captures through 131 tickets with no missing mappings. Position 5458 used tail experts (layer 15, expert 280) and (layer 40, expert 491). Position 5459 used five tail routes, including layer 11 experts 185 and 424. This is correlation; earlier layers and tokens can propagate error.

On captured 64-row requests containing these positions, matched two-rank stage replays found one gate BF16 difference for layer 11 expert 424 at token 18 (position 5459) on rank 0, without an intermediate quantisation change in that group. In the layer-15 request, the quantisation differences for expert 280 occurred at another token. Per-expert reference relative L2 for expert 280 over four single-tail-route tokens was 0.001521 / 0.000224 for grouped GEMM and 0.001521 / 0.0000686 for the first rows candidate (ranks 0 / 1). These figures are not a causal attribution.

An opt-in output-owned batched rows prototype matched grouped-GEMM output hashes on both split ranks for the two outlier requests and on three earlier full-width captures. For the two 64-row requests, short non-interleaved kernel traces had gate/up medians 80 us versus 90 us and down medians 37 us versus 100 us for the original rows kernel. For three 2048-row captures its gate/up median was 1479 us versus 684 us; down was 506 us versus 558 us. These traces contain few samples and do not establish end-to-end performance. The production-layout KLD and balanced decode/prefill A/B remain required. Machine-local route evidence: artifacts/RAD46-route-attribution/ in the attribution worktree; replay and profiler evidence is in the stageprobe and outputlane worktrees.

### Output-lane first production capture (2026-11-03)
The first attested three-arm MEN26 quality capture passed all six helper-KLD limits: tail p99.9 1.90005385, versus helper-off 2.79228854 and two-XTX 3.19804552. Both controls matched the previous per-position files bit for bit. The gfx1030 helper log records the large-lane rows setting and output-lane selection; candidate output rows match the grouped-GEMM diagnostic bit for bit. This is one capture only; determinism and the matched prefill/decode A/B remain pending. The original activation message did not itself require the rows setting, so the independent rows-enabled log is necessary for interpreting this capture. Evidence: outputlane worktree artifacts/RAD46-outputlane/capture-1/.

### Output-lane repeat quality capture (2026-11-03)
A second attested three-arm MEN26 capture passed every helper-KLD limit. Byte comparisons of each arm's per-position .rows files against capture 1 were identical. The gfx1030 helper independently logged both large-lane rows enabled and output-lane selected. Matched end-to-end decode and prefill throughput remains outstanding. Evidence: outputlane worktree artifacts/RAD46-outputlane/capture-{1,2}/.

### Matched output-lane throughput comparison (2026-11-03)
Four interleaved production-layout pairs completed successfully. Mean pp1024 throughput fell from 1928.32 to 1320.18 tokens/s (-31.54%); mean tg512 increased from 75.92 to 83.20 tokens/s (+9.59%). All candidate tail helper logs recorded rows enabled and output-lane selection. This candidate fails the no-prefill-regression acceptance criterion despite passing deterministic KLD. In pair 1, large-lane tail gate/up rose from 2282.3 to 6481.6 us/request while route row totals matched. An isolated RB16 weight-reuse experiment is queued, unvalidated. Evidence: outputlane worktree artifacts/RAD46-outputlane/throughput-ab/.

### Larger-batch and quad-DPP experiments (2026-11-03)
The RB16 output-lane variant built and matched both outlier request hashes, but a short large-request trace showed a layer-13 gate/up dispatch around 12 ms versus 2–3 ms for RB8. A four-lane quad-DPP prototype matched both outlier requests and three 2048-row captures. A revised ordered assembly fold reduced allocated VGPRs from 216 to 120 without scratch and matched the accurate output-lane hashes in the same captures. In a small three-way trace its median gate/up and down were 1084/494 us, versus original rows 710/488 us and output-lane 1567/502 us; these are exploratory kernel timings, not end-to-end acceptance evidence. Further optimization and full quality/throughput validation remain necessary. Evidence: worktrees codex/rad46-outputlane-rb16, rad46-quaddpp and rad46-quadasm artifacts/logs/manual/.

- chg_20261103_034702_improve-accuracy-of-the-rdna2_5987
- 2026-11-03T03:47:06.402874+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4225
- 2026-11-03T04:16:37.951882+00:00 (updated-by): Updated: section:notes
- RV4226
- 2026-11-03T04:23:54.990531+00:00 (updated-by): Updated: section:notes
- RV4227
- 2026-11-03T04:50:36.770110+00:00 (updated-by): Updated: section:notes
- RV4228
- 2026-11-03T05:16:27.079680+00:00 (updated-by): Updated: section:notes
- 2026-11-03T05:44:58.533827+00:00 (updated-by): Updated: section:notes
- 2026-11-03T05:51:22.661842+00:00 (updated-by): Updated: section:notes
- 2026-11-03T06:25:08.429090+00:00 (updated-by): Updated: section:notes
- 2026-11-03T07:19:18.078059+00:00 (updated-by): Updated: section:notes
