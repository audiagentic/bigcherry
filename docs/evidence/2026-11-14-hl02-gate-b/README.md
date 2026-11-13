# HL02 Gate B — concurrent CPU BAR push versus GPU pull at all-reduce sizes (2026-11-14)

**Measured on the TP3 host under exclusive `gpu` leases.** Synthetic transfer evidence for HL02 Gate B and VTP38's
design, not an all-reduce implementation or an inference benchmark.

## Setup

- Harness: `tools/lab/radiance/hl02-host-attributes/threeway_push.hip`, runner `run_threeway.sh`, analyser
  `analyse_threeway.py` (rejects any incomplete session or verification error). Built with ROCm 7.2.4 hipcc
  `-O2 -mavx2` for gfx1100 and gfx1201; executable SHA-256 `7f78ba155128f36e9d319a7f1dbcec01e6cabb36fe8d084f38a526db17edbc9d`. The committed source differs from
  the measured one only in one output buffer size (160 to 200 bytes, for the model-constant guard).
- Device roles in this run (run metadata, not an engine constant): device 0 the XTX on the PEG half whose host reads
  halve the other cards' host traffic (HL01), device 1 the other XTX, device 2 the R9700 on the CPU x4 port.
- Receive buffer: a card-local VMM allocation exported as a DMA-BUF and mapped by the CPU (RAD50's path, in one
  process). CPU push: one thread per pushing card, pinned to P-cores 0, 2, 4, 32-byte non-temporal stores from a
  cached host buffer, store fence, `DMA_BUF_IOCTL_SYNC` start/end inside the timed region. GPU pull: a 64 x 256 kernel
  copies the same message from mapped coherent host memory. All transfers in an arm start together from one barrier.
- Every word of every transfer is verified on the receiving GPU after the timed region (generation-unique pattern).
- Arms: `pull_all`; `push_all`; `push_one_pull_rest` (CPU pushes to device 0, the others pull); `push_pair_pull_rest`
  (devices 0 and 2 pushed, device 1 pulls); the push-all and pull-all arms with **GPU producers** (all three cards
  writing mapped host memory at full rate throughout, a stress far above an all-reduce's partial writes);
  `cpu_sum3` (three CPU threads sum three BF16 partials to one, fp32 accumulate, round to nearest even).
- Sizes 320 KiB, 1, 4 and 10 MiB; 30 iterations per arm and size, arm order rotated each iteration; three sessions
  (forward, reverse, forward). First iteration excluded as warm-up. Table: median across sessions of each session's p50.
- Result: 3 sessions complete, **0 verification errors**; 3 producer-arm transfers ran behind a producer instead of
  beside it and are flagged and excluded.

## Results

Full table: [summary.md](summary.md); every transfer: [threeway.csv](threeway.csv).

| 10 MiB (4 MiB) | slowest card | per card GB/s, device 0 / 1 / 2 |
|---|---|---|
| all three pull | 2007 us (816) | 12.7 / 8.7 / 5.3 |
| CPU pushes to all three | 1998 us (804) | 5.3 / 5.8 / 5.6 |
| CPU pushes to device 0, others pull | **1670 us (678), -17%** | 10.5 / 12.6 / 6.6 |
| CPU pushes to devices 0 and 2, device 1 pulls | 1976 us (812) | 5.3 / 12.6 / 5.6 |
| CPU pushes to all three beside full-rate GPU producers | 2068 us (1006) | 5.1 / 5.6 / 5.4 |
| all three pull beside full-rate GPU producers | 19860 us (4059) | 1.1 / 1.1 / 0.5 |
| CPU sum of three BF16 partials (3 threads) | 807 us (340) | 39 GB/s read |

At 1 MiB push-to-one is -10% (206 against 228 us); at 320 KiB it is no better (94 against 91 us).

## Findings

1. **Push to the card that triggers the contention, pull the rest.** Taking device 0 off the host-read path restores
   device 1's pull to 12.6 GB/s (its idle rate) and lifts the R9700's from 5.3 to 6.6 GB/s; the R9700 then sets the
   makespan. -17% at 4-10 MiB, -10% at 1 MiB.
2. **Concurrent CPU pushers halve each other.** One pusher writes 10.5 GB/s; two or three write ~5.3-5.8 GB/s each,
   also when the two targets sit behind different root ports (devices 0 and 2). Push to all three is no faster than
   all-pull, and push to two loses the push-to-one gain. The limit is on the CPU/host side of concurrent BAR writes
   (not identified further here).
3. **CPU push survives host-write load; GPU pull does not.** With all three cards writing host memory flat out, pushes
   keep ~5 GB/s while pulls fall to ~1 GB/s.
4. **The CPU sum costs ~0.8 ms per 10 MiB** at three threads (~39 GB/s of reads), so a host-reduced all-reduce only
   competes if the sum overlaps the transfers.

## Consequences

- VTP38: the push-to-all end state is withdrawn; the measured candidate is a mixed receive (CPU push to the
  contention-triggering rank, GPU pull for the others), chosen per rank from a host profile.
- HL02 Gate B: push-to-one, push-to-pair and push-to-all measured with producers and per-card completion. SDMA, two
  threads for one x8 card, CPU time and memory bandwidth were not measured.

## Incident during the first attempt

The first harness version's producer kernel ran until a host flag stopped it. In one arm a pull kernel queued behind
it on the same card, the run hung, and killing the process left the R9700 unable to complete a GPU reset (its SMU did
not respond, reset returned -62); no process could open the GPU device until the host was rebooted. The producer is
now bounded by device time (60 ms) and a transfer that runs behind it is reported as `serialized`. No data from that
attempt is used here.
