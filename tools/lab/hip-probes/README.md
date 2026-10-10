# hip-probes

Standalone HIP programs that test one hardware or runtime question at a time, outside llama.cpp and radiance, so a
design decision does not have to be read out of an engine run (QFP41 threading, QFP49 cross-card sum).

`hip_overlap_probe.cpp` (one file, built by `run.sh` with `hipcc`):

| test | question |
|---|---|
| `links` | How fast does a card exchange one prefill-sized message (5.24 MB) with pinned host memory: up, down, both at once, and with every card doing it together? |
| `wire` | What does a 16-bit wire cost and save: a round trip as f32 against f16 with the conversions done on the card. |
| `overlap` | Does a transfer on its own stream run beside kernels on the compute stream, and does either slow down? |
| `submit` | Many small kernels on every card: one host thread for all cards against one thread a card. |
| `graphs` | One thread a card capturing, instantiating and launching HIP graphs at once, with and without a process-wide lock: are the results right? |
| `stagger` | A loop of compute + exchange on every card as today, against two half-batches staggered so one half's exchange runs under the other half's compute. |

Run on Brutus through the queue, on the cards to test:

    VIS=0,1,2 SCRIPT hp1 tools/lab/hip-probes/run.sh @<any build run> <out-dir> [test...]

The compute kernel streams over 256 MiB, so it stands for a memory-bound kernel; its time is not the model's. The
ratios (what the exchange adds, what staggering recovers) are what carry over, the milliseconds are not.

## Results, 2026-10-11 (Brutus: 2x RX 7900 XTX at PCIe 4.0 x8, R9700 at x4, RX 6900 XT at 3.0 x4; runs hp1, hp2)

**links** (one 5.24 MB message; the production prefill sum travels as bf16, 2.62 MB):

| card | 2.62 MB up / down / both at once | 5.24 MB up / down / both at once | 5.24 MB, every card at once |
|---|---|---|---|
| RX 7900 XTX | 0.19 / 0.20 / 0.21 ms | 0.37 (14.1 GB/s) / 0.40 (13.1 GB/s) / 0.41 ms | both at once 0.59 ms |
| R9700 | 0.40 / 0.41 / 0.54 ms | 0.78 (6.7 GB/s) / 0.79 (6.6 GB/s) / 1.07 ms | both at once 1.21 ms |
| RX 6900 XT | 0.74 / 0.76 / 0.82 ms | 1.48 (3.6 GB/s) / 1.50 / 1.64 ms | both at once 1.72 ms |

Up and down overlap fully on an XTX and partly on the R9700. The R9700's link sets the floor of a three-card
exchange: about 0.54 ms for a bf16 prefill sum, where RCCL's kernel measures 0.89 ms after the last card arrives
(a ring passes 4/3 of the message over each link).

**overlap**: a transfer on its own stream runs beside the compute stream with no cost to either, on every card. XTX:
compute alone 5.20 ms, transfers alone 6.17 ms, started together all done at 6.23 ms (11.37 one after the other).

**submit**: 4,000 small kernels a card on four cards. One host thread for all: submitted in 12.5 ms (0.8 us a
launch), finished at 18.6 ms. One thread a card: 7.0 ms (1.8 us a launch), finished at 15.0 ms. Submission threads
buy little even when the work is nothing but launches.

**graphs**: one thread a card, 300 rounds of capture + instantiate + launch of a 12-kernel graph, with another
thread's ordinary stream call landing during the capture. Relaxed capture (llama.cpp's mode) and thread-local
capture: 0 of 900 rounds wrong, 0 errors, with or without a process-wide lock. Global capture mode makes the other
thread's call fail, which is that mode's contract (run hp1). So plain concurrent capture is not what breaks the
1356 dispatch workers; what this probe does not cover is updating an instantiated graph's parameters from several
threads.

**stagger**: 96 rounds of [compute, then every card exchanges one message through the host] on the three target
cards. Compute only 174.8 ms. As today (the card waits for the exchange): 342.5 ms, the exchange adds 96%. Two
half-batches staggered: 182.9 ms, the exchange adds 5%. Staggering recovers 95% of what the exchange costs.
