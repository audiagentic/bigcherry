# hip-probes

Standalone HIP programs that test one hardware or runtime question at a time, outside llama.cpp and radiance, so a
design decision does not have to be read out of an engine run (QFP41 threading, QFP49 cross-card sum).

`hip_overlap_probe.cpp` (one file, built by `run.sh` with `hipcc`):

| test | question |
|---|---|
| `links` | How fast does a card exchange one prefill-sized message (5.24 MB) with pinned host memory: up, down, both at once, and with every card doing it together? |
| `overlap` | Does a transfer on its own stream run beside kernels on the compute stream, and does either slow down? |
| `submit` | Many small kernels on every card: one host thread for all cards against one thread a card. |
| `graphs` | One thread a card capturing, instantiating and launching HIP graphs at once, with and without a process-wide lock: are the results right? |
| `stagger` | A loop of compute + exchange on every card as today, against two half-batches staggered so one half's exchange runs under the other half's compute. |

Run on Brutus through the queue, on the cards to test:

    VIS=0,1,2 SCRIPT hp1 tools/lab/hip-probes/run.sh @<any build run> <out-dir> [test...]

The compute kernel streams over 256 MiB, so it stands for a memory-bound kernel; its time is not the model's. The
ratios (what the exchange adds, what staggering recovers) are what carry over, the milliseconds are not.
