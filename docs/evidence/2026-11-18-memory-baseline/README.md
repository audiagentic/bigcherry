# Host memory bandwidth baseline

Captured 2026-11-18 to compare this host before and after increasing RAM from the current configuration. This is a STREAM-style host-memory throughput measurement; it does not identify installed DIMM count, rank, timings, channel population, or configured DRAM data rate.

## Visible system configuration

- CPU: 12th Gen Intel Core i7-12700KF; Linux reports 1 socket, 12 cores, and 2 threads per core. 20 logical CPUs were online/available to this run.
- Memory visible to the environment: 91 GiB total, about 74 GiB available at capture time.
- Linux CPU frequency governor: `powersave`.
- DIMM model/count, BIOS memory rate, and timings: unavailable. The environment does not expose SMBIOS/DMI tables or `dmidecode`; do not treat bandwidth as a substitute for those configuration details.

## Method

`stream.c` runs Copy, Scale, Add, and Triad loops over three 512,000,000-byte arrays (1.43 GiB total), with OpenMP and 20 threads. Each run performs seven repetitions per kernel. Reported bandwidth counts the conventional STREAM traffic per element (two arrays for Copy/Scale/Triad, three for Add). The table shows the best repetition within each run; means are also included to expose run variability. The 512 MiB working arrays exceed the reported 25 MiB L3.

Build and rerun from the repository root:

```sh
mkdir -p work/tmp
gcc -O3 -march=native -fopenmp -std=c11 \
  docs/evidence/2026-11-18-memory-baseline/stream.c \
  -o work/tmp/bigcherry-stream
OMP_NUM_THREADS=20 OMP_PROC_BIND=close OMP_PLACES=cores \
  work/tmp/bigcherry-stream 64000000 7
```

Keep the compiler flags, element count, repetitions, thread settings, power policy, and background load consistent in the follow-up measurement. Record any changed RAM speed/timings and slot population from BIOS or a DMI-capable host alongside the next result.

## Results

Three runs, each with seven repetitions per kernel. Rates are decimal GB/s.

| Kernel | Run 1 best (mean) | Run 2 best (mean) | Run 3 best (mean) | Mean of run bests |
| --- | ---: | ---: | ---: | ---: |
| Copy | 67.61 (61.63) | 68.29 (65.42) | 69.69 (68.13) | 68.53 |
| Scale | 66.58 (62.78) | 68.98 (64.64) | 67.50 (65.39) | 67.69 |
| Add | 70.54 (68.59) | 72.30 (68.30) | 71.45 (69.76) | 71.43 |
| Triad | 47.08 (46.10) | 47.85 (44.88) | 46.55 (45.36) | 47.16 |

Use Triad as the primary before/after comparison (mean of run-best values: 47.16 GB/s); the other kernels provide context. This is an empirical result for this OS/runtime and load, not a guaranteed DRAM peak or a DIMM speed readout.
