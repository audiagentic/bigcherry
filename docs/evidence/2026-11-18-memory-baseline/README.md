# Host memory bandwidth baseline

Captured 2026-11-18 to compare this host before and after increasing RAM from the current configuration. This is a STREAM-style host-memory throughput measurement. Linux exposes the DIMM SPD EEPROMs, which identify module capabilities; it does not expose the BIOS-trained settings currently in use.

## Visible system configuration

- CPU: 12th Gen Intel Core i7-12700KF; Linux reports 1 socket, 12 cores, and 2 threads per core. 20 logical CPUs were online/available to this run.
- Memory visible to the environment: 91 GiB total, about 74 GiB available at capture time.
- Linux CPU frequency governor: `powersave`.
- Board: Gigabyte Z690 AERO D; BIOS F30 (reported date 2025-06-19).
- Two read-only SPD EEPROMs are visible. Both report DDR5 UDIMM modules, 48 GiB each (96 GiB total), with matching decoded module geometry and advertised profile data.
- SPD JEDEC timing: DDR5-5200, 42-42-42-84 (CL-RCD-RP-RAS).
- SPD contains higher XMP/EXPO profiles, including DDR5-6400 32-39-39-84 and DDR5-6000 42-46-46-76. These are advertised profiles, not evidence that either is currently selected.
- Active BIOS-trained data rate and timings, slot names, and channel population remain unavailable. The environment exposes neither SMBIOS/DMI tables nor memory-controller configuration; the SPD devices are read-only here.

## Method

`stream.c` runs Copy, Scale, Add, and Triad loops over three 512,000,000-byte arrays (1.43 GiB total), with OpenMP and 20 threads. Each run performs seven repetitions per kernel. The original harness counted two arrays for Copy/Scale/Triad and three for Add. This undercounts Triad traffic; see the correction note below. The table shows the best repetition within each run; means are also included to expose run variability. The 512 MiB working arrays exceed the reported 25 MiB L3.

Build and rerun from the repository root:

```sh
mkdir -p work/tmp
gcc -O3 -march=native -fopenmp -std=c11 \
  docs/evidence/2026-11-18-memory-baseline/stream.c \
  -o work/tmp/bigcherry-stream
OMP_NUM_THREADS=20 OMP_PROC_BIND=close OMP_PLACES=cores \
  work/tmp/bigcherry-stream 64000000 7
```

Keep the compiler flags, element count, repetitions, thread settings, power policy, and background load consistent in the follow-up measurement. Record the BIOS-selected speed/timings and slot population from firmware setup or a host tool with memory-controller access alongside the next result.

## Results

Three runs, each with seven repetitions per kernel. Rates are decimal GB/s.

| Kernel | Run 1 best (mean) | Run 2 best (mean) | Run 3 best (mean) | Mean of run bests |
| --- | ---: | ---: | ---: | ---: |
| Copy | 67.61 (61.63) | 68.29 (65.42) | 69.69 (68.13) | 68.53 |
| Scale | 66.58 (62.78) | 68.98 (64.64) | 67.50 (65.39) | 67.69 |
| Add | 70.54 (68.59) | 72.30 (68.30) | 71.45 (69.76) | 71.43 |
| Triad | 47.08 (46.10) | 47.85 (44.88) | 46.55 (45.36) | 47.16 |

The historical Triad mean of run-best values is 47.16 GB/s under the original two-array counting convention; do not use it as an unqualified standard STREAM comparison. This is an empirical result for this OS/runtime and load, not a guaranteed DRAM peak or a DIMM speed readout.


## Method correction identified during migration planning

Triad (`a = b + 3*c`) reads two arrays and writes one, so the conventional
traffic multiplier is three, not the two used by this harness. The table is
preserved as originally observed. Multiplying its Triad rates by 1.5 converts
the counting convention only; it is not a new benchmark run or retained raw
time measurement. BHK04 owns the small harness correction and a fresh comparable
before/after sample. No benchmark or live RAM change was performed for this note.
