# HL02 — three-session mapped-host contention and CPU BAR-push controls (2026-11-13)

**Measured on Brutus under exclusive `gpu` leases.** This is new experimental
evidence following PR #951, not a TP3 host-reduced implementation or inference
benchmark. See [HL01](../../planning/completed/run-radiance-host-link/HL01.md)
for the physical slot-swap diagnosis and [HL02](../../planning/active/run-radiance-host-link/HL02.md)
for the remaining gates.

## Provenance and reproducibility

- Lab-owned source/runner: `tools/lab/radiance/hl02-host-attributes/`
  (`host_attributes.hip`, `run_matrix.py`, `run_push_control.py`,
  `analyse_matrix.py`).
- Source baseline: merged PR #951, git commit `523ac2fbd`. A fresh
  experiment branch adds the lab harness, analysis and this evidence.
- Build: `hipcc -O2 --offload-arch=gfx1100 --offload-arch=gfx1201`;
  HIP compiler version `7.2.53211-97f5574fe2`, Linux `7.0.13-gfxoff`.
  New HIP probe executable SHA256
  `1468ffaaee3de7d9a1c81a6f401fc2018071cb9e298cbf7586e4803daa4efffb`.
- RAD50 existing CPU-write comparison: archived `rad50-tester` binary SHA256
  `e1e349b2686dfa1d7d2192da523290fc97502cfa1c7874bcfa784f0aa93af92c`;
  original safety and performance history:
  `docs/evidence/2026-11-10-rad50-cpu-bar-push/`. Same independent GPU
  verifier and DMA-BUF mapping protocol; this experiment supplies controlled
  primary-GPU background traffic.
- Device roles in this run: CLI primary 0 = first PEG-half XTX, victim 1 =
  second PEG-half XTX, victim 2 = R9700 on separate CPU x4. This mapping is
  **run data**, never a general runtime assumption. No foreign KFD processes
  were present in preflight; every individual measurement acquired and
  released the exclusive GPU lease. Final lease state: free.
- Three complete interleaved sessions per victim/direction, forward/reverse/
  forward HIP allocation mode order; each condition transferred a mapped
  64 MiB working set repeatedly for at least 0.6 s device/wall timed, after
  GPU initialization and full-word check. Background kernel transferred a
  separate mapped 64 MiB buffer with source GPU reads or writes; remained
  active throughout each victim measurement. Each run records its **actual**
  background GB/s, rather than assuming the requested activity implies load.
- Serial RAD50 path: three sessions per receiver (forward/reverse/forward
  background order), each with idle, primary-read and primary-write controls
  and a second idle repetition. Each RAD50 arm used 10 iterations of a
  4 MiB payload; scalar CPU preparation, GPU consumption, synchronization
  and full-word verification are included in its own serial sequence time.
- `attributes.csv` — all **108** GPU host bandwidth observations, including
  device and wall GB/s, allocation flag bits, individual session ID and
  GPU-word error count. `backgrounds.csv` — all **36** measured primary
  background trials. `rad50-serial-push.csv` — all **24** RAD50 serial
  pull/push comparisons, including 10-iteration p50 and sequence speedup.
  These are normalized complete numeric receipts extracted from the strict
  JSONL run output, with host-specific paths and transient PCI enumerations
  omitted; `analyse_matrix.py` rejects incomplete/error results. An offline
  negative test injected one nonzero `gpu_word_errors` value into a **copy**
  of the receipts: the analyser exited nonzero and identified the corrupt
  `xtx-read-1.jsonl` session before producing output. The original data
  were untouched.

## Test 1 — coherent/non-coherent/write-combined mapped-host bandwidth

All numbers are **median device GB/s over three independent sessions**.
`idle` means no primary GPU workload; `primary reads` and `primary
writes` mean full-rate 64 MiB primary GPU kernel traffic.

| Victim and operation | HIP mapping | Idle | Primary reads | Primary writes |
|---|---|---:|---:|---:|
| Second XTX read | coherent | 13.428 | 6.748 | 13.422 |
| Second XTX read | non-coherent | 13.426 | 6.750 | 13.418 |
| Second XTX read | write-combined | 13.429 | 6.749 | 13.422 |
| Second XTX write | coherent | 14.263 | 6.740 | 14.259 |
| Second XTX write | non-coherent | 14.253 | 6.743 | 14.248 |
| Second XTX write | write-combined | 14.261 | 6.743 | 14.259 |
| R9700 read | coherent | 6.742 | 3.394 | 6.741 |
| R9700 read | non-coherent | 6.742 | 3.394 | 6.740 |
| R9700 read | write-combined | 6.742 | 3.390 | 6.740 |
| R9700 write | coherent | 7.141 | 3.470 | 7.141 |
| R9700 write | non-coherent | 7.142 | 3.456 | 7.140 |
| R9700 write | write-combined | 7.141 | 3.463 | 7.141 |

All **108/108** complete observations had **zero GPU verification errors**;
all 36 background controls also reported zero errors and completed
successfully. Primary XTX actual read bandwidth across the controls was
**13.421–13.442 GB/s**, and primary write bandwidth was
**14.259–14.265 GB/s**.

**Finding A (HIP mapping-attribute gate): NO-GO for a simple flag-only
solution.** Under primary GPU host reads, switching the *victim's* mapped
host allocation from `hipHostMallocCoherent` to
`hipHostMallocNonCoherent` or `hipHostMallocWriteCombined` did **not**
meaningfully recover victim throughput. It remained at approximately half
idle. Do **not** promote a non-coherent mailbox solely based on the flags.

This does not prove these flags changed the actual PCIe Transaction Layer
No-Snoop attribute or remove all possible coherent-snoop strategies.
Transaction attributes were not captured independently; a specific
transaction-level hypothesis requires a separate instrumented test.

**Finding E (trigger control): POSITIVE diagnosis for these workloads.**
Primary GPU **writes alone**, at ~14.26 GB/s, did not reduce other GPU
host reads or writes. Primary GPU **reads**, at ~13.43 GB/s, halved both.
The result independently reproduces the previous HL01 root-port read
trigger and narrows it from generic GPU transfer activity to this read
transaction pattern. It does **not** identify an exact CPU arbitration
queue or prove duty-cycle behavior.

## Test 2 — existing RAD50 4 MiB CPU BAR-push control

Median of each session's 10-iteration serial-sequence p50 values; idle
includes two controls/session (n=6), while each background arm has
n=3 sessions. The speedup column is the median of the paired speedup
reported by RAD50, **not** a ratio derived from median components.

| Receiver | Primary GPU activity | GPU pull p50 (µs) | CPU push p50 (µs) | Median push speedup |
|---|---|---:|---:|---:|
| Second XTX | idle | 1095.20 | 790.90 | 1.38x |
| Second XTX | host read | 1378.13 | 745.04 | 1.85x |
| Second XTX | host write | 1066.35 | 789.32 | 1.35x |
| R9700 | idle | 1316.18 | 825.64 | 1.54x |
| R9700 | host read | 1885.35 | 818.89 | 2.30x |
| R9700 | host write | 1251.60 | 819.02 | 1.53x |

All 24 RAD50 comparisons returned `RAD50_AB_COMPLETE` and exit 0.
Background GPU reader sustained **13.425–13.434 GB/s** during these
RAD50 comparisons; the write-only primary sustained
**14.260–14.265 GB/s**. No background was silently substituted for an
idle comparison. The separate CPU process, DMA-BUF mapping, per-transfer
SYNC, GPU payload verification and sequential handoff were unchanged.

**Finding D: positive limited bypass evidence.** At this 4 MiB *serial
synthetic* shape, CPU BAR push retains roughly its idle p50 while the
mapped-host GPU-pull baseline worsens with primary GPU read contention.
The relative sequence speedup is ~1.85x on the other XTX and ~2.30x on
R9700. This supports advancing CPU push to the realistic concurrent
producer/push experiment, but is not evidence of three-way aggregate BAR
capacity or TP3 layer/TTFT improvement.

## What this evidence does not prove

- No measured PCIe No-Snoop transaction attribute, CPU-internal queue
  selection, non-coherent control-memory protocol or partial-read duty
  cycle. The tested flags alone do not mitigate the problem.
- No **simultaneous push to all three ranks** with the GPU partial
  producers active, CPU FP32 sum, 10 MiB transfer, real GPU-local ready
  tickets or bounded all-rank abort/drain. These remain HL02 B/C and VTP38.
- No per-layer production critical-path gain, exact multi-rank BF16
  arithmetic, 12-run stress or KLD admission. VTP38 must compare against
  the improved validated VTP36 rad0032/rad0033 GPU protocol.
- A limited 4 MiB synthetic transfer and three sessions do not justify
  production p99 or a default transport switch.

**Next recommended experiment:** Build/extend the single existing RAD50
laboratory tester for **simultaneous** CPU pushes to all XTX/XTX/R9700
DMA-BUF targets while all three GPUs write 320 KiB–10 MiB partials
into mapped host slots; measure per-destination rate, CPU memory
bandwidth/affinity, slowest-rank completion and generation/fault safety.
Do not start with an engine patch or treat the negative HIP-attribute
result as an unconditional CPU microarchitecture proof.
