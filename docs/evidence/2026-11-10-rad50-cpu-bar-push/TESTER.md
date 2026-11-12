# RAD50 single test executable

**Built and exercised on GPUs under exclusive leases; hardware results are gate-specific.**
The combined binary contains both gfx1100 and gfx1201 device images and
reuses the experiment's existing implementations, without a second
test executable per architecture. The remote shell **currently permits the one exact executable name `rad50-tester`**;
its `--list`, `--self-check` and `combine-ref` modes have run successfully.
No separate per-test permission is needed for these compiled-in modes.
This is not a way to bypass remote command policy.

The executable's source is `rad50_tester.cpp`; `Makefile` links
`pool_inventory.cpp`, `bar_write_canary.hip`,
`bar_ipc_canary.hip`, `bar_transfer_ab.hip`, `bar_dmabuf_ab.hip`,
`bar_live_ready_canary.hip`, `bar_vmem_cap.hip`, `combine_reference_test.cpp` and
`cpu_handoff_bench.cpp` directly, with their original `main`
functions renamed at compile time. Future *reviewed, compiled-in*
RAD50 tests can use the same binary name after a rebuild. It does
**not** forward commands to a shell, invoke arbitrary user-specified
program paths, load user plugins or run independently restricted
test executables. The IPC case re-executes **this same binary**, with
a fixed internal child mode and a passed Unix socket, as required
to test separate-process HSA CPU memory access.

## Build from the RAD50 worktree (no GPU required)

```sh
source tools/env/bigcherry-env.sh
make -C tools/lab/rad42-bar-push \
  OUT_DIR="$BC_WORK_ROOT/builds/rad42-bar-push" rad50-tester
```

The initial test binary was built with `OPT=-O1` for lower-cost
proof preparation. Before making a performance claim from `transfer` or
`cpu-handoff`, rebuild the **same binary** using `OPT=-O2` and
`make -B` (while no other GPU benchmark lease holder is measuring),
and record the build flags with the evidence. For example:

```sh
make -B -C tools/lab/rad42-bar-push \
  OUT_DIR="$BC_WORK_ROOT/builds/rad42-bar-push" OPT=-O2 rad50-tester
```

The executable basename and permissions do not change.

The only executable to allowlist and invoke is:

```text
$BC_WORK_ROOT/builds/rad42-bar-push/rad50-tester
```

The binary basename, not the architecture or test mode, is
`rad50-tester`. Compiling it is not a GPU correctness test.
`--help`, `--list`, and `--self-check` are card-free.

## Usage

Use the absolute path for the IPC test (the child re-executes itself).
The commands below set a shell variable for brevity.

```sh
T="$BC_WORK_ROOT/builds/rad42-bar-push/rad50-tester"

"$T" --help
"$T" --list
"$T" --self-check             # 34 strict argument cases, no GPU

# For every GPU test below: CHECK GPU USE, ACQUIRE AND HOLD THE
# EXCLUSIVE 'gpu' LEASE FIRST. '--lease-confirmed' is an explicit
# operator acknowledgement, NOT an automatic lease check or acquisition.
"$T" --test pools --lease-confirmed
"$T" --test hip-smoke --device 2 --lease-confirmed
"$T" --test serial --device 0 --lease-confirmed
"$T" --test serial --device 1 --lease-confirmed
"$T" --test serial --device 2 --lease-confirmed
"$T" --test ipc --device 0 --lease-confirmed
"$T" --test ipc --device 1 --lease-confirmed
"$T" --test ipc-map --device 0 --lease-confirmed
"$T" --test ipc-gpu-cpu --device 0 --lease-confirmed
"$T" --test vmem-cap --device 0 --lease-confirmed
"$T" --test transfer --device 1 --iterations 30 --lease-confirmed
"$T" --test live-ready --device 0 --lease-confirmed

# Reference without GPU; host-only bench still needs exclusive lease.
"$T" --test combine-ref
"$T" --test cpu-handoff --mode yield --samples 3000 --lease-confirmed
```

| Mode | Argument requirements | What it establishes |
| --- | --- | --- |
| `pools` | `--lease-confirmed` | API discovery only |
| `hip-smoke` | `--device N --lease-confirmed` | HIP-only launch, no HSA pointer or BAR |
| `serial` | `--device N --lease-confirmed` | Serial CPU BAR write then GPU verification |
| `ipc` | `--device N --lease-confirmed` | Separate CPU-only child HSA IPC attach, serial GPU verify |
| `ipc-map` | `--device N --lease-confirmed` | Same-GPU-agent cross-process import/CPU grant metadata; no stores |\n| `ipc-gpu-cpu` | `--device N --lease-confirmed` | Guarded 4 KiB CPU write via imported host VA, parent GPU verifies only on success |\n| `vmem-cap` | `--device N --lease-confirmed` | GPU-local VMM/DMA-BUF export, mapping and CPU access query; no stores |\n| `transfer` | `--device N --iterations 10..200 --lease-confirmed` | Single-process transfer A/B, NOT end-to-end serving |
| `live-ready` | `--device N --lease-confirmed` | GPU running before CPU stores, single-process readiness canary |
| `combine-ref` | No extras | Actual FP16 combine format, NaN and fallback parity (CPU) |
| `cpu-handoff` | `--mode spin|yield|sleep1|condvar --samples 100..100000 --lease-confirmed` | CPU-to-CPU wake overhead only, NOT GPU notification |

Arguments are **strictly parsed**, with known mode names and bounded
numeric ranges; unrecognized arguments, duplicate fields, scripts
and arbitrary executable names are rejected. There is no 'all' mode:
each safety gate must explicitly succeed before moving to the next.
Success code 0, syntax error 2; test failures/nonzero statuses remain
nonzero (e.g. 3), and a skipped CPU feature check may return 77.

HIP ordinals must be verified against the observed PCI identities at
test time. On the measured host, 0 and 1 are the RX 7900 XTX cards,
2 is the R9700. An HSA pool query does not prove BAR store correctness.

## Safety and progression

- Check `agent-lease status`, GPU utilisation and GPU PIDs **outside**
  the binary; obtain an exclusive `gpu` lease before running any GPU
  or host benchmark mode. Release it afterwards. Do not interfere with
  foreign leases or active serving. This runner does **not** check lease
  ownership; `--lease-confirmed` only blocks accidental unacknowledged
  GPU execution.
- Gate 2 passed on both XTX cards using the standalone canary. The
  R9700 HIP-only dispatch **passes**, but its HSA BAR verification
  did not complete. Treat them as separate gates.
- Gate 2b `ipc` **failed** on XTX0 with HSA 4104 (CPU-agent import).
  Same-GPU IPC import passes on both XTXs but XTX0's imported host VA
  is not CPU-writable. VMM DMA-BUF export works but CPU access query
  fails 4100 by those APIs, but the **DMA-BUF mmap alternative now passes cross-process 4 KiB and 4 MiB CPU writes and parent GPU verification on both XTXs**.
  Never proceed to `transfer` on an unsupported mapping. The live
  ready mode tests observation of concurrent publication, not a
  formal guarantee of posted-write ordering.
- Record PCI identity, HIP/ROCm version, source commit, input
  arguments, exit code, and raw output in the RAD50 journal.
- It remains a **lab-only executable**; no serving process changes.
  This runner never pushes, merges, edits production files, or
  automatically runs more gates.
- A future test must be added to a fixed source dispatch case and
  reviewed before rebuilding `rad50-tester` with the same basename;
  no generic subcommand executing other binaries will be added.

## Build/verification status

- Linked HIP fat binary with `--offload-arch=gfx1100` and
  `--offload-arch=gfx1201` into the project work root.
- C++ syntax check passed and link succeeded. The 23 existing
  Python card-free gate tests still apply.
- `--list` returned the compiled modes, `--self-check` passed 34/34
  strict argument cases, and `--test combine-ref` passed seven FP16
  reference checks (2,560 elements); each exited 0 without any GPU.
  Selected hardware modes have since run under exclusive leases;
  see `PROGRESS.md` for their exact outcomes. Recheck lease/processes
  before any further hardware mode.
  No subprocess/alias/interpreter was used to evade policy.

## 2026-11-09 DMA-BUF interprocess probe (Gate 2b partial pass)

The single executable also provides `--test dmabuf-map` (no CPU stores), `--test dmabuf-write` (4 KiB) and `--test dmabuf-full` (4 MiB). Each requires `--device N --lease-confirmed`. A GPU-local HSA VMM handle is exported as DMA-BUF, passed via SCM_RIGHTS to a CPU-only child, then CPU-mapped with Linux `mmap` and DMA_BUF_IOCTL_SYNC START/END. The child writes deterministic data only in the write/full modes; the parent checks every 32-bit word from a GPU kernel after the child ACKs. Both XTXs passed 4 KiB and 4 MiB once each. These are **serial** proofs, not live GPU readiness, generation reuse or any bandwidth result. Never run a GPU mode without a fresh exclusive `gpu` lease and GPU-process preflight; the runner's flag does not acquire the lease.

## Gate 3 — new paired mapped-host GPU-pull vs DMA-BUF CPU-push speed harness

See [PERFORMANCE.md](PERFORMANCE.md) for required lease preflight, OPT=-O2 performance-build instructions, A/B controls, exact GPU validation, CSV-like outputs and optimization decision thresholds. The fixed mode is `--test dmabuf-ab --device N --iterations 30 --mode direct|staged --lease-confirmed`. It uses one persistent CPU-only child for both the existing mapped-host GPU-pull baseline and the DMA-BUF CPU BAR-push candidate, samples 4 KiB/64 KiB/256 KiB/1 MiB/4 MiB, alternates path order, and reports p50/p95/p99 plus exact GPU word verification. The `ab_analyze.py` companion checks logs. **Built/card-free tested; hardware speed results pending exclusive GPU lease.** No new permissions or worktrees needed.

### Bounded Gate 3 speed trials

The allowlisted `rad50-tester --test dmabuf-ab` accepts an optional `--bytes 4096|65536|262144|1048576|4194304` to benchmark just one payload size and avoid remote-shell timeout for full five-size runs. Other test modes reject `--bytes`. With an exclusive verified GPU lease, first run `--device 0 --iterations 10 --mode direct --bytes 4096 --lease-confirmed`, then device 1. If both pass full-word GPU checks, use 30+ iterations, larger sizes and staged method. Benchmarking needs an `OPT=-O2` rebuild when other GPU measurements are finished. Full method and analysis in [PERFORMANCE.md](PERFORMANCE.md). Parser self-check is **39/39**; completed synthetic serial GPU results are in [GATE3_RESULTS.md](GATE3_RESULTS.md). Strict `ab_analyze.py` accepts only completed success logs, while `--provisional` retains partial diagnostic rows without claiming a pass.

## Gate 3b — production FP16 combine A/B in one tester

`--test combine-ab` calls BigCherry's actual `combine_row_f16` on a separate CPU-only helper, directly into the GPU-local DMA-BUF mapping or mapped-host baseline, with an independent GPU bitwise verifier of normal, masked, fallback-sentinel and NaN rows. `--mode staged` uses a host staging plane plus a full memcpy. Row width is 2,560 FP16 elements (5,120 bytes); optional `--rows` choices are 1, 16, 64, 256, 768. Without the selector all five cases run. Example: `rad50-tester --test combine-ab --device 0 --iterations 100 --mode direct --rows 768 --lease-confirmed` **only under a separately held exclusive GPU lease**. The runner does not acquire or verify the lease. CPU helper source planes are deterministic synthetic BF16 profiles; real function, not real model traffic. See [GATE3_COMBINE_RESULTS.md](GATE3_COMBINE_RESULTS.md) and [PERFORMANCE.md](PERFORMANCE.md). Card-free self-check 47/47.

## Gate 4 experimental live canary — NOT cleared

The unified executable provides `--test dmabuf-live --device N --iterations 10..200 --rows 1|16|64|256|768 --lease-confirmed`, where the GPU is armed before the separate CPU-only helper writes the FP16 payload. Both XTXs passed 10 generations at 1/16/256 rows, but XTX0's first 768-row generation **failed twice**, including after reducing host-control polling to one GPU lane; the GPU did not observe the ready ticket. The fixed `dmabuf-live-fg` mode changes host registration to `hipExtHostRegisterUncached` and is compiled but **not hardware-tested**. Strict `live_analyze.py` checks complete runs; `--provisional` retains partial/failure evidence without claiming a pass. See [GATE4_LIVE_RESULTS.md](GATE4_LIVE_RESULTS.md). Hardware tests require a separately acquired exclusive GPU lease; `--lease-confirmed` is only an acknowledgement.

## Gate 4 live cross-process GPU-ready modes (2026-11-09)

The single approved executable now includes `--test dmabuf-live --device N --iterations 10..200 --rows 1|16|64|256|768 --lease-confirmed`. The GPU kernel is armed before the separate CPU helper performs a real combine into DMA-BUF GPU VRAM, synchronizes CPU writes, and publishes the generation. GPU checks every FP16 word and acknowledges before the parent reuses the slot. **Initial mapped-host control passes 1/16/256 rows ×10 on both XTXs, but fails at 768 rows on XTX0 twice. Gate 4 BLOCKED.**

A separately named fixed `--test dmabuf-live-fg` mode (same args) is compiled but **not hardware tested**. It uses hipExtHostRegisterUncached for the host flag and is an XTX candidate; the installed HIP header excludes Navi4X/R9700. No new executable permissions needed. On an idle host with fresh exclusive lease, run one-row XTX0 first, then 768 rows; stop at any failure. Use `live_analyze.py` for strict completed-run review or `live_analyze.py --provisional` for failed/partial logs. Full source, evidence and safety steps: `GATE4_LIVE_RESULTS.md`. CLI 60/60, Python 72/72 card-free.

## RAD50 per-slice GPU-live variants after GFX11 watchdog correction

The existing unified `rad50-tester` now has fixed `dmabuf-live-parallel` (CPU helper publishes ready itself after DMA_BUF_IOCTL_SYNC_END, separately launched GPU waiter + 256-block verifier + done) and `dmabuf-live-relay` (the parent publishes ready in `hipHostMallocCoherent` after the helper socket ACK, same GPU kernels). Both require bounded `--device`, `--iterations` (10..200), `--rows` (1/16/64/256/768) and `--lease-confirmed`; actual lease ownership and no foreign KFD process must be externally verified. Original `dmabuf-live` and `dmabuf-live-fg` also use `wall_clock64()` with a HIP-queried wall-clock rate instead of GFX11-broken `clock64()`. All tested modes pass 768 rows after this fix. For complete/compact log validation use `slice_analyze.py`; `--provisional` retains partial/failure evidence without claiming a pass. Benchmark slices and stop/go: `SLICE_OPTIMIZATION.md`. Existing mapped-host GPU-pull remains the production fallback.

## 2026-11-09 GFX11 live tests (latest)

Fixed `dmabuf-live-parallel` and `dmabuf-live-relay` now use a `wall_clock64()` deadline scaled by the HIP device wall clock rate; the old `clock64()` deadline caused false gfx11 timeouts. Both paths passed 100 generations of 768 rows on both XTXs with GPU pre-armed and all FP16 words checked. Use `--test dmabuf-live-parallel --device 0|1 --iterations 100 --rows 768 --lease-confirmed` only under your own exclusive GPU lease. Direct publication (~464 us p50) vs parent relay (~469–471 us p50) is **not** a paired inference comparison. Long runs print first three, every tenth and final sample plus full-run percentiles and failure/cleanup markers. Parse with `slice_analyze.py` for strict completion, or `--provisional` for failed/partial logs; details in `SLICE_OPTIMIZATION.md`.

## S4 verifier-grid experimental option (2026-11-09)

The existing allowlisted `rad50-tester` accepts `--verifier-blocks 1|32|64|128|256` **only** with `--test dmabuf-live-parallel` or `--test dmabuf-live-relay`. Default remains 256, and GPU waiter, independent helper, full GPU FP16 comparison, SYNC and ticket protocol are unchanged. For example, **only under an independently acquired exclusive lease and verified no-foreign-KFD preflight**:

`rad50-tester --test dmabuf-live-parallel --device 0 --iterations 30 --rows 768 --verifier-blocks 128 --lease-confirmed`

The strict `slice_analyze.py` rejects mismatched grid metadata, incomplete records and missing post-cleanup success; old 256-block log format is compatible. 22 complete S4 logs strictly passed; one locally interrupted capture is marked PARTIAL. Keep the 256-block baseline; see [S4_VERIFIER_SWEEP.md](S4_VERIFIER_SWEEP.md).

## S1 CPU-only combine host baseline

The existing rad50-tester has a **GPU-free** mode for the actual production combine_row_f16:

    rad50-tester --test combine-cpu --rows 768 --iterations 100 --helpers 2 --mode hot

Rows are 1/16/64/256/768, helpers 1/2/4, iterations 10..200 and mode hot/unique. No --device or --lease-confirmed; this benchmark allocates no GPU memory and never launches HIP. Each generation verifies both host-direct combine and combine into a host staging plane plus memcpy against independent scalar FP16 bits, then emits first-three/tenth/final samples and full-run quantiles. Use cpu_analyze.py for strict completion; --provisional preserves partial logs. It is NOT a GPU-local BAR or model inference timing. See [S1_CPU_COMBINE.md](S1_CPU_COMBINE.md) for CPU contention and validation caveats.
