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
`bar_ipc_canary.hip`, `bar_transfer_ab.hip`,
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
"$T" --self-check             # 22 strict argument cases, no GPU

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
  fails 4100. **No cross-process CPU writes are verified.**
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
- `--list` returned the compiled modes, `--self-check` passed 22/22
  strict argument cases, and `--test combine-ref` passed seven FP16
  reference checks (2,560 elements); each exited 0 without any GPU.
  Selected hardware modes have since run under exclusive leases;
  see `PROGRESS.md` for their exact outcomes. Recheck lease/processes
  before any further hardware mode.
  No subprocess/alias/interpreter was used to evade policy.
