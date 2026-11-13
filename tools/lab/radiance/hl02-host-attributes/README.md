# HL02 — mapped host attributes and primary GPU read/write contention

**Owner:** HL02. This is a hardware-only laboratory experiment, not a production
receive method, benchmarking framework, or GPU-to-GPU transport implementation.
It reuses the existing `rad50-tester` for its CPU BAR-push control.

## Question

On a host where reads issued by the GPU on the first PEG half halve other
CPU-attached GPUs' host reads **and writes**:

1. Do HIP mapped-host allocations declared `coherent`, `noncoherent` or
   `writecombined` change the **contended** bandwidth, rather than just idle
   throughput?
2. Is the trigger a primary-GPU host **read** rather than a host write?
3. Does the already-verified RAD50 CPU-DMA-BUF push bypass the slowdown in a
   **serial 4 MiB synthetic** producer/consumer protocol while the primary GPU
   reads?

**Limit:** HIP host-allocation flags do not prove the outgoing PCIe No-Snoop
attribute. The first gate is evidence about these HIP allocation modes only.
This tester does not measure three concurrent CPU-to-GPU BAR streams, a host
reduction, a full 10 MiB all-reduce or inference latency.

## Input and build

- HIP/ROCm with gfx1100 and gfx1201 support.
- One isolated writable worktree and build output under the project work root.
- The existing, separately validated `rad50-tester` from RAD50 for the serial
  compare. The result records that executable's checksum.
- **Exclusive GPU lease**, acquired by the caller before every measurement;
  verify no foreign KFD processes. Never kill or signal foreign processes.

```sh
hipcc -O2 --offload-arch=gfx1100 --offload-arch=gfx1201 \
  tools/lab/radiance/hl02-host-attributes/host_attributes.hip \
  -o "${BC_WORK_ROOT}/builds/hl02-attributes/host_attributes"

# With separately held GPU lease:
# Per-device/transfer matched idle, primary-read and primary-write controls:
python3 tools/lab/radiance/hl02-host-attributes/run_matrix.py \
  --binary "${BC_WORK_ROOT}/builds/hl02-attributes/host_attributes" \
  --output "artifacts/HL02-matrix/xtx-read.jsonl" \
  --primary 0 --victims 1 --directions read --rounds 1 --seconds 0.6 --mib 64

# Serial 4 MiB CPU push vs GPU pull:
python3 tools/lab/radiance/hl02-host-attributes/run_push_control.py \
  --reader-binary "${BC_WORK_ROOT}/builds/hl02-attributes/host_attributes" \
  --rad50-binary "${BC_WORK_ROOT}/builds/rad42-bar-push/rad50-tester" \
  --output "artifacts/HL02-matrix/push-xtx.jsonl" \
  --primary 0 --victim 1 --iterations 10
```

CLI ordinals are from the **run's observed topology**, not model or engine
constants. `--reverse` reverses the order of host-memory attributes or
load-control arms, respectively. Three complete sessions with alternating
order were used for the 2026-11-13 evidence. Each primary-background process
is a direct child of the runner, which terminates **only its own process** and
captures its completed bandwidth.

The C++ probe initializes and verifies all GPU-readable words on the GPU
before/after each workload; the write path checks the final pattern. A
per-condition event timer separately counts device-time and wall-time GB/s.
Results are only eligible when every condition is a complete, zero-error
record. The serial RAD50 comparison has independent GPU full-word checks.

## Output, safety and disposition

Original JSONL files are generated in ignored, run-scoped `artifacts/`
folders. Normalized per-sample CSVs and the analysis are retained in
`docs/evidence/2026-11-13-hl02-transfer-matrix/`; they omit host absolute paths
and physical BDF identity, retaining topology roles and command parameters.

The default GPU pull remains the reference and fallback. No production engine,
tooling package or runtime layout is changed. The documented negative result
on HIP non-coherent flags does **not** authorize PCI configuration writes.

Next gates: actual PCIe transaction attributes if actionable, primary duty
cycle, sustained three-destination CPU pushes simultaneous with GPU partial
writes, 10 MiB GPU-live R9700 and XTX fault/lifetime correctness, and an exact
TP3 CPU sum evaluated against the improved VTP36 GPU candidate.
