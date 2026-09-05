# RCCL heterogeneous viability, qualification, and replay runbook

## Purpose

This runbook owns the executable procedure for investigating whether RCCL can be made correct on heterogeneous AMD GPU architecture groups and, only after correctness is established, qualifying and tuning RCCL execution plans for BigCherry.

It is intentionally separate from ordinary HIP compute-kernel autotuning.

The governing sequence is:

```text
prove RCCL source-level viability
    ->
prove exact collective correctness
    ->
define portable topology/compatibility identity
    ->
qualify candidate plans in crash isolation
    ->
measure admissible candidates
    ->
promote verified winners
    ->
optionally integrate compact runtime RCCL replay/tuning
```

Performance tuning MUST NOT precede the source-level viability gate.

## Existing evidence that must not be re-litigated

HI85, HI84, HI88, HI18, and HI134 already establish the following facts for the tested Brutus/ROCm/RCCL stack:

1. Same-architecture dual RX 7900 XTX (`gfx1100 + gfx1100`) RCCL is a valid control topology.
2. Heterogeneous-architecture RCCL participant groups can hard-abort inside `ncclGroupEnd()`.
3. Production-sized tensor-split workloads reproduce failures that tiny synthetic cases can miss.
4. The failure was reproduced independently of BigCherry.
5. Device ordering did not establish a general remedy.
6. `Ring`/`Tree` crossed with `Simple`/`LL`/`LL128` was already tested on a heterogeneous pair; all six combinations failed.
7. Therefore algorithm/protocol tuning is not currently evidence of heterogeneous RCCL safety.
8. Patch 1225 records an earlier fail-closed guard design for unsafe
   heterogeneous RCCL entry; it is not a universal architecture prohibition,
   not proof of complete current coverage, and must not be assumed present in
   the tested/default binaries. HI138 localized the Brutus hazard to a
   physical device/path capability and demonstrated XTX+R9700 CPU-direct RCCL
   success on qualified paths.
9. META is the currently proven-correct heterogeneous reduction path on the target Brutus topologies.
10. HI134's META work does not constitute an RCCL repair and must not be reopened as one.

These are scoped prerequisites and immutable historical evidence for the exact
tested topology, device set, source/build, and runtime. Before relying on a
guard, verify the actual patch composition and the current shared-admission
implementation; do not treat patch 1225 alone as protection for every
`ncclCommInitAll()` entry point.

Do not spend a new campaign rediscovering them.

## Safety invariants

The following rules apply throughout this runbook:

```text
1. RCCL admission remains fail-closed by a shared, reusable predicate
   consulted by EVERY ncclCommInitAll() entry point in the tree --
   not just patch 1225's original call site. Patch 1225 as currently
   written (docs/planning/active/gpu-collectives/GP02.md, rewritten
   2026-09-02) is a temporary, over-conservative implementation of
   this invariant: its real predicate is raw GPU-architecture
   inequality, which would incorrectly reject the {0,2}/{1,2}
   XTX+R9700 topology this runbook has since confirmed safe (device 3
   specifically, not architecture mismatch in general, is the actual
   hazard -- see HI138 below). 1225 also only protects the ORIGINAL
   comm_init_nccl() call site -- it does nothing for any other
   ncclCommInitAll() a patch brings up independently (confirmed: both
   the retired 1243 and the current 0840_hybrid_allreduce_dispatch
   each do exactly this with zero admission check). GP02 owns
   replacing 1225's guard with the shared predicate described above;
   until GP02 lands, treat 1225 as insufficient by itself and do not
   assume its presence protects anything beyond its one call site.

2. No unqualified heterogeneous communicator may enter RCCL
   through the production path.

3. Crash-prone RCCL experiments run in isolated child processes.

4. A GPU fault, signal, timeout, or device-loss result ends that
   child process.

5. After a hard failure, re-run a known-good homogeneous control
   before trusting subsequent evidence.

6. Correctness is a hard eligibility gate.

7. Crash-freedom alone is not correctness.

8. Performance is evaluated only among correct/admissible candidates.

9. No machine-local GPU ordinal, PCI BDF, serial, UUID, hostname,
   or /dev/dri path may become persistent tuning identity.

10. Unknown/unqualified heterogeneous signatures do not fall through
    to stock RCCL selection while the topology is known unsafe.

11. META remains the safe production path until an RCCL repair has
    passed all required gates.
```

---

## Phase 0 — Freeze the environment

Define:

```bash
export BC=/path/to/bigcherry
export RCCL_SRC=/path/to/pinned/rccl/source
export RCCL_PREFIX=/path/to/debug/rccl/install
export RCCL_TESTS=/path/to/rccl-tests
export ROCM_PATH=/opt/rocm
export OUT="$BC/artifacts/rccl-heterogeneous/<run-id>"

mkdir -p "$OUT"/{cases,traces,source}
```

Record:


```bash
{
    echo "timestamp=$(date -Is)"

    echo "=== BigCherry ==="
    git -C "$BC" rev-parse HEAD
    git -C "$BC" status --short

    echo "=== RCCL ==="
    git -C "$RCCL_SRC" rev-parse HEAD
    git -C "$RCCL_SRC" status --short

    echo "=== ROCm ==="
    hipconfig --full

    echo "=== rocminfo ==="
    rocminfo

    echo "=== SMI ==="
    rocm-smi --showproductname --showbus
} > "$OUT/environment.txt" 2>&1
```

The evidence record must identify the exact RCCL source revision and exact resulting library.

---

# Phase 1 — Repair heterogeneous RCCL viability

## P1.1 Freeze the first target

Primary source-debug target:

```text
one gfx1100 rank
+
one gfx1201 rank
+
one exact production-sized AllReduce known to reproduce HI85
```

Use the smallest deterministic production-representative failing size available from existing HI85/HI18 evidence.

Do not begin with a broad size sweep.

Do not begin with D=3 or D=4.

A two-rank reproducer is preferred because it removes unnecessary topology complexity while preserving the architecture mismatch.

## P1.2 Locate the current RCCL build controls

Do not assume a stale CMake/install option name.

Inspect the pinned RCCL checkout:

```bash
git -C "$RCCL_SRC" grep -n \
  -e 'AMDGPU_TARGETS' \
  -e 'GPU_TARGETS' \
  -e 'CMAKE_HIP_ARCHITECTURES' \
  -- \
  CMakeLists.txt cmake install.sh 2>/dev/null \
  | tee "$OUT/rccl-target-controls.txt"
```

Build RCCL with debug/symbol information and explicit coverage for at least:

```text
gfx1100
gfx1201
```

Record the complete configure/build/install command in:

```text
$OUT/rccl-build-command.txt
```

Do not continue if the resulting RCCL library lacks required architecture coverage.

Where active ROCm object inspection tooling is available, prefer the
non-deprecated inspection path:

```bash
if command -v clang-offload-bundler >/dev/null 2>&1; then
    objcopy --only-section=.hip_fatbin \
      "$RCCL_PREFIX/lib/librccl.so" "$OUT/rccl.hip_fatbin"
    clang-offload-bundler --list --type=o \
      -input="$OUT/rccl.hip_fatbin" \
      | tee "$OUT/rccl-code-objects.txt"
fi
```

`roc-obj-ls` is deprecated/non-functional on the validated ROCm 7.2.4
installation, and `llvm-objdump --offloading` crashed on that bundle format.
If the inspection tool or input format differs on another stack, record the
tool/version and use an equivalent only when it produces the same architecture
coverage evidence and provenance fields.

Require evidence for both target architectures before classifying a later failure as a collective-dispatch problem.

## P1.3 Build RCCL Tests against the exact RCCL under investigation

Use the RCCL Tests source associated with the current ROCm/RCCL source tree.

RCCL Tests must link against the exact debug/custom RCCL build.

After building:

```bash

ldd "$RCCL_TESTS/build/all_reduce_perf" \
  | tee "$OUT/all-reduce-ldd.txt"
```

The loaded RCCL library must resolve to:

```text
$RCCL_PREFIX
```

Do not accept "build succeeded" as proof of correct runtime linkage.

## P1.4 Establish the homogeneous control

Identify device ordinals for the current machine:

```bash
rocminfo
rocm-smi --showproductname --showbus
```

Example environment only:

```bash
export XTX0=0
export XTX1=1
export R9700=2
```

Run one same-architecture control using an exact message size:

```bash
HIP_VISIBLE_DEVICES="$XTX0,$XTX1" \
HIP_ENABLE_DEFERRED_LOADING=0 \
NCCL_DEBUG=INFO \
RCCL_OVERRIDE_ALGO=Ring \
RCCL_OVERRIDE_PROTO=Simple \
"$RCCL_TESTS/build/all_reduce_perf" \
    -b <bytes> \
    -e <bytes> \
    -g 2 \
    -n 5 \
    -w 1 \
    -c 1 \
    -T 20 \
    -M 1 \
    -Z json \
    -x "$OUT/cases/homogeneous-control.json"
```

Require:

```text
process survives
correctness passes
reported algorithm/protocol match request
```

STOP if the homogeneous control fails.

## P1.5 Reproduce one heterogeneous failure

Run one representative heterogeneous case:

```bash
HIP_VISIBLE_DEVICES="$XTX0,$R9700" \
HIP_ENABLE_DEFERRED_LOADING=0 \
NCCL_DEBUG=INFO \
RCCL_OVERRIDE_ALGO=Ring \
RCCL_OVERRIDE_PROTO=Simple \
"$RCCL_TESTS/build/all_reduce_perf" \
    -b <known-failing-production-bytes> \
    -e <known-failing-production-bytes> \
    -g 2 \
    -n 1 \
    -w 0 \
    -c 1 \
    -T 20 \
    -M 1 \
    -Z json \
    -x "$OUT/cases/heterogeneous-baseline.json"
```

This is not another algorithm/protocol discovery sweep.

HI88 already established that all six Ring/Tree × Simple/LL/LL128 combinations failed in the tested heterogeneous configuration.

Use one representative forced combination for source debugging unless evidence from the source investigation specifically requires another.

## P1.6 Run each dangerous case in process isolation

Any wrapper added to BigCherry must execute one candidate per child process.

Required result classifications:

```text
pass
wrong_result
unsupported
init_failure

launch_failure
signal
timeout
gpu_fault
device_lost
harness_failure
```

A hard failure must not leave the process alive for the next candidate.

If a reusable BigCherry wrapper is required, add:

```text
tools/bigcherry/profiling/rccl_qualify.py
tools/tests/test_rccl_qualify.py
```

The wrapper is diagnostic tooling only.

It must not modify production RCCL selection.

Suggested result row:

```json
{
  "schema_version": 1,
  "case_id": "gfx1100_gfx1201__allreduce__f32__<bytes>__ring__simple",
  "diagnostic_visible_devices": [0, 2],
  "device_arches": ["gfx1100", "gfx1201"],
  "collective": "allreduce",
  "reduction_op": "sum",
  "dtype": "f32",
  "bytes": 0,
  "algorithm": "Ring",
  "protocol": "Simple",
  "requested_channels": null,
  "observed_algorithm": "Ring",
  "observed_protocol": "Simple",
  "observed_channels": null,
  "returncode": null,
  "classification": "gpu_fault",
  "correct": false,
  "elapsed_seconds": 0.0
}
```

`diagnostic_visible_devices` is evidence only and MUST NOT become persistent replay identity.

## P1.7 Locate the common RCCL source path

Use the pinned source rather than relying on assumed line numbers:

```bash
git -C "$RCCL_SRC" grep -n \
  -e 'ncclGroupEnd' \
  -e 'ncclAllReduce' \
  -e 'devFunc' \
  -e 'ncclDevKernel_Generic' \
  -e 'allReduce' \
  -e 'AllReduce' \
  | tee "$OUT/source/dispatch-search.txt"
```

Trace the real path from:

```text
host collective enqueue
    ->
communicator/topology/tuning decision
    ->
work descriptor construction
    ->
device-function/kernel identity
    ->
module/code-object/function resolution
    ->
collective kernel launch
    ->
protocol primitives
```

Do not assume the failure site reported by `ncclGroupEnd()` is necessarily the instruction that originally caused the asynchronous device error.

## P1.8 Instrument per-rank dispatch facts

For the smallest deterministic failing case, record enough information to compare the gfx1100 and gfx1201 ranks at the final host/device dispatch boundary.

Capture, using the exact field names available in the pinned RCCL source:

```text
rank
logical participant count
detected architecture
algorithm
protocol
channel count
work element / work descriptor type
collective function identity
device-function identity / devFunc equivalent
kernel symbol / kernel class

module/code-object identity if available
launch grid/block geometry
transport/path class
peer-access decision
first HIP API reporting an error
```

Temporary debugging output belongs in the RCCL source investigation branch, not BigCherry production telemetry.

## P1.9 Trace the reproducer

Profile only the smallest deterministic failing case.

Example shape:

```bash
rocprofv3 \
    --hip-trace \
    --kernel-trace \
    --rccl-trace \
    --output-format csv \
    -d "$OUT/traces/baseline" \
    -- \
    env \
      HIP_VISIBLE_DEVICES="$XTX0,$R9700" \
      HIP_ENABLE_DEFERRED_LOADING=0 \
      NCCL_DEBUG=INFO \
      RCCL_OVERRIDE_ALGO=Ring \
      RCCL_OVERRIDE_PROTO=Simple \
    "$RCCL_TESTS/build/all_reduce_perf" \
      -b <known-failing-production-bytes> \
      -e <known-failing-production-bytes> \
      -g 2 \
      -n 1 \
      -w 0 \
      -c 1 \
      -T 20 \
      -M 1
```

Record:

```text
last successful HIP API
last successfully dispatched RCCL kernel
rank/device associated with it
function/kernel requested on each rank
first asynchronous device error
API/synchronization call where the error becomes visible
```

## P1.10 Classify the fault before editing

Classify the failure into one or more of:

```text
A. Missing/wrong code-object coverage
B. Wrong per-rank kernel/function resolution
C. Heterogeneous communicator/topology construction defect
D. Transport/P2P incompatibility
E. Algorithm/protocol-specific kernel defect
F. Shared generic collective-kernel dispatch defect
G. Work-descriptor/kernel ABI mismatch
H. Synchronization/lifetime defect
I. Arithmetic/correctness defect
J. Other — requires explicit evidence
```

Do not create a source fix until the evidence identifies a concrete failing boundary.

## P1.11 Implement the smallest RCCL source repair

The repair must address the demonstrated source-level cause only.

Examples of admissible repair shapes, depending on evidence:

```text
per-rank architecture-correct function lookup
architecture-correct code-object selection
heterogeneous-safe generic-kernel selection
work-descriptor compatibility correction
transport capability correction
```

Do not:

```text
disable correctness checks
silence HIP errors
catch/ignore a device fault
force META inside RCCL
change BigCherry patch 1225 to permit unqualified production use
replace the collective algorithm without evidence
```

## P1.12 Phase-1 validation gate

After each candidate source repair:

1. Rebuild RCCL.

2. Re-run the homogeneous control.
3. Re-run the exact heterogeneous reproducer.
4. Run at least 20 fresh-process repetitions of the repaired heterogeneous case.
5. Require correctness on every run.
6. Run the relevant production-sized reduction shapes.
7. Re-run the original failing topology with RCCL Tests.
8. Run a real llama.cpp/BigCherry integration qualification only in an isolated experimental source/build where the current shared fail-closed admission safety predicate remains active. If a guard bypass is the subject of the experiment, isolate it from production paths and record that fact explicitly.

Phase 1 passes only when:

```text
same-architecture control remains correct
+
heterogeneous production-sized AllReduce survives repeatedly
+
every rank receives a numerically correct result
+
no HIP device fault occurs
+
the repaired behavior is explained by the source-level change
```

One passing tiny synthetic collective is insufficient.

---

# Phase 2 — Topology-aware qualification, tuning, and replay

Phase 2 MUST NOT start until Phase 1 passes.

## P2.1 Keep operation identity separate from topology identity

Define separate concepts.

### ReductionOperationSignatureV1

Contains semantic operation facts only:

```text
schema_version
collective = allreduce
reduction_op = sum
dtype
exact element_count
exact byte_count
slice/shape semantics required by BigCherry correctness matching
graph/capture semantics only if they alter valid execution
```

Do not insert HIP ordinals, PCI addresses, UUIDs, hostnames, or topology data.

### TopologyIdentityV1

Portable description of the communicating hardware graph:

```text
schema_version
participant_count

nodes:
    architecture/capability class

edges:
    peer-access capability
    transport/link/locality class required for collective behavior
```

Persistent topology identity must be independent of machine-local enumeration.

Equivalent physical/semantic topology observed under different HIP ordinals must produce the same topology identity.

A real semantic link/peer-access change must produce a different identity.

### PlacementIdentityV1

Add only if evidence proves logical-rank placement changes correctness or cost.

If required, represent:

```text
logical rank
    ->
canonical topology role
```

Do not bind persistent placement identity directly to HIP ordinal or PCI BDF.

Symmetric devices should collapse when an automorphism proves them equivalent.

### RCCLCompatibilityRevision

Bind every qualified result to the RCCL implementation that was actually tested.

At minimum include enough data to distinguish incompatible:

```text
RCCL source revision
RCCL ABI/API compatibility revision
build configuration relevant to device kernels
code-object/architecture coverage

BigCherry collective-contract version where applicable
```

## P2.2 Candidate schema

Initial candidate:

```text
algorithm
protocol
channels
chunk_size
```

Only include a field when the exact pinned RCCL version exposes a real, controllable, observable mechanism for it.

Do not invent chunk/channel controls from another RCCL/NCCL version.

The current RCCL direct overrides:

```text
RCCL_OVERRIDE_ALGO
RCCL_OVERRIDE_PROTO
```

may be used for qualification when supported by the pinned build.

RCCL Tests `-M 1` must be used to verify what algorithm/protocol/channel configuration actually executed.

## P2.3 Qualification key

An RCCL candidate is qualified against an exact scope:

```text
RCCLCompatibilityRevision
+
ReductionOperationSignatureV1
+
TopologyIdentityV1
+
PlacementIdentityV1 if required
+
candidate
```

No fuzzy communicator matching.

No global conclusion such as:

```text
"Ring is unsafe everywhere"
```

may be produced from one topology failure.

## P2.4 Crash-isolated qualification

Run each candidate in a separate child process.

Required classifications:

```text
pass
wrong_result
unsupported
init_failure
launch_failure
signal
timeout
gpu_fault
device_lost
```

Compatibility/admissibility is separate from performance cost.

Unsafe candidates are ineligible regardless of speed.

## P2.5 Search order

Use staged search:

```text
1. protocol safety/admissibility
2. algorithm safety/admissibility
3. channels, if controllable
4. chunk sizing, if controllable
5. performance comparison among surviving candidates
```

Do not explode the entire Cartesian product before basic safety gates are known.

## P2.6 Correctness

Reuse the existing BigCherry reduction correctness contract and `test-hip-reduce` methodology where applicable.

Correctness must be checked against an independent reference and must retain:

```text
input identity
provider provenance

exact reduction signature
cross-device agreement
analytical F32 error bounds where that contract applies
```

A process exiting zero is insufficient.

## P2.7 Performance measurement

Performance comparison is allowed only after qualification.

Use:

```text
exact production reduction sizes/shapes
multiple repetitions
stable environment
same RCCL compatibility revision
same topology identity
same placement identity where relevant
```

Do not infer production benefit solely from isolated microbenchmark latency if the collective affects full inference scheduling.

Use `bigcherry profile-campaign` or an equivalent real-workload profiler for
diagnostics. At an acceptance boundary, an alternative is valid only if it
produces or imports the same canonical evidence and provenance fields.

## P2.8 Promotion

A promoted RCCL collective winner requires:

```text
exact identity match
qualified compatibility
correctness evidence
crash-free evidence
performance evidence
winner verification
```

No candidate is promoted merely because it was the fastest measured row.

The acceptance evidence must retain, at minimum: BigCherry commit and patch
composition; RCCL source SHA and build options; ROCm/tool versions; GPU
architecture and topology facts; algorithm/protocol; exact command and
environment; collective and message size; run count and order; correctness;
crash/timeout/device-loss outcome; and artifact references.

## P2.9 Runtime safety behavior

Until a heterogeneous RCCL topology has a valid exact qualification:

```text
known-bad or unqualified heterogeneous topology
+
no exact verified RCCL winner
    ->
DO NOT enter RCCL
    ->
retain current safe META/default path
```

This supersedes the earlier idea that an unknown heterogeneous collective could simply fall back to RCCL's stock internal policy.

Stock RCCL fallback may only be reconsidered after Phase 1 demonstrates that the RCCL implementation itself is safe for the relevant communicator class.

## P2.10 Tuner/plugin integration

Do not implement a BigCherry RCCL tuner plugin merely because the architecture allows one.

Before implementation:

```text
1. inspect the exact pinned RCCL source for tuner API support
2. record exact ABI/version
3. prove the API can control the required candidate dimensions
4. show a measured benefit over RCCL's own selection
5. prove the plugin cannot bypass BigCherry's admissibility contract
```

The earlier absence of tuner headers from a packaged runtime is not by itself proof that the source tree lacks an API.

Likewise, source-level API existence is not proof that the pinned production build exposes or supports it.

## P2.11 Runtime representation

Production replay should remain compact:

```text
resolved exact collective key
    ->
verified compact binding
```

Rich topology/qualification evidence belongs offline.

Runtime should not reconstruct or search a large tuning campaign.

Principle:

```text
resolve once
bind once
launch many
```

The runtime fast path may accelerate lookup but MUST preserve the exact equivalence relation used during qualification.


---

# Required artifact layout

Each source/qualification campaign should preserve:

```text
artifacts/rccl-heterogeneous/<run-id>/
├── environment.txt
├── rccl-build-command.txt
├── rccl-target-controls.txt
├── rccl-code-objects.txt
├── all-reduce-ldd.txt
├── cases.jsonl
├── cases/
│   ├── ...
├── source/
│   ├── dispatch-search.txt
│   ├── instrumentation-notes.md
│   └── fault-classification.md
└── traces/
    └── ...
```

The result set must be sufficient for another engineer to determine:

```text
what source was tested
what hardware/topology was tested
what exact collective was run
what candidate was requested
what candidate actually executed
whether it crashed
whether it was correct
where the source-level fault occurred
what source change repaired it
```

---

# Decision tree

```text
homogeneous RCCL control FAILS
    ->
environment/build/runtime invalid
    ->
STOP

heterogeneous RCCL FAILS before any useful collective kernel dispatch
    ->
investigate build/code-object/function resolution

heterogeneous RCCL reaches collective kernel but faults
    ->
investigate shared kernel/work descriptor/transport path

one source repair makes production-sized hetero AllReduce
correct and repeatable
    ->
PHASE 1 PASS
    ->
begin Phase 2 qualification

some qualified algo/protocol plans pass and others fail
    ->
retain exact scoped admissibility
    ->
tune only survivors

all plans are correct but RCCL default is already fastest
    ->
no BigCherry tuner required

BigCherry-selected plan produces repeatable material improvement
    ->
consider compact replay/tuner integration

no exact qualified heterogeneous winner
    ->
META/current safe path
```

---

# Relationship to existing BigCherry plans

```text
HI58
    SPLIT_REDUCE telemetry foundation

HI18
    reduction signature/correctness/provider-selection foundation

HI84
    heterogeneous D=3/D=4 correctness/topology campaign

HI85
    heterogeneous RCCL crash evidence + patch-1225 safety boundary


HI88
    same-architecture RCCL algorithm/protocol tuning evidence

HI134
    heterogeneous META profiling/tuning

HI86
    broader execution-planner architecture; not owned by this runbook

HI132
    generic profile-campaign infrastructure; not owned by this runbook
```

The source-level heterogeneous RCCL repair plan owns execution of this runbook.

---

# Closure criteria

This runbook is considered successfully exercised only when either:

### Outcome A — RCCL heterogeneous viability established

```text
a source-level cause is demonstrated
+
a bounded source repair exists
+
production-sized heterogeneous AllReduce is correct and repeatable
+
same-architecture behavior remains correct
+
qualification identity is recorded
```

after which Phase 2 may proceed;

or:

### Outcome B — RCCL heterogeneous viability rejected with source evidence

```text
the failure has been localized sufficiently to show that
repair is infeasible/unacceptable for BigCherry's supported scope
```

and the repository records the reason for retaining META-only heterogeneous execution.

A failure to find a tuning override is not by itself Outcome B; HI88 already established that fact.

---

## CLOSED: Outcome B (2026-08-29, HI138)

Phase 1 executed against RCCL rebuilt from source (ROCm/rccl commit
`57e58688f44c77076ad536ef1f6b68741fc6e694`, reports as RCCL 2.28.3),
explicit `--amdgpu_targets "gfx1100;gfx1201;gfx1030"`, code-object
coverage verified for all three real architectures via
`clang-offload-bundler --list` on the extracted `.hip_fatbin` section
(`roc-obj-ls` is non-functional/deprecated and `llvm-objdump --offloading`
crashes on this bundle format on this ROCm 7.2.4 install).

**Root cause, localized via `AMD_LOG_LEVEL=4` verbose ROCclr tracing on the
exact failing reproducer**: RCCL's generic device kernel
(`ncclDevKernel_Generic_1`/`_2`/`_4` -- the arch-independent, unroll-factor
-keyed fat-binary entry points `enqueue.cc`'s `ncclGetKernelIndex`/
`ncclKerns[]` resolve to) declares a `hidden_hostcall_buffer` hidden kernel
argument. Hostcall requires PCIe atomics support from the target device.
This hardware has none anywhere (consistent with HI84's separate finding:
no PCIe P2P bridge for ANY GPU pair on this box, homogeneous or
heterogeneous). ROCclr's AQL dispatcher correctly refuses the kernel
submission:

```text
ShaderName : ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)
Pcie atomics not enabled, hostcall not supported
AQL dispatch failed!
hipExtLaunchKernel: Returned hipErrorIllegalState
```

-- a real, working-as-designed runtime capability check, not a ROCclr bug.
This is the same failure class HI85 originally observed, now localized to
its exact mechanism.

**Repair attempted**: rebuilt RCCL with `-DCOLLTRACE=OFF` +
`CMAKE_BUILD_TYPE=Release` (NDEBUG reaching device compilation), on the
hypothesis that dormant device-side `assert()`/COLLTRACE diagnostic code
was forcing the hostcall metadata even though a plain Ring/Simple AllReduce
never calls into it. **Result: negative.** Identical failure persists.
Structurally verified (proper parse of the real `llvm-readobj --notes`
AMDGPU_METADATA YAML, not a textual/offset heuristic) that all three
generic kernel variants still declare `hidden_hostcall_buffer` on this
exact rebuilt image.

**Boundary**: removing the requirement would need either (a) RCCL
kernel-generation/device-link changes making hostcall declaration
conditional on actual runtime use, or (b) post-link code-object metadata
surgery. Both are outside this runbook's P1.11 admissible-repair scope
(no kernel-generation redesign, no metadata manipulation).

**Scope correction (2026-08-29, same day, after further evidence)**: the
above cause and repair-failure evidence are real, but the closure below was
initially over-generalized to "heterogeneous RCCL rejected" broadly. Direct
PCIe capability inspection (`lspci -vvv` AtomicOpsCap/Ctl) and the AMDGPU
kernel driver's own boot-time self-test (`dmesg`/`journalctl -k`: `amdgpu
0000:17:00.0: PCIE atomic ops is not supported`) show the missing PCIe
AtomicOps completion capability is a property of **one specific device's
PCIe path** -- physical device 3 (RX 6900XT), whose upstream root port is
the chipset-routed slot (PCIe 3.0 x4 via the PCH) -- not a property of
heterogeneous-architecture communicators in general. The other three GPUs
(2x RX 7900 XTX + R9700, all on CPU-direct root ports) all pass the same
boot-time test cleanly. External prior art (ROCm GitHub issues #2429,
#6074, #6520) documents the identical failure signature on other boards
with the same CPU-direct-vs-chipset-slot split, confirming this is a
per-PCIe-path property, not an RDNA-generation or architecture-mixing
limitation.

**Disposition (revised)**: Outcome B applies specifically to any RCCL
communicator that includes physical device 3 (RX 6900XT) -- its upstream
PCH root port lacks PCIe AtomicOps completion capability, a real hardware
limitation not fixable via kernel parameters, BIOS settings, or `setpci`
(forcing it has been reported elsewhere to hang the system). It does
**not** establish that RCCL is broadly unusable across mismatched
architectures. A shared fail-closed guard remains required for any topology
including device 3; patch 1225 is neither proof that the guard is currently
shipping nor sufficient protection for every entry point.

**CONFIRMED (2026-08-29, same day)**: live hardware test, XTX+R9700
(devices 0,2 -- gfx1100 + gfx1201, both CPU-direct root ports), Ring/Simple
512KiB: **PASS**, exit 0, zero correctness errors, correct algorithm/
protocol reported. 20/20 fresh-process repetitions passed with zero
failures (P1.12 gate satisfied). The mandatory homogeneous XTX/XTX control
also re-confirmed passing with this build. RCCL is viable for any topology
on this hardware excluding physical device 3: {0,1}, {0,2}, {1,2}, and
{0,1,2}. Device 3 (6900XT) requires META for any group it participates in.
**Phase 2 may now proceed for the RCCL-viable subset** -- see HI138's plan
item for the recommended next scope (a fresh Phase 2 sub-item, not
continued growth of this closed Phase 1 investigation).

**Separate finding, not part of this closure**: patch 1225 was found to be
`state=untested` and excluded from every default build's patch-set during
this investigation -- the guard this closure depends on is not currently
shipping in any tested binary. Tracked as its own follow-up, not folded
into this Outcome B record.

Full evidence artifacts: `artifacts/rccl-heterogeneous/rq04-01/` on Brutus
(environment capture, both RCCL builds, rccl-tests builds, all case
JSON/stdout logs, extracted/verified AMDGPU kernel metadata).

## Addendum (2026-09-02): GP03 size-adaptive dispatch reconfirms the boundary at the production integration layer

The above closure was established with RCCL Tests (`all_reduce_perf`) directly.
This addendum reconfirms the identical device-3 boundary one layer up, inside
BigCherry's actual production reduction dispatch path (`GP03`,
`docs/planning/active/gpu-collectives/GP03.md`) -- i.e. real `llama-bench`
`-sm tensor` runs through `ggml_backend_cuda_comm_*`, not the RCCL Tests
harness. No new source-level finding; this is an independent confirmation at
a different call site plus one new negative result specific to GP03's own
code path.

**Confirmed safe (RCCL-viable subset, no device 3):**

| topology | pairing | result | pp512 t/s | tg32 t/s |
|---|---|---|---|---|
| `{0,1}` | XTX + XTX (homogeneous, gfx1100+gfx1100) | clean | 1502.49 | 37.23 (n=128 run) |
| `{0,2}` | XTX + R9700 (gfx1100+gfx1201) | clean | 1205.82 | 29.58 |
| `{1,2}` | XTX + R9700 (gfx1100+gfx1201) | clean | 1283.78 | 30.46 |

All three ran GP03's real size-adaptive dispatch end-to-end: small (tg-sized,
~20KB) reductions stayed on the internal pinned-memory path, large (pp-sized,
~10MB) reductions correctly routed to a secondary `ncclCommInitAll()`
communicator and completed the real collective with no crash and no
correctness issue observed.

**New negative result -- device 3 crashes GP03's dispatch, not just raw
RCCL Tests:**

`{0,3}` (XTX + 6900XT) was run through the same GP03 binary
(`GGML_CUDA_ALLREDUCE=internal`, real `llama-bench -sm tensor` pp512).
GP03's secondary `ncclCommInitAll()` call is a **best-effort init with no
device-3 awareness** -- it does not consult patch 1225's guard or any
topology qualification list. On `{0,3}` it reports `rc=0` (success) because
`ncclCommInitAll` only builds communicator state and does not exercise the
PCIe-atomics hostcall path. The dispatch layer's own admissibility check
(`comms.size() == backends.size()`) therefore also reports "ready," and the
first real pp-sized reduction is routed to RCCL and hard-aborts inside
`ggml_backend_cuda_comm_allreduce_nccl` (`GGML_ASSERT`/`ggml_cuda_error`) --
same root cause as the original HI138 finding (device 3's chipset-routed PCH
root port lacks PCIe AtomicOps completion capability), reached through
GP03's own code path this time, with no fallback: the process aborts rather
than degrading to the internal/META path.

**Implication for GP03 (recorded as a hard blocker on GP03's own plan
item):** GP03 must not ship, even experimentally, without either (a)
consuming patch 1225 / GP01's qualified-topology list before ever attempting
its secondary `ncclCommInitAll()`, refusing that call outright whenever
device 3 is in the active device set, or (b) an equivalent topology check
inlined into GP03 itself. `ncclCommInitAll` returning success is init-time
evidence only -- Safety invariant 7 above ("crash-freedom alone is not
correctness") generalizes here to "communicator-init success alone is not
runtime admissibility": the same "looks fine at init, faults on first real
collective" pattern applies to any caller of RCCL on this hardware, not only
RCCL Tests.

## Addendum (2026-09-02): RCCL 2.30.4 regresses the confirmed-viable
{0,2}/{1,2} subset -- RCCLCompatibilityRevision must be pinned, not assumed

HI138's Phase 1 closure (and everything above) qualified against one exact
RCCL build: source revision 57e58688f44c77076ad536ef1f6b68741fc6e694,
reporting as RCCL 2.28.3/1.0.70204 depending on which packaged install is
queried (`vendor/rocm/7.2.4`). Per this runbook's own P2.1
`RCCLCompatibilityRevision` requirement, that qualification does not
automatically extend to a different RCCL build merely because both report
similar version numbers or are used on the same box.

Real hardware finding (GP06, `docs/planning/active/gpu-collectives/GP06.md`):
running `rccl_qualify.py`'s crash-isolated matrix ({0,1} homogeneous
control, {0,2}/{1,2} HI138-confirmed-viable heterogeneous, {0,3} device-3
negative control x Ring/Tree x Simple/LL/LL128) against RCCL 2.30.4 --
tested via TWO independently-built ROCm installs sharing that version
(`vendor/rocm/7.14`, a TheRock dev build, and a freshly-installed
`vendor/rocm/10.0`, ROCm 10.0.0 stable) -- reproduces the identical
regression on both: {0,1} passes, but {0,2}/{1,2}/{0,3} ALL fail
(`gpu_fault`), not just the device-3 case this runbook already explains.
Re-running the SAME matrix against RCCL 1.0.70204 (`vendor/rocm/7.2.4`)
reproduces HI138's original disposition exactly: {0,1}/{0,2}/{1,2} PASS,
only {0,3} fails.

This is evidence of a real RCCL-version regression between 1.0.70204 and
2.30.4 for the {0,2}/{1,2} topology class specifically -- not a build
artifact of one install, since two independently-built 2.30.4s agree. The
underlying mechanism has not yet been source-localized (GP06's step 2,
open) the way HI138 localized the device-3/hostcall failure; only the
existence and reproducibility of the regression is established so far.

**Runbook-level correction**: any qualification result recorded under this
runbook's Phase 1/Phase 2 procedures MUST bind to the exact RCCL version
tested (source revision preferred; packaged version string as a fallback
identifier only when source revision isn't available), and MUST NOT be
treated as transferable to a different RCCL build without re-running the
qualification matrix against that build specifically -- this applies even
when the newer build reports the "same" version number as another build
that was actually tested (2.30.4 was seen from two source-independent
installs in this finding). GP01 (Phase 2 execution) is blocked pending
GP06 pinning which RCCL version bigcherry's default build should actually
ship for the RCCL-viable subset.

Full evidence: `artifacts/rccl-heterogeneous/gp06-version-comparison/` on
Brutus (environment.txt + `rccl-qualify-{7.2.4,7.14,10.0}.jsonl` +
per-case stdout/stderr under `cases/rccl-qualify-{7.2.4,7.14,10.0}/` --
preserved from `/tmp` 2026-09-02, see GP06's own notes).

---

# Tooling: where the qualification tools live and how to run them

This section describes tools that exist and are checked in today. It is
kept separate from the procedural phases above (which describe the
governing method, independent of any one tool's current CLI surface).

## `tools/bigcherry/profiling/rccl_qualify.py`

One crash-isolated RCCL Tests case, run in its own subprocess so a GPU
fault or hard abort cannot take down a sibling case or this process.
Diagnostic tooling only -- never touches `GGML_HIP_REDUCE_PLAN`, patch
1225, or any other production selection state. Exposes `RcclTopology`,
`RcclCase`, `RcclCaseResult`, and `run_case()` / `append_result()` as a
library API; the 10-state classification (`pass` / `wrong_result` /
`unsupported` / `init_failure` / `launch_failure` / `gpu_fault` /
`device_lost` / `signal` / `timeout` / `harness_failure`) matches this
runbook's P1.6/P2.4 required classifications exactly.

**Current known gap (tracked as GP07,
`docs/planning/active/gpu-collectives/GP07.md`):** case/result identity
does not yet carry an `RCCLCompatibilityRevision` (see P2.1 above), and
repeated-rep artifacts are not yet uniquely named -- a 20-rep qualification
run currently overwrites its own per-attempt stdout/stderr/rccl.json. Do
not treat a `rccl_qualify.py` result as durable qualification evidence
across RCCL versions until GP07 lands; always record which exact RCCL
build produced a result by hand until then (see GP06's addendum above for
why this matters -- RCCL 2.30.4 silently regresses a topology 1.0.70204
qualifies cleanly).

## `tools/bigcherry/profiling/rccl_qualify_campaign.py`

Drives `rccl_qualify.run_case()` across a matrix of topologies x
algorithms x protocols. As checked in today:

```bash
python -m bigcherry.profiling.rccl_qualify_campaign \
    --binary /path/to/rccl-tests/build/all_reduce_perf \
    --output-dir artifacts/rccl-heterogeneous/<run-id>
```

Runs the six-cell Ring/Tree x Simple/LL/LL128 matrix at a fixed 512KiB
size against a hardcoded topology list (`{0,2}`, `{1,2}`, `{0,1,2}` --
XTX+R9700 pairs/triple only). Writes `cases.jsonl` plus per-case
stdout/stderr/rccl.json under `--output-dir`.

**Current known gap (also GP07):** no `{0,1}` positive control, no `{0,3}`
device-3 negative control, no repetitions, no RCCL-compatibility argument,
no post-fault control-recheck, and no way to vary element count or
topology from the CLI -- every qualification campaign broader than the
committed six-cell matrix (e.g. GP06's real 3-way RCCL-version comparison)
has so far been run through an uncommitted, ad-hoc variant of this script
rather than the checked-in one. GP07 exists specifically to close that gap
so future campaigns don't repeat it -- extend this file in place rather
than writing another one-off driver.

## Planned: validate vs. optimize diagnostics package (GP08)

Two separate tools, sharing one identity model, once GP07/GP08 land:

- **VALIDATE** (`tools/bigcherry/profiling/`, this section): crash-safety
  and correctness qualification for one exact (RCCL revision, topology,
  candidate) combination. Produces a durable PASS/FAIL artifact. This is
  the only place RCCL admissibility is ever decided.
- **OPTIMIZE** (planned: `tools/bigcherry/campaign/collective_benchmark.py`):
  real end-to-end performance comparison (pp/tg t/s, MTP completion
  throughput) across provider arms (`rccl` / `internal` / `hybrid` /
  `meta`) for a topology that already has a PASS qualification artifact.
  Refuses to run an RCCL-requiring arm against an unqualified topology --
  never silently falls back to a different provider. Reuses
  `tools/bigcherry/campaign/benchmark.py`'s existing paired-schedule
  statistics machinery rather than a new one-off runner; this is the
  durable form of the manual A/B/C methodology GP03 used informally to
  validate patch 1243's real hardware numbers.

Do not build a single tool with a `--mode validate|optimize` flag: the two
sides have different safety contracts (VALIDATE deliberately runs
crash-prone cases in isolation; OPTIMIZE must never do that), and
conflating them risks an optimization run silently exercising an
unqualified, potentially crash-prone topology.
