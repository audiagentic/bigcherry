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
8. Patch 1225 is the production safety boundary preventing unsafe heterogeneous RCCL entry.
9. META is the currently proven-correct heterogeneous reduction path on the target Brutus topologies.
10. HI134's META work does not constitute an RCCL repair and must not be reopened as one.

These are prerequisite truths.

Do not spend a new campaign rediscovering them.

## Safety invariants

The following rules apply throughout this runbook:

```text
1. Production patch 1225 stays enabled.

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

Where ROCm object inspection tooling is available:

```bash
if command -v roc-obj-ls >/dev/null 2>&1; then
    roc-obj-ls "$RCCL_PREFIX/lib/librccl.so" \
      | tee "$OUT/rccl-code-objects.txt"
fi
```

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
8. Run a real llama.cpp/BigCherry integration qualification only in an isolated experimental source/build where patch 1225's production protection has not been weakened.

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

Use `bigcherry profile-campaign` or equivalent real-workload profiling for final integration evidence.

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
architectures. A CPU-direct heterogeneous pair (e.g. XTX+R9700, gfx1100 +
gfx1201, devices excluding physical device 3) is predicted by this
evidence to work and remains queued for live hardware confirmation. If
confirmed, Phase 2 may proceed for that narrower RCCL-viable topology set;
device 3 stays on META regardless. Patch 1225's fail-closed guard remains
required for any topology including device 3.

**Separate finding, not part of this closure**: patch 1225 was found to be
`state=untested` and excluded from every default build's patch-set during
this investigation -- the guard this closure depends on is not currently
shipping in any tested binary. Tracked as its own follow-up, not folded
into this Outcome B record.

Full evidence artifacts: `artifacts/rccl-heterogeneous/rq04-01/` on Brutus
(environment capture, both RCCL builds, rccl-tests builds, all case
JSON/stdout logs, extracted/verified AMDGPU kernel metadata).
