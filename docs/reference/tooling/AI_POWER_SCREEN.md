# Independent AI power-profile screening (PWR12)

For the next independent coverage slice (PWR13), see [GPU qualification code-level implementation guide](../../design/gpu-qualification-code-guide.md). It documents source-level limitations and proposed fail-closed gates; it does not change the current screen or qualify a profile.

## Decision and boundary

This is **platform qualification tooling**, not a Radiance component or
serving-time gate. It can run with no model, engine build, serving process,
PyTorch, Windows application or downloaded ROCm package. The platform retains
source in the existing tools tree; build the standalone executable on demand with
the SDK that is actually installed. Nothing is added to the engine runtime
archive or engine CMake. No maintained fork of Sweet-Spot Tuner.

The original review is in [power tuning evidence](../power-tuning-evidence-2026-11.md).
The source reference is [Sweet-Spot Tuner v3.5.15](https://gitlab.com/spawnitems-group/sweetspot-tuner/-/tree/v3.5.15)
at a2a31ce63cc1d40af72997de621115eaa2dcd54d (MIT); the implementation here is
independent. It uses its idea of a small repeated FP16 GEMM and load gaps,
rather than the original app, dynamic ABI declarations or hardcoded card
settings. No copied third-party code or assets.

One canonical `engines/radiance/power-profiles.yaml` remains the named profile
intent. The platform reads it through `hardware.power_profiles`; no new
profile file format or duplicate resolver. PWR07/08 and the existing owner
transaction remain responsible for consent, hardware changes and recovery.
For an independent test PWR12 may call the existing platform PowerTransaction
used by gpu-power; it does not implement a second writer or modify the native
serving owner. PWR10 and PWR02 remain
responsible for qualification and performance/energy promotion. A screen is
a diagnostic result, never production authorization.

## Run independently

From a repository checkout, using the installed ROCm SDK and platform Python environment (with its existing PyYAML dependency):

```sh
test -n "${BC_WORK_ROOT:?set the project work root}"
test -n "${ROCM_PATH:?select the installed ROCm SDK}"
mkdir -p "$BC_WORK_ROOT/ai-power-screen"
nice -n 19 "$ROCM_PATH/bin/hipcc" -O2 -std=c++17 -Wall -Wextra \
    tools/bigcherry/hardware/ai_probe.cpp \
    -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lhipblaslt -lrocblas -lcrypto \
    -o "$BC_WORK_ROOT/ai-power-screen/ai-probe"
"$BC_WORK_ROOT/ai-power-screen/ai-probe" --self-test
```

The self-test is CPU-only. The native memory transform needs the actual target architectures in the hipcc build (add the corresponding --offload-arch flags when building on a different host). For real GPU work, use the verified, named GPU lease,
one selected physical PCI BDF and expected architecture. Do not run during
another workload. Use `PYTHONPATH=tools python -m bigcherry gpu-ai-screen --help`
for the platform command. The native executable is a low-level diagnostic:
use the wrapper for an operational screen, including temperature guard and
lease verification. No direct low-level invocation without a lease.

Stock reference example, **only after the operator verifies factory clock,
voltage and cap settings**. The tool does not reset the card. The actual cap
must equal its positive factory readback and any readable voltage offset must
be zero; factory clocks cannot be universally inferred, hence the explicit
operator assertion. Set the BDF/architecture/role, duration and safe hotspot
ceiling using actual device facts, not the example tuner's settings.

```sh
agent-lease acquire gpu --ttl 12m --est 5m --wait -- \
    env PYTHONPATH=tools python -m bigcherry gpu-ai-screen \
    --probe "$BC_WORK_ROOT/ai-power-screen/ai-probe" \
    --profiles engines/radiance/power-profiles.yaml \
    --profile stock --stock-confirmed \
    --bdf "$GPU_BDF" --arch "$GPU_ARCH" --role "$GPU_ROLE" \
    --dtype fp16 --n 768 --seconds "$SCREEN_SECONDS" --pause-ms 0 2 \
    --max-hotspot-c "$SAFE_HOTSPOT_C"
```

For a candidate, invoke the same platform command **inside the existing
authorized power transaction's lease and child-lifetime scope**, after the
owner has applied and verified the candidate. Change only `--profile`,
replace `--stock-confirmed` with `--reference <stock-run-id>`. A new managed run is allocated automatically. The operator/transaction must keep the lease through probe
exit and restore. Do not manually leave a candidate applied across leases.
For independent temporary application, add `--apply-profile --confirm-idle`
to the candidate command, only with explicit operator consent and no GPU users
or unresolved journals from **any** owner. The existing platform transaction
performs the range/writeability/original-state preflight, durable journal,
apply/readback and restoration. It uses the same recovery directory as
`gpu-power`. The native owner has per-session journals, so the operator must
also verify those have been resolved; the platform journal scan is not a
global native-session registry. No candidate is applied before matching the
required reference receipt, verifying current controls and checking thermal
limits. The selected BDF alone may be changed. Cancellation is cooperative
through reaping/restoration. Lost lease or unverified child exit refuses
restoration and retains the active journal. A failed restore or full readback
mismatch is nonpass, with the run ID/journal in the result; use existing
`gpu-power restore --run-id ID --processes-exited` under a verified lease after
checking all GPU users are gone. The existing controller does not support
SCLK offset application on RDNA4; that control explicitly refuses independent
apply, and can be screened observe-only under its authorized native owner.
No new controller, native journal schema, reset, or profile promotion. Empty placeholder
profiles (e.g. efficient/quiet/performance without rules) are refused, not
treated as successfully tested. The candidate must match each requested
setting; other observed controls and device identity must match the stock
reference. Unsupported settings and absent required readbacks refuse.

Use a single supported dtype initially. Set `--provider hipblaslt` (default)
or `--provider rocblas` explicitly. The latter supports the tested FP16,
BF16, FP32 and INT8 paths on gfx1030 and gfx1100; E4M3 FP8 uses hipBLASLt
where supported. No silent provider fallback or claim that a selected library
exercises every instruction. For current-settings diagnostics without a
factory assertion, use stock with `--observe-current` instead of
`--stock-confirmed`. That receipt is not a factory-stock reference. An
explicitly authorized **research-only** candidate may use it only with
`--observe-current --apply-profile --confirm-idle --reference <run-id>`;
its original controls and identity must exactly match the observed reference.
The ordinary factory-confirmed candidate path still refuses observed references.
Both paths remain diagnostic and cannot authorize production. Use a managed reference run ID. Previous standalone JSON is historical diagnostic evidence only; re-record stock through the managed command.

Include `memory` in `--dtype` with explicit even `--memory-mib` (total for two
buffers) and `--headroom-mib` (at least 256 MiB). The probe checks live free
VRAM before and after allocation, uses bounded host chunks, address- and
iteration-dependent patterns, H2D/D2D/D2H transfers and a GPU complement
transform, checking both complete buffers. The requested memory budget is
not a fraction automatically chosen from available VRAM. For example, a
small smoke may use `--dtype memory --memory-mib 128 --headroom-mib 512`;
larger resident-memory tests require an appropriate measured safe budget.
No multi-GPU/collective or simultaneous model-resident coverage is implied.

For an actual thermal-envelope screen select `--min-hotspot-c` and optional
`--warmup-seconds` per case, in addition to the required safe maximum.
The workload runs for warmup plus requested measured time. Native monotonic
timestamps and sampled temperatures must establish the entire requested
interval inside the actual workload, ending within 1 s of native completion,
with no sample gap above 1 s and every sample at or above the minimum.
Post-exit cooling samples cannot serve as in-load coverage; if it never reaches that interval, the case is inconclusive even
with correct results. The maximum always applies (including warmup/setup).
The memory sensor is sampled if available; its reported limit and an
optional `--max-memory-c` bound it. A requested missing sensor refuses.
Power sample mean is descriptive, not phase energy. No forced temperature
target, fan setting or cooling policy. A short warm-window check is not
a long hot soak.

Use a named GPU lease and an environment carrying that same instance;
an anonymous lease without AGENT_INSTANCE is correctly refused. Multiple requested types and burst
gaps run in order, stopping at the first nonpass; a supported subset never
silently passes an unsupported request. Choose `--seconds` per case and
a lease longer than total cases plus setup/cleanup. Evidence is not a serving launch input. The existing atomic power-receipt writer is reused. A supervisor may stop the screen at the
explicit safe ceiling or a lower available driver-reported limit. Missing
junction telemetry, failed identity, timeout, runtime error, numerical
mismatch, lease loss, or an interrupted run does not pass. This is a
polled guard, not a thermal control system. Hard driver hangs/SIGKILL cannot
guarantee reaping: verify child exit and retain the existing owner recovery
fence before any restore or new mutation.

## Managed evidence and retention

Every screen creates an existing WSL01 capture run at
`$BC_WORK_ROOT/runs/capture/<run-id>/` (subkind gpu-ai-screen, plan PWR13 when an instruction case, explicit matrix shape or mixed residency is requested, otherwise PWR12,
engine none). The usual `run.manifest.json` records lifecycle, source and
input digests, exit code, and a typed reference to the stock run for candidates.
The single `<run-id>.power.json` uses the existing platform Receipt v1
envelope: request, start/stop observations, transaction and ai_screen
observations (including per-case diagnostics). There is no new top-level
receipt schema or result store. The independent screen has no serving
manifest; that binding is explicitly unavailable. The transaction uses
the same run ID and existing gpu-power recovery directory, outside cleanup
of the run itself. Unverified restoration is held from retention.

Use `bigcherry run status <run-id>` and `bigcherry run harvest <run-id>`
under the usual workspace rules. Existing capture/v1 harvest keeps the
JSON evidence in evidence-hold; all screens are diagnostic-only, including
passes, not synthetic inference metrics or calibration records. Candidate
reference must be a completed managed run with passing coverage and
explicit factory confirmation (or explicitly selected observed research
reference as described above). Retain the typed run reference and digest
of its exact receipt. The old arbitrary --output argument is removed.
No engine runs or hardware settings are changed by this output alignment.

## Independent direct instruction cases (PWR13)

Use the same command, leased and observe-only, with
`--dtype alu wave lds --n 256 --seconds "$SCREEN_SECONDS" --pause-ms 0 2`
and explicit safe thermal ceilings. Here `--n` is the number of workgroups,
each with 256 lanes, rather than matrix size; provider is resolved to
hip-native and batch to 1. The matrix provider/batch options apply only to
matrix cases. No new writer, engine dependency, or qualification store.
New cases refuse --apply-profile pending cross-owner recovery hardening.

Each launch changes an epoch seed. Each lane produces four independently
CPU-computed uint32 words. ALU checks unsigned wrapping multiply-add,
multiply-high, rotate/xor and byte extraction; wave checks runtime-width
lane permutation, xor shuffle and ballot mask; LDS uses two cross-lane
exchanges separated by barriers, including a cross-wave read. The CPU
reference constructs lane maps independently, and checks every output
and both guard regions after each launch. Inputs are read back before and
after load. Recorded width (32 or 64), group geometry, exact positive
launch/lane/word/guard/input counts and allocation/headroom are mandatory.
CPU-side self-test covers wave32/wave64 boundary lanes and exact golden
ALU/LDS values. Host readback limits intensity: these are instruction
correctness tests, not proof of sustained GPU saturation or hot coverage.

Build all actual targets with explicit offload-arch flags. For a new
compiler/build, inspect each embedded code object rather than assuming
the HIP intrinsic name proves native coverage. With the SDK's llvm-objdump
in scratch, extract the binary's offload bundles using `--offloading` and
disassemble each target using `-d --demangle`. Within instruction_alu,
require multiply-high/low and bit extraction/rotate; within
instruction_wave, require the lane permutation operation and predicate
comparison; within instruction_lds, require explicit LDS read/write
and barrier instructions. This inspection is a build validation step,
not an automatic runtime ISA attestation. The managed receipt binds the
actual executable digest. See evidence below for the tested code objects.
The direct cases do not yet prove explicit WMMA/MFMA, masked partial-wave
semantics, packed dot or general vector-FP coverage.

## Explicit rectangular and padded matrices (PWR13)

The same command accepts `--matrix-shape M N K LDA LDB LDC` for
column-major NN GEMMs. Dimensions may be 1-2048, each leading dimension
between its logical row count and that count plus 256. Without the option,
all six values resolve to the existing square `--n`. With it, only matrix
cases use the explicit geometry; memory and direct instruction cases retain
their own size. No implicit fallback or provider change. Explicit geometry
cases are observe-only pending recovery hardening.

Separate BLAS layouts and actual strides reach the provider. The CPU
reference handles rectangular dense, sparse and diagonal inputs; padded
input bytes are distinct sentinels, padding in the output is pre-poisoned
and must stay untouched. The full stored output including padding is
checked before reuse by the device comparator (when batched) and at each
host audit; logical and padding comparison counts are recorded separately.
Exact geometry is part of the receipt and reference identity. The K bound
preserves exact FP16 output representability and the bounded stored span
times batch fits the device verification counter. Unsupported provider
shapes remain nonpasses. No transpose or nonzero-beta coverage yet.

## Mixed residency and compute (PWR13)

Select a matrix dtype and explicit `--resident-mib` with
`--headroom-mib` (at least 256 MiB). The allocation is additional to
the matrix input, output, oracle, guards, counters and workspace.
The live free-memory checks before and after allocation cover the
whole known budget. This mode is observe-only pending owner recovery
hardening, not an authorization to alter the power profile.

The resident buffer is initialized and fully read back before compute.
A separate nonblocking HIP stream changes one bounded 4 MiB chunk per
matrix batch using a deterministic epoch XOR, and the host verifies
the changed chunk after synchronization. A final full-buffer verification
is required; each chunk must be visited at least once during compute.
The last chunk may be partial. The requested duration is a minimum:
complete sweep coverage may extend execution. The existing timeout,
lease and thermal stop remain authoritative; incomplete coverage is
not a pass. Receipts validate exact bytes, chunk geometry, epochs,
full sweeps and words verified, not just allocation success.
The independently issued queues and live resident allocation are
tested; actual simultaneous execution, model residency, peak saturation
and peer traffic are not proven by this test.

## Numerical and identity contract

The native probe uses actual HIP-visible PCI identity and validates the
expected architecture before allocation. It selects a hipBLASLt heuristic or explicitly uses the installed rocBLAS
public default GEMM dispatch with atomics disabled. It records provider and
library versions; hipBLASLt also records opaque algorithm bytes, while
rocBLAS honestly records default dispatch, not an exact internal solution ID. It
checks every active HIP/library operation. Native protocol 7 / screen
observations schema 8 retains the three-corpus check and rotates three deterministic operand corpora: dense
signed, sparse signed, and signed-diagonal A times dense B. Floating
inputs are exactly representable 0 or +/-1/16, integer inputs 0 or +/-1.
The CPU integer dot sum is exact over the supported dimension range. The
outputs are FP16 for FP16, FP32 for BF16/FP32/E4M3 FP8, and INT32 for
INT8. Only floating signed zero is interchangeable. Each corpus owns
separate input, output and reference buffers. Outputs are poisoned before
every launch with values outside the valid output range, and guarded.
Default batch 1 reads every output on CPU; batch >1 checks every element
on device, then host-checks the last output and reference for every corpus
used in that batch, including output guards. Input and reference bytes are
also checked before and after load. A deliberate one-element error at the
first, middle and last element of each corpus must be detected at entry
and exit. Exact per-corpus launches and host checks, and total host/device
compared elements, are required in the receipt. All three corpora must
execute, plus the elapsed duration. Memory retains batch 1 and at least
two full passes. Positive pause is between batches. This is a stronger
correctness workload, not a comparable throughput benchmark for old probes.
Old schema-2 through schema-7 receipts are preserved as historical diagnostics; the
protocol handshake refuses old native binaries before any transaction,
and the new validator refuses an old reference.

The stock receipt and candidate must match physical BDF/architecture, role,
probe binary digest, kernel version, types, geometry, resident size, batch size, duration
and pause recipe. Compare actual HIP/driver/library versions, provider, dispatch identity and
workspace for each case. A rocBLAS default route does not prove identical internal
kernel selection. Every iteration is checked against the independent CPU-generated oracle,
using the selected checking mode. Verify observed controls before, between and after
cases; record the profile digest and reference receipt digest. A profile
label alone is not evidence. The output includes the observed junction
temperature range. This is a short independent arithmetic screen, **not**
proof of factory state, hot steady-state coverage, loaded library content
identity, all matrix instructions, memory residency, driver reset absence
or complete inference stability. A nonexact provider at the stock reference
is unqualified and must not be interpreted as undervolt instability.

Existing native kernel replay for IQ8R/IQR/attention/GDN/MoE and collective
transfers remains the separate production-role gate. Do not launch an engine
from this platform diagnostic or create an arbitrary subprocess protocol.
Use those existing harnesses under the appropriate production owner.
A nonexact floating corpus would require independently justified tolerance
and a frozen stock reference; it is not needed for the exact screen. Do not introduce another kernel backend
framework or make every profile test every model. Full hot conditioning
under normal cooling, memory temperatures, cold and hot transitions,
driver health, actual production replay/quality and synchronized phase
energy belong to the finalist validation, using existing PWR10/PWR02.
No forced cooling, factory reset, crash-edge search, automatic voltage change
or profile promotion by this tool. Explicit authorized candidate transactions
use only the existing controller.

## First implementation evidence (2026-11-26)

Source built successfully against installed ROCm 7.2.4; CPU oracle
self-test passed. The initial 27 targeted offline screen/profile/lease
tests passed. Short real GPU probes were serialized under individual GPU
leases; actual observed cap was factory and readable offset was zero before
testing, but clock factory state was not independently confirmed. This is
a no-write smoke, not a passing stock qualification receipt or stability soak.
All runs used N=768, one second of repeated math with a 2 ms pause.
No physical device locators or machine-specific paths are tracked.

| Architecture | Requested arithmetic | Result |
|---|---|---|
| gfx1201 | FP16 | 307 clean iterations |
| gfx1201 | BF16 → FP32 | 275 clean iterations |
| gfx1201 | E4M3 FP8 → FP32 | 290 clean iterations |
| gfx1201 | INT8 → INT32 | 301 clean iterations |
| gfx1201 | FP32 | 277 clean iterations |
| gfx1100 | FP16 | 248 clean iterations after signed-zero-only equivalence |
| gfx1100 | BF16 → FP32 | 93,503 nonzero element differences in first result; provider unqualified |
| gfx1030 | FP16 | SDK did not have TensileLibrary_lazy_gfx1030.dat; provider unavailable |

The first four gfx1201 passes predate the signed-zero equivalence refinement;
they already matched all raw output bytes. The original gfx1100 FP16 strict
comparison reported 8,246 elements differing in sign of zero; the refined
oracle verifies exact nonzero values. The gfx1100 BF16 differences included
observed 0x30000000 vs zero and 0x3d000003 vs 0x3d000000; the precise provider
cause is unresolved. Do not lower the oracle tolerance to make that case pass.
The gfx1030 missing provider returned a heuristic error, not a stability
verdict. No actual power controls or profile settings were changed. No temperature conditioning, near-full VRAM, stock-vs-candidate, energy
qualification or engine replay was performed in this first slice.


## Completion implementation and measured evidence (2026-11-26)

The first slice merged in #1627 (ca5f35ad9). The independent extension uses the
same platform owner and no engine changes. Actual ROCm 7.2.4 compilation
for gfx1030/gfx1100/gfx1201 and CPU oracle self-test passed. 72 targeted
offline tests passed, including actual existing PowerTransaction with the
fake sysfs/amdgpu driver (candidate readback, mismatch, verified restoration),
unreaped-child and restoration failure paths, lease-loss kill/reap, invalid
reference refusal, memory budget and thermal interval coverage.

Short named-lease observe-current CLI runs, N=768, 1 s, 2 ms gap, except
the indicated memory and hot-window rows. No operator factory assertion
or power mutation. Every passing CLI result reported unchanged before/after
identity and controls; every result was non-production-qualified.

| Arch | Provider / dtype | Clean iterations |
|---|---|---:|
| gfx1030 | rocBLAS FP16 | 288 |
| gfx1030 | rocBLAS BF16 -> FP32 | 264 |
| gfx1030 | rocBLAS FP32 | 238 |
| gfx1030 | rocBLAS INT8 -> INT32 | 270 |
| gfx1100 | rocBLAS BF16 -> FP32 | 312 |
| gfx1100 | rocBLAS FP32 | 264 |
| gfx1100 | rocBLAS INT8 -> INT32 | 267 |
| gfx1201 | hipBLASLt E4M3 FP8 -> FP32 | 332 |
| gfx1030 | HIP memory 128 MiB | 9 complete passes |
| gfx1100 | HIP memory 128 MiB | 14 complete passes |
| gfx1201 | HIP memory 128 MiB | 12 complete passes |
| gfx1100 | hipBLASLt FP16, 1 s conditioning + 2 s final, 40 C floor | 1,584 sustained / 738 burst |

Earlier gfx1100 FP16 and gfx1201 all five types passed as above. The
gfx1100 hipBLASLt BF16 mismatch and gfx1030 missing hipBLASLt library
are retained, not reclassified; rocBLAS supplies an independently checked
route without any additional SDK. FP8 is not claimed on gfx1030 or gfx1100.
Memory tests leave at least the specified 512 MiB headroom before and
after allocation. The XTX sampled 48 C during both short warm-window
cases; a 70 C floor with the same bounded recipe correctly returned
inconclusive despite 1,585 clean FP16 iterations. These prove the guard,
not hot operating-envelope stability. No new power profiles or defaults.

In the preceding implementation slice, actual clock/voltage/cap application was deliberately not performed.
The controller integration is tested on fake sysfs; real changed-profile
validation, representative extended hot/resident soak, driver-health and
production replay/KLD/phase-energy evidence remain PWR12's physical
qualification boundary with PWR10/PWR02. Never interpret the screen as
permission to alter hardware. The existing active plans were not edited.

## Authorized physical research screens (2026-11-26)

The operator authorized temporary hardware writes. All four cards reported
positive factory caps, zero readable voltage offset and auto performance
mode, but factory maximum clock state was not independently established.
The observed reference mode was selected explicitly; no factory assertion,
native qualification receipt, authored profile change, or production
promotion was made. The candidate profile input was a managed capture
`20261125T232552Z-pwr12-profile-input-53adfc83cd0e`, derived from the canonical
YAML and actual board-cap minima. The 2700 MHz clock-only and -75 mV research
arms were explicitly selected alongside the existing 2700/-125 candidate.
Each mutation had a separate named GPU lease, idle and unresolved-journal
checks covering both platform and native owners, and exact original readback
after reaping/restoration. The two owner journal schemas are distinct;
neither directory alone is a global fence. Do not use unattended cross-owner
application until the owner admission/recovery relationship is resolved.

All four observed references passed supported FP16/BF16/FP32/INT8 arithmetic
(and E4M3 FP8 on gfx1201), plus 512 MiB memory. One-second matched diagnostic
arms passed minimum caps (both gfx1100 and gfx1030: 272 W; gfx1201: 210 W),
and both gfx1100 cards passed 2700 MHz at 0/-75/-125 mV. All ten temporary
applications restored exactly. These are short screens, not hot qualification.

A 30 s FP16 N=2048 host-readback screen reached only 49 C and averaged
17.2 W: `20261125T232930Z-gpu-ai-screen-fc2e6257ed8b`. The batched
implementation was therefore added, and all supported dtype paths and
512 MiB memory passed again on all four physical cards at N=768, batch 32
(memory batch 1). The initial mixed screen missing the memory batch tag
and the earlier BF16 hot-window failure remain retained diagnostics; no
passing status was manufactured. ROCm 7.2.4 three-architecture native
build and CPU self-test passed, as did 78 targeted offline tests.

### Bounded hot research

N=768, batch 32, 10 s warmup plus 30 s measured, zero inter-batch pause.
BF16 uses rocBLAS on gfx1100/gfx1030; gfx1201 uses hipBLASLt FP32.
Requested measured junction floors were 60 C on gfx1100 and 50 C on
gfx1201/gfx1030, with junction and memory ceilings of 80 C. These are
bounded research conditions, not universal production thresholds.
Power is the arithmetic mean of polled board power, not integrated energy.
Iteration counts include warmup. Temperatures for passes are the verified
in-load hot window; inconclusive rows show maximum observed junction.
All candidate transactions restored exact original controls.

| Card/role | Arm | Result | Junction C | Clean GEMMs | Sample mean W | Managed run ID |
|---|---|---|---|---:|---:|---|
| gfx1100 helper0 | observed | pass | 70–72 | 292544 | 199.4 | 20261125T233549Z-gpu-ai-screen-7bfa40c4b2df |
| gfx1100 helper0 | 2700/0 | pass | 62–66 | 273216 | 156.9 | 20261125T233630Z-gpu-ai-screen-c0a4abf3e5fb |
| gfx1100 helper0 | 2700/-125 | inconclusive | max 60 | 280832 | 123.5 | 20261125T233711Z-gpu-ai-screen-1dab6288dc0d |
| gfx1100 helper1 | observed | pass | 74–77 | 300832 | 234.8 | 20261125T233753Z-gpu-ai-screen-c83b3e0390f6 |
| gfx1100 helper1 | 2700/0 | pass | 65–67 | 281632 | 174.9 | 20261125T233834Z-gpu-ai-screen-2294c3fbd862 |
| gfx1100 helper1 | 2700/-125 | inconclusive | max 59 | 282272 | 131.3 | 20261125T233915Z-gpu-ai-screen-dce4ef490e29 |
| gfx1201 engine | observed | pass | 65–78 | 150080 | 208.8 | 20261125T233957Z-gpu-ai-screen-abd015478395 |
| gfx1201 engine | 210 W cap | thermal stop | max 80 | — | 204.2 | 20261125T234038Z-gpu-ai-screen-4ba2c9004f0e |
| gfx1030 helper2 | observed | pass | 63–69 | 199232 | 217.6 | 20261125T234109Z-gpu-ai-screen-6f2194df315d |
| gfx1030 helper2 | 272 W cap | pass | 67–72 | 199008 | 221.1 | 20261125T234150Z-gpu-ai-screen-2985454b1aec |

The XTX clock-only arms reduced sampled board power by 21.3% and
25.5%, while clean iteration counts fell approximately 6.6% and 6.4%.
This synthetic workload is not serving throughput. The -125 mV arms
were numerically clean but did not meet the thermal coverage gate;
neither is qualified. The R9700 cap was stopped at the ceiling, not
retried at a higher limit. The 6900 XT cap passed but did not establish
power savings. Preserve all outcomes, not only passes.


### Large-residency research

Separate memory cases requested 20 s and at least two full checked passes,
with an explicit 2048 MiB reserve and budgets rounded down to 512 MiB from
observed total VRAM minus 3072 MiB. These are independent allocations,
not concurrent model residency or hot GEMM/VRAM coverage.

| Card/role | Arm | Allocation MiB | Complete passes | Result | Managed run ID |
|---|---|---:|---:|---|---|
| gfx1100 helper0 | observed | 20992 | 2 | pass | 20261125T234240Z-gpu-ai-screen-88fa0285175c |
| gfx1100 helper0 | 2700/0 | 20992 | 2 | pass | 20261125T234307Z-gpu-ai-screen-24a0f2f6d7ee |
| gfx1100 helper1 | observed | 20992 | 2 | pass | 20261125T234346Z-gpu-ai-screen-abca1fd9f379 |
| gfx1100 helper1 | 2700/0 | 20992 | 2 | pass | 20261125T234413Z-gpu-ai-screen-d570544c6a49 |
| gfx1201 engine | observed | 29184 | 2 | pass | 20261125T234444Z-gpu-ai-screen-bfcef0f1a2c2 |
| gfx1030 helper2 | observed | 12800 | 2 | pass | 20261125T234829Z-gpu-ai-screen-97b163609ed5 |
| gfx1030 helper2 | 272 W cap | 12800 | 2 | pass | 20261125T234916Z-gpu-ai-screen-64226fcb1beb |

The two XTX arms restored exactly and each allocation left about
3522 MiB available after allocation. The R9700 left about 3364 MiB;
the thermally stopped candidate was not repeated. The gfx1030 reference
and 272 W cap pair completed after Gate E, each leaving about 3532 MiB
free after allocation; the candidate restored exact original controls.
Sample mean board power was 48.36/48.44 W, respectively, showing no saving.

The managed capture/v1 harvest verified all 38 completed screen bundles
including nonpasses, plus the separately managed candidate input. Raw
receipts and verified evidence-hold copies retain native results,
reference links, original/restored controls and transaction journals;
the historic diagnostic capture remains retained. No production
qualification is implied. Factory verification, extended thermal and
burst coverage, actual production kernel/replay quality, driver health,
and measured phase energy/throughput remain outstanding under PWR12
and its existing PWR10/PWR02 ownership boundary.

## Extended observed-reference BF16 research (2026-11-26)

Both gfx1100 helpers ran rocBLAS BF16, N=768, batch 32, 30 s warmup and
60 s measurement per pattern (0/2 ms pause). The required in-load junction
floor remained 60 C and the junction/memory ceilings remained 80 C.

| Card | Sustained clean GEMMs | Verified sustained junction C | Burst clean GEMMs | Overall | Managed run ID |
|---|---:|---|---:|---|---|
| gfx1100 helper0 | 669120 | 72–75 | 418624 | inconclusive | 20261126T003601Z-gpu-ai-screen-f853a264fda9 |
| gfx1100 helper1 | 678784 | 75–78 | 416928 | inconclusive | 20261126T003902Z-gpu-ai-screen-1a5e1e347d28 |

Both sustained cases passed; burst cases were numerically clean but failed
the continuous hot-window requirement. No controls changed. The full receipts
and an initial pre-launch lease-verification refusal
(20261126T003525Z-gpu-ai-screen-835df6a9a21e) were harvested using capture/v1.
The refusal was not a GPU workload. A combined inconclusive reference cannot
authorize candidate application. Separate sustained-only research uses the
same thermal bounds; this does not replace the missing burst coverage or
factory/production qualification.

### Separate sustained-only matched clock comparison

Same provider, BF16, N=768, batch 32, 30 s warmup / 60 s final, 60 C
junction floor and 80 C junction/memory ceilings; 0 ms pause only.
The 2700 MHz candidate used zero voltage offset, not an undervolt.

| Card | Profile | Clean GEMMs | Mean sampled W | Verified junction C | Managed run ID |
|---|---|---:|---:|---|---|
| gfx1100 helper0 | observed | 667328 | 207.60 | 74–76 | 20261126T004204Z-gpu-ai-screen-6a482ae0c172 |
| gfx1100 helper0 | 2700 MHz / 0 mV | 635616 | 168.78 | 66–67 | 20261126T004508Z-gpu-ai-screen-ca59ff2a7abe |
| gfx1100 helper1 | observed | 679680 | 242.52 | 75–78 | 20261126T004335Z-gpu-ai-screen-4310f902d81b |
| gfx1100 helper1 | 2700 MHz / 0 mV | 635648 | 177.32 | 65–66 | 20261126T004640Z-gpu-ai-screen-acf831cdd992 |

All four passed; both candidate applications restored original controls
exactly. Mean sampled board power fell 18.7% and 26.9%, while completed
GEMMs fell 4.8% and 6.5%, respectively. These are synthetic fixed-duration
screen counts and sampled power, not phase-integrated energy or serving
throughput. All four receipts were harvested. Matched post-candidate observations follow.
Burst thermal coverage remains inconclusive.
No factory reference assertion or production qualification.

### Post-candidate matched observed references

| Card | Clean GEMMs | Mean sampled W | Verified junction C | Managed run ID |
|---|---:|---:|---|---|
| gfx1100 helper0 | 668864 | 207.40 | 73–75 | 20261126T004812Z-gpu-ai-screen-e7610fe49260 |
| gfx1100 helper1 | 679584 | 240.49 | 74–77 | 20261126T004944Z-gpu-ai-screen-553b88ebbb82 |

Both post-candidate observed references passed and were harvested. Pre/post
request, full start and stop controls and device identity matched for each
card; candidate original/restored controls matched exactly. Against the
midpoint of the two observed arms, mean sampled board power fell 18.7%
and 26.6%, with 4.9% and 6.5% fewer completed GEMMs, respectively.
The observed-candidate-observed sustained research is complete. This is not
factory-stock confirmation, a hot burst pass, production-role replay or
phase-integrated energy/throughput qualification. No production profile
was changed or promoted.

### Matched voltage sweep at 2700 MHz

Managed input `20261126T005904Z-pwr12-voltage-input-cfdc66286460` is
linked to the earlier experimental input and hash-verified. The sustained
recipe and thermal limits are unchanged. Each card advances in 25 mV
steps from -25 to -125 mV only after a passing previous step and exact
restoration; a nonpass stops that card. These are observed-reference
diagnostics; no factory confirmation or production qualification.

| Card | Offset mV | Clean GEMMs | Mean sampled W | Verified junction C | Result | Managed run ID |
|---|---:|---:|---:|---|---|---|
| gfx1100 helper0 | -25 | 621888 | 154.52 | 62–64 | pass | 20261126T010017Z-gpu-ai-screen-bb063285fb11 |
| gfx1100 helper1 | -25 | 635488 | 166.08 | 63–64 | pass | 20261126T010148Z-gpu-ai-screen-9a8a4492b230 |
| gfx1100 helper0 | -50 | 635680 | 145.91 | 61–63 | pass | 20261126T010333Z-gpu-ai-screen-7585383d5229 |
| gfx1100 helper1 | -50 | 636864 | 158.10 | 63–65 | pass | 20261126T010505Z-gpu-ai-screen-7ec9b6af2691 |
| gfx1100 helper0 | -75 | 636672 | 138.57 | 60–61 | pass | 20261126T010638Z-gpu-ai-screen-e993831118d0 |
| gfx1100 helper1 | -75 | 636704 | 150.35 | 63–64 | pass | 20261126T010810Z-gpu-ai-screen-8ae0e2f0beaf |
| gfx1100 helper0 | -100 | 636992 | 130.22 | max 60 | inconclusive | 20261126T010943Z-gpu-ai-screen-aaf17d1ffc5a |
| gfx1100 helper1 | -100 | 639072 | 142.92 | 61–62 | pass | 20261126T011115Z-gpu-ai-screen-f21a5f0cf7b6 |
| gfx1100 helper1 | -125 | 637376 | 134.34 | 61–62 | pass | 20261126T011813Z-gpu-ai-screen-bf39ddd2c3d1 |

Both restored exactly and both receipts were harvested. Relative to the
2700 MHz / 0 mV arms, sampled board power decreased 8.5% and 6.3%;
completed GEMM counts decreased 2.2% and 0.03%. The matrix oracle checks
actual numerical results, but not the installed Radiance kernel hierarchy.
Finalists require native correctness/replay and quality/energy evidence
under the existing owner. No production settings were changed.

The -50 mV arms also passed with exact restoration and verified harvests.
Against 2700 MHz / 0 mV, sampled watts fell 13.6% and 10.8%; completed
GEMM counts changed +0.01% and +0.19%, respectively. No serving claim.

The first -75 mV arm passed with exact restoration and verified harvest;
its junction window was close to the lower bound.

The completed sweep stopped helper0 at -100 mV: numerical output was clean,
but continuous hot coverage was missing. Helper1 passed through -125 mV.
All four final receipts were harvested and exact restoration verified.
No deeper helper0 arm was run. The highest passing sustained offsets in
this recipe are -75 and -125 mV, not production stability limits.

## PWR13 rotating-corpus implementation evidence (2026-11-26)

Native protocol 2 compiled with ROCm 7.2.4 for gfx1030/gfx1100/gfx1201;
CPU oracle self-test and protocol check passed. Hardware offline suite:
197 passed, one skipped. New refusal fixtures cover missing/invalid per-corpus
counts, old protocol/schema, handshake failure and binary change. Existing
fake power transaction cases pass; no writer or production defaults changed.

Observe-current, no writes, unchanged start/stop controls and identity in
all six managed captures below; separate named leases, unchanged 80 C
junction/memory ceiling. Matrix cases N=256, three seconds sustained unless
specified; no hot-floor or production claim.

| Role | Provider | Batch | FP16 | BF16 | FP32 | INT8 | FP8 | Managed run ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gfx1100 helper0 | rocBLAS | 1 | 12575 | 16130 | 15825 | 16681 | not run | 20261126T015751Z-gpu-ai-screen-82c5b9fae077 |
| gfx1100 helper1 | rocBLAS | 2 | 12156 | 15640 | 15486 | 16184 | not run | 20261126T015805Z-gpu-ai-screen-5cb973591371 |
| gfx1201 engine | hipBLASLt | 32 | 72512 | 62560 | 56832 | 65472 | 65504 | 20261126T015818Z-gpu-ai-screen-835094a5e951 |
| gfx1030 helper2 | rocBLAS | 32 | 81056 | 51776 | 58880 | 67104 | not run | 20261126T015836Z-gpu-ai-screen-6c7afecb7598 |

Counts are clean GEMMs. All required cases passed. On gfx1100 helper0,
N=768 BF16 batch 32, 10 s sustained and 10 s with 2 ms burst pause:
41536 and 30464 clean GEMMs; 24498929664 and 17968398336 device element
comparisons, 2296774656 and 1684537344 host element comparisons
(20261126T015614Z-gpu-ai-screen-512f35e587ee). On gfx1030 helper2, 128 MiB
memory completed 28 full passes (20261126T015921Z-gpu-ai-screen-4880e38ee7de).
All six receipts harvested. No production qualification, no voltage/clock
change. Exact instruction classes, mixed residency, varied shapes,
transitions/thermal soak, cross-owner transaction hardening and native
role/serving tests remain separate gates. No reused schema-2 reference.

### PWR13 direct instruction evidence (2026-11-26)

ROCm 7.2.4 built gfx1030/gfx1100/gfx1201 with explicit offload targets;
CPU golden self-test and 202 hardware tests passed (one skipped). The
initial volatile shared declaration lowered to flat accesses; the final
nonvolatile shared storage and full-group barriers emit the intended LDS
instructions. Extracted code objects were inspected for each target:

| Architecture | Integer | Wave | LDS |
| --- | --- | --- | --- |
| gfx1030 | v_mul_hi_u32, v_mul_lo_u32, v_bfe_u32, v_lshl_or_b32 | ds_bpermute_b32, predicate compare | ds_read_b32, ds_write_b32, s_barrier |
| gfx1100 | same integer instructions | ds_bpermute_b32, predicate compare | ds_load_b32, ds_store_b32, s_barrier |
| gfx1201 | same integer instructions | ds_bpermute_b32, predicate compare | ds_load_b32, ds_store_b32, s_barrier_signal/wait |

Separate named leases, observe-only, 256 groups of 256 lanes, actual
wave32 on all four cards, 5 s sustained and 5 s with 2 ms gap for each
case. All 24 cases passed. Entries are checked launches (sustained/burst),
not inference throughput.

| Role | ALU | Wave | LDS | Host-verified words | Managed run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| gfx1100 helper0 | 12658/1472 | 12443/1428 | 13227/1470 | 11193024512 | 20261126T030010Z-gpu-ai-screen-06054a43468d |
| gfx1100 helper1 | 12607/1458 | 12447/1443 | 13104/1513 | 11159994368 | 20261126T030043Z-gpu-ai-screen-ecde5445c70e |
| gfx1201 engine | 10569/1461 | 10392/1459 | 10767/1501 | 9476243456 | 20261126T030117Z-gpu-ai-screen-8df0f8658a21 |
| gfx1030 helper2 | 8520/1418 | 8376/1373 | 8638/1454 | 7806386176 | 20261126T030151Z-gpu-ai-screen-b69a4bec4e03 |

Total 39635648512 checked words, no mismatch; all four captures harvested
and unchanged start/stop controls and identities verified. These are
host-readback-limited smoke tests, not hot soak, saturation, matrix
instruction coverage, or production qualification. No power changes.

### PWR13 rectangular geometry evidence (2026-11-26)

Protocol 4/schema 5, ROCm 7.2.4, three explicit offload targets and CPU
golden self-test. 210 hardware tests passed (one skipped). An initial
serialization defect omitted matrix_geometry in the native result; four
runs were inconclusive, not numerical passes. Retained and harvested:
20261126T030657Z-gpu-ai-screen-f116c6b3b977,
20261126T030703Z-gpu-ai-screen-146c3fb219d5,
20261126T030709Z-gpu-ai-screen-7b7bfe41cfe3,
20261126T030714Z-gpu-ai-screen-fdb488b188c8.
Fixed the serializer and changed rejection handling to retain a parsed
native result and diagnostic rather than losing the case on coverage
failure; malformed JSON retains a bounded output tail.

Final matched observe-only repeats: shape A=(64,96,128,80,144,80),
shape B=(65,33,63,72,64,72), 4 s sustained per case, batch 32,
junction/memory ceiling 80 C. Entries are verified GEMM launches.

| Role | Shape | Provider | FP16 | BF16 | FP32 | INT8 | FP8 | Managed run ID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| gfx1100 helper0 | A | rocBLAS | 191200 | 160928 | 150688 | 197312 | not run | 20261126T030830Z-gpu-ai-screen-3bdb98246035 |
| gfx1100 helper1 | B | rocBLAS | 195040 | 185920 | 189472 | not run | not run | 20261126T030848Z-gpu-ai-screen-f2d4aa324f6f |
| gfx1201 engine | A | hipBLASLt | 183008 | 93280 | 76736 | 69664 | 90784 | 20261126T030903Z-gpu-ai-screen-aa8790a4cd8a |
| gfx1030 helper2 | B | rocBLAS | 149504 | 93696 | 131328 | not run | not run | 20261126T030929Z-gpu-ai-screen-e8f5ac430825 |

All 15 requested cases passed: 2158560 clean GEMMs, 2082375360
device-side padding and 195222690 host-side padding comparisons.
Eight captures harvested including the initial nonpasses. All final
start/stop identities and controls matched; no writes, no hot/production
claim. The deliberately different shapes are coverage samples rather than
a complete per-card Cartesian product.

### PWR13 mixed-residency evidence (2026-11-26)

Protocol 5/schema 6, ROCm 7.2.4, three target code objects and CPU
oracle self-test passed. Offline hardware suite: 213 passed, one skipped.
BF16 N=768 batch=32, sustained, minimum 15 s (129 MiB smoke 5 s),
512 MiB requested headroom and 80 C junction/memory ceilings.
All five observe-only cases passed with unchanged start/stop controls.
Complete initial/final memory readbacks and every compute-time changed
chunk are counted in the verified words below.

| Logical role | Resident MiB | GEMMs | Verified words | Compute sweeps | Run ID |
| --- | ---: | ---: | ---: | ---: | --- |
| helper0 smoke | 129 | 17120 | 616038400 | 16 | 20261126T031427Z-gpu-ai-screen-68c247443543 |
| helper0 | 20992 | 167936 | 16508780544 | 1 | 20261126T031527Z-gpu-ai-screen-99c339967e08 |
| helper1 | 20992 | 167936 | 16508780544 | 1 | 20261126T031643Z-gpu-ai-screen-e17f14d0d900 |
| engine | 29184 | 233472 | 22951231488 | 1 | 20261126T031824Z-gpu-ai-screen-462d2bd264f1 |
| helper2 | 12800 | 102400 | 10066329600 | 1 | 20261126T032004Z-gpu-ai-screen-4d63afab4334 |

Full-size workload durations were 50.086, 49.170, 58.256 and
46.403 seconds respectively; complete sweep coverage extended beyond
the minimum duration. The full cohort verified 671744 GEMMs and
66035122176 32-bit resident words, not unique bytes or overlap.
All five captures harvested. A job-monitor restart interrupted queued
launches for engine/helper2; no duplicate active processes or lease
were present before resubmission. No mutation, hot floor, peak
saturation, peer-transfer or serving proof is claimed.

### PWR13 vector, packed dot and native matrix instructions (2026-11-26)

Protocol 6/schema 7 adds vector-fp, dot4, dot2, dot2-pos, wmma-fp16,
wmma-bf16, wmma-fp16-pos, wmma-bf16-pos and wmma-iu8. They use the
same exact per-lane CPU verification path; WMMA returns eight words
per lane and executes a native 16x16x16 wave32 tile with nonzero C.
gfx11/gfx12 use their distinct fragments. IU8 uses unsigned A and
signed B across the byte range with int32 accumulation. gfx1030
refuses native WMMA; no scalar emulation is counted as coverage.
Vector FP uses exact dyadic arithmetic. Packed dot2 uses explicit
native asm on gfx1030 and native builtins elsewhere.

ROCm 7.2.4, all three offload targets, CPU self-test and 222 hardware
tests passed (one skipped). Disassembly confirms vector FMA/multiply,
gfx1030 v_dot4c_i32_i8 and v_dot2_f32_f16; gfx1100/gfx1201
v_dot4_i32_iu8 and native dot2 (gfx1100 dot2acc), and native
FP16/BF16/IU8 WMMA on gfx1100/gfx1201 only.

| Case | gfx1030 | gfx1100 (both physical cards) | gfx1201 |
| --- | --- | --- | --- |
| vector-fp / dot4 | pass | pass | pass |
| dot2, signed | pass | mismatch | pass |
| dot2-pos | pass | pass | pass |
| FP16/BF16 WMMA, signed | not applicable | mismatch | pass |
| FP16/BF16 WMMA, nonnegative | not applicable | pass | pass |
| mixed IU8 WMMA | not applicable | pass | pass |

Signed FP16 and BF16 WMMA on both gfx1100 cards produced the same
0x3f7ffffa instead of exactly 0x3f800000 (1.0) in a cancellation
case; signed FP16 dot2 also produced small repeatable discrepancies
on both cards. These are observed-baseline failures, consistent with
limited internal precision, not demonstrated undervolt faults. Do
not relax the exact oracle or substitute a nonnegative pass for signed
coverage. A future signed precision/repeatability contract must be
independently justified and retain both mathematical error and exact
matched reference output evidence. No stock/factory or production
qualification is asserted.

Managed captures (all harvested, unchanged controls):

| Scope | Role | Outcome | Run ID |
| --- | --- | --- | --- |
| Initial vector/dot4/WMMA | helper0 | mismatch | 20261126T033215Z-gpu-ai-screen-17dbbec9efb4 |
| Initial vector/dot4/WMMA | helper1 | mismatch | 20261126T033235Z-gpu-ai-screen-9859a20d434b |
| Initial vector/dot4/WMMA | engine | pass | 20261126T033255Z-gpu-ai-screen-aca24c0f3c8f |
| Initial vector/dot4 | helper2 | pass | 20261126T033331Z-gpu-ai-screen-0d164375b282 |
| Detailed FP16 WMMA diagnostic | helper0 | mismatch | 20261126T033421Z-gpu-ai-screen-a3668d9ef3a7 |
| Expanded vector/dot2 cohort | helper0 | mismatch | 20261126T033558Z-gpu-ai-screen-74792e2854d1 |
| Expanded vector/dot2 cohort | helper1 | mismatch | 20261126T033615Z-gpu-ai-screen-c886d38294d2 |
| Expanded vector/dot2/WMMA | engine | pass | 20261126T033630Z-gpu-ai-screen-9280586815e4 |
| Expanded vector/dot2 | helper2 | pass | 20261126T033718Z-gpu-ai-screen-33e238f18b56 |
| Nonnegative WMMA | helper0 | pass | 20261126T033740Z-gpu-ai-screen-b57f565609c7 |
| Signed BF16 WMMA | helper0 | mismatch | 20261126T033753Z-gpu-ai-screen-a8896c1b493c |
| Nonnegative WMMA | helper1 | pass | 20261126T033756Z-gpu-ai-screen-80c4457722fc |
| Signed BF16 WMMA | helper1 | mismatch | 20261126T033809Z-gpu-ai-screen-faa56b21994f |
| Nonnegative dot2 / mixed IU8 | helper0 | pass | 20261126T034002Z-gpu-ai-screen-e19a0cc59b80 |
| Nonnegative dot2 / mixed IU8 | helper1 | pass | 20261126T034016Z-gpu-ai-screen-6a6b5b4ea0b9 |
| Nonnegative dot2 / mixed IU8 | engine | pass | 20261126T034030Z-gpu-ai-screen-2bb703aef891 |
| Nonnegative dot2 | helper2 | pass | 20261126T034044Z-gpu-ai-screen-c6346bc58f9b |

70 passing cases and 7 mismatch cases; 28,578,021,376 host-verified
words in passing cases across the staged binaries. Evidence is bound
to each binary digest; a new binary cannot use an old reference.
All workload launches were short separate leased observe-only
diagnostics; sustained and 2 ms burst patterns, no hot floor.

### Managed workload recipes (PWR13)

Use `gpu-ai-qualify plan --recipe screen|extended --probe PROBE --profiles PROFILES
--bdf DEVICE --arch ARCH --role ROLE --headroom-mib MIB --max-hotspot-c C
--max-memory-c C`. The recipe is resolved into a verified managed capture.
Supply `--native-plan FILE` to retain the RCS27 planned routes and gaps;
this does not execute them. A non-stock profile can be planned but cannot run.

Under a GPU lease, use `gpu-ai-qualify run --plan ID --group NAME`.
Each group is a separate screen receipt. Harvest those receipts, then use
`gpu-ai-qualify report --plan ID --result RUN_ID` (repeat the result flag).
The report is itself a retained managed capture and reports missing groups,
nonpasses, and unsupported cases. It always sets production_qualified false.
No new power-control mechanism or scheduler is introduced.

### Diagnostic phase telemetry

Telemetry schema 1 adds timestamped raw interface readings and status,
memory temperature timestamps, bracketed/gap-bounded native-load energy
estimates, and requested-warmup interval estimates. These are estimates
from average-power readings, not meter-grade energy. An incomplete trace
has no joule value. Driver-health observation uses a private read-only
kernel log cursor; unavailable/overrun coverage is explicit and observed
AMDGPU fault classes stop the workload. The scope is system AMDGPU,
not an inferred physical-card fault attribution.

### Independent peer transfer scope

Build the existing GP11/GP13 peer-copy fixture with the installed SDK.
Under the GPU lease, invoke `gpu-ai-qualify peer --probe PROBE --bdf A B
--arch ARCH_A ARCH_B --headroom-mib MIB --max-hotspot-c C --max-memory-c C`.
The command retains a separate managed peer capture for the explicit pair.
Unavailable peer access is a nonpass, never a staged-copy pass. Both
endpoints must retain their original observed controls. This is separate
from native collective and production serving qualification.

### PWR13 scoped RDNA3 precision observations (2026-11-26)

The original signed dot2, FP16 WMMA and BF16 WMMA cases stay strict.
Recipe schema 2 explicitly maps only those three signed groups to
gfx1100-only reference cases. No other architecture, instruction,
positive corpus, integer or matrix-library case accepts an exception.
Native protocol 7 and observation schema 8 bind the new semantics;
the probe uses system libcrypto SHA-256 on the host.

Reference cases require observe-current, verified zero voltage offset,
unchanged controls and at most 64 workgroups. Eight deterministic
epochs use the same input/seed generator as the strict test. Every
output is checked against the independent exact CPU lattice; every
repeat must equal the entire first output byte-for-byte (floating zero
is canonicalized by the original instruction test). Record each
epoch's SHA-256, total differing words, and maximum absolute error.
The validator additionally requires a conservative independent
FP32 arithmetic budget: gamma(s) = s*u/(1-s*u), u=2^-23 (one full
relative binary32 ULP), s=9 and max sum-of-magnitudes=5 for the
four-product dot2 chain including C; s=33 and sum bound 79 for a
16-product tile plus C. This is a deliberately conservative
corpus-screening policy, not an AMD precision guarantee or a
standalone pass tolerance. Neither a rounded CPU match nor a
small absolute error alone qualifies the output.

Every native observation remains precision_observed (exit 2),
including successful repeated epochs. To allow the known precision
signature, run the same group under separate leases at least three
times under one managed plan. After review, supply the two prior
independent, harvested observations with repeated
`--precision-reference RUN_ID` options to `gpu-ai-qualify report`,
and the later observation with `--result RUN_ID`. The report
revalidates scope, timestamps, native counts, exact recipe/input and
device identity, unchanged observed snapshot, and equality of every
epoch fingerprint and maximum CPU error across references and result
(including sustained/burst equality). A different output, missing
reference, changed runtime/plan, or unrelated nonpass cannot be allowed.
The aggregate identifies allowed_precision separately from pass;
no underlying receipt is rewritten. The explicit reference IDs are
review input, not auto-trained tolerances. They may be reused for
later runs of that same observed plan while its inputs/state match.
Do not apply this to a new binary or voltage/clock candidate; actual
factory-confirmed reference reuse across changed controls remains
gated by the power owner.

Source: AMD RDNA3 Shader Instruction Set Architecture, section 7.9
(https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf):
WMMA internally uses dot instructions and specifies rounding mode but
does not provide a bit-exact internal reduction model. Independent
7900 XT research corroborates cancellation residues and order
sensitivity (https://github.com/j0nsh1n/rdna3-fp8, matrix arithmetic
measurements). Neither is a general numerical waiver.

Observed evidence: both cards agree on all eight full-output fingerprints
for each of the three instruction cases, including both gap patterns.
18 independent captures (36 completed cases), 10244849664 compared words
including repeated observations; maximum absolute errors below:

| Case | Max absolute error | Complete same-card reference/repeat sets |
| --- | ---: | --- |
| dot2-rdna3-reference | 1.1920928955078125e-7 | helper0, helper1 |
| wmma-fp16-rdna3-reference | 2.384185791015625e-6 | helper0, helper1 |
| wmma-bf16-rdna3-reference | 2.384185791015625e-6 | helper0, helper1 |

Managed plans: 20261126T042742Z-gpu-ai-qualification-plan-337ad0bf6aee,
20261126T042743Z-gpu-ai-qualification-plan-89f54ae31455.
Both complete independent reports pass with precisely the three allowed
precision groups and all remaining screen groups strict:
20261126T043305Z-gpu-ai-qualification-report-8d680c97f580 and
20261126T043753Z-gpu-ai-qualification-report-001a1473983b.
Without explicit references, their respective reports
20261126T043304Z-gpu-ai-qualification-report-fb81ac154baf and
20261126T043752Z-gpu-ai-qualification-report-26bd346a7b28 refuse all
three signed groups. The initially refused helper1 BF16 repeat was
resumed after the GPU lease became available; the two additional
captures are 20261126T043540Z-gpu-ai-screen-f0dcc1b07107 and
20261126T043550Z-gpu-ai-screen-b4710ae77cf4. All observations retained
original controls. Production and native qualification remain separate.

### Exact workaround experiments

The strict dot2-exact case uses FP32 vector multiply/add with the same
signed dyadic inputs. The two wmma-*-bias cases transform the bounded
integer-valued signed corpus into nonnegative inputs (A+1, B+1), use
a native WMMA with zero accumulator, then correct with the row/column
sums and original C. Algebra: AB = (A+1)(B+1) - sum(A+1) -
sum(B+1) + K, at each output coordinate. Every result must equal the
original exact signed CPU oracle, not the recorded RDNA3 deviation.

Both gfx1100 cards passed all 12 sustained/burst cases:
20261126T042608Z-gpu-ai-screen-2dac27928eed and
20261126T042630Z-gpu-ai-screen-07404c152d83, unchanged controls.
Disassembly of the final code object confirms vector FP32 FMA
for dot2-exact and actual FP16/BF16 WMMA plus FP32 correction for the
bias cases on gfx1100/gfx1201. gfx1030 has no WMMA. Workaround
success applies only to this exact bounded corpus. General arbitrary
floating inputs need sign-separated positive/negative products,
range/exponent analysis, quality and performance qualification; four
nonnegative WMMAs may be much more expensive. Prefer existing exact
integer WMMA where representation permits. Do not change native
serving kernels solely on this diagnostic.

The operational recipe selects FP32 on gfx1201 (previous effective
heat-producing workload) and BF16 on gfx1100/gfx1030. It retains
the 60 C required final window and 80 C example ceilings, not a
weakened thermal gate. Peer access and kernel health are still
unavailable: the unprivileged kernel journal query returned no
JSON kernel records, so it cannot replace the inaccessible kmsg
cursor. No permissions, clocks, voltage or caps were changed.

### Performance and adoption decision (2026-11-26)

The RDNA3 precision allowance changes the independent qualification path only;
no Radiance serving kernel, dispatch, power control or production profile has
changed. No serving-throughput A/B was performed, so do not claim a measured
zero-percent throughput change. The 12 workaround passes establish correctness
on the bounded signed corpus, not comparative kernel speed. In particular,
launch counts from the host-verified probes include CPU oracle generation,
readback, comparison, launch overhead and optional sleep; they cannot be used
as native throughput or equivalent-work energy measurements.

Static inspection of the ROCm 7.2.4 gfx1100 protocol-7 code object finds four
native packed-dot instructions in the original signed dot2 case, whereas
`dot2-exact` uses vector FP32 FMA/multiply, with less input packing in this
synthetic probe. Each signed FP16/BF16 WMMA variant uses one native WMMA;
the biased variant still uses one WMMA, but adds operand construction,
row/column sums and correction arithmetic. Static instruction counts and
instruction classes are not execution time, occupancy or serving throughput.
Sign splitting may require four native products; compensation may not recover
error already introduced inside a native instruction. Neither is qualified.

Adoption gate: use RCS27/RCS28 existing kernel-family selection and native
replay; identify actual signed floating WMMA/dot users from selected model
rows (not all IQR or integer kernels). First demonstrate measurable quality
benefit versus the shipped path, with the appropriate numerical contract;
then compare equivalent-work kernel timing, registers/LDS/spills/occupancy,
matched cold/warm shape replay, and real prefill/decode throughput, latency,
quality and phase energy through the existing owner workflow. Record all
rejected variants and results. Do not promote a slower or lower-quality
replacement; the currently shipped fast kernels remain the default. The
precision-reference exception does not authorize voltage changes or
candidate qualification with changed controls.


### Progressive early screening (PWR14)

Use `gpu-ai-qualify plan --recipe sanity` for two seconds of nominal
ALU/BF16 work, or `--recipe slim` for architecture-selected matrix types,
six direct instruction cases with 2 ms gaps, supported positive/mixed
integer WMMA cases and a short address-pattern VRAM integrity test.
Slim overrides duration/pause/memory fraction of existing groups; it
does not duplicate the case registry, modify other recipes, relax the
verification counts, or alter the thermal safety ceilings. Memory is
1/128 of total VRAM, rounded down within available headroom; not a
full-capacity screen. Native WMMA on gfx1030 is explicitly not applicable.

Execute `gpu-ai-qualify run --plan PLAN --all` within the existing GPU
lease. The executor rechecks idle/lease/source/physical identity at each
group, stops on the first nonpass, harvests each completed capture and
emits the existing managed report including recipe, planned nominal
workload seconds and not-run groups. Precision-reference groups require
explicit individual runs and reference receipts; all-group mode refuses
them before work. Nominal time excludes setup, teardown and lease wait.

A slim pass is not a screen/extended, signed precision, hot, full-memory,
native model kernel or production pass. Use it for early rejection;
promising candidates escalate through screen, operational and extended
then the existing RCS27/28 and PWR10/PWR02 owners. The qualification
runner remains observe-only until owner recovery gates permit candidate
transactions; no profile promotion or automatic sweep is introduced.


Observe-only cohort (2026-11-26, protocol 7/schema 8; end-to-end
plan + run + automatic harvest/report elapsed, excludes queue wait):

| Role / architecture | Recipe | Workload s | Wall s | Cases | Verified GEMMs | Managed report |
|---|---|---:|---:|---:|---:|---|
| helper2 / gfx1030 | sanity | 2 | 3.554 | 2 | 38,496 | 20261126T054234Z-gpu-ai-qualification-report-b53e9d266b22 |
| helper2 / gfx1030 | slim | 11 | 15.270 | 11 | 81,472 | 20261126T054249Z-gpu-ai-qualification-report-77ac6abb4637 |
| helper0 / gfx1100 | slim | 14 | 19.448 | 14 | 113,408 | 20261126T054309Z-gpu-ai-qualification-report-3636035c07cc |
| helper1 / gfx1100 | slim | 14 | 19.259 | 14 | 114,816 | 20261126T054329Z-gpu-ai-qualification-report-25e0331d8f4a |
| engine / gfx1201 | slim | 15 | 21.847 | 15 | 101,600 | 20261126T054351Z-gpu-ai-qualification-report-567c56a6ab31 |

All 56 cases and all five reports passed within their declared scope;
16 group captures harvested, unchanged controls and physical identities,
no transaction requested. Driver health remains incomplete. Existing
screen nominal load budgets are 88/128/136 s and extended 460/500/508 s
(gfx1030/1100/1201). These are not measured comparison wall times.
Persistent GPU contexts are not required to achieve the current target;
evaluate setup overhead further before changing the native protocol.
