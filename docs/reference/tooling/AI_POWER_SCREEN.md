# Independent AI power-profile screening (PWR12)

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
    -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lhipblaslt -lrocblas \
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
confirmed stock receipt, verifying current controls and checking thermal
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
`--stock-confirmed`; the output cannot be used as a stock candidate reference.
Use a managed stock run ID. Previous standalone JSON is historical diagnostic evidence only; re-record stock through the managed command.

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
timestamps and sampled temperatures must bracket the entire final requested
interval with no sample gap above 1 s and every sample at or above the
minimum; if it never reaches that interval, the case is inconclusive even
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
`$BC_WORK_ROOT/runs/capture/<run-id>/` (subkind gpu-ai-screen, plan PWR12,
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
reference must be a completed managed stock run with passing coverage and
explicit factory confirmation. Retain the typed run reference and digest
of its exact receipt. The old arbitrary --output argument is removed.
No engine runs or hardware settings are changed by this output alignment.

## Numerical and identity contract

The native probe uses actual HIP-visible PCI identity and validates the
expected architecture before allocation. It selects a hipBLASLt heuristic or explicitly uses the installed rocBLAS
public default GEMM dispatch with atomics disabled. It records provider and
library versions; hipBLASLt also records opaque algorithm bytes, while
rocBLAS honestly records default dispatch, not an exact internal solution ID. It
checks every active HIP/library operation. The corpus consists of two
deterministic signed matrices. Floating inputs are exactly representable
+/- 1/16, integer inputs +/-1. A CPU independent integer dot sum is exact
for the chosen dimension range. The outputs are FP16 for FP16, FP32 for
BF16/FP32/E4M3 FP8, and INT32 for INT8. The sole floating bit-equivalence
exception is +0 versus -0; no tolerance is applied to other values. Poison
the destination before every matmul and read back all elements after
synchronizing. Require at least two successful iterations and elapsed
coverage. Sustained mode has zero inter-launch pause; positive pause
exercises burst/idle transitions, not an assertion of full GPU saturation.

The stock receipt and candidate must match physical BDF/architecture, role,
probe binary digest, kernel version, types, geometry, duration and pause
recipe. Compare actual HIP/driver/library versions, provider, dispatch identity and
workspace for each case. A rocBLAS default route does not prove identical internal
kernel selection. Every iteration is independently checked against the CPU oracle. Verify observed controls before, between and after
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

Actual clock/voltage/cap application was deliberately not performed.
The controller integration is tested on fake sysfs; real changed-profile
validation, representative extended hot/resident soak, driver-health and
production replay/KLD/phase-energy evidence remain PWR12's physical
qualification boundary with PWR10/PWR02. Never interpret the screen as
permission to alter hardware. The existing active plans were not edited.
