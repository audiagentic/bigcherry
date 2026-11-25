# Independent AI power-profile screening (PWR12)

## Decision and boundary

This is **platform qualification tooling**, not a Radiance component or
serving-time gate. It can run with no model, engine build, serving process,
PyTorch, Windows application or downloaded ROCm package. The platform ships
source in the existing tools tree; build the small executable on demand with
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
transaction remain responsible for consent, actual hardware changes, recovery
and restoration. PWR12 must not call a second writer. PWR10 and PWR02 remain
responsible for qualification and performance/energy promotion. A screen is
a diagnostic result, never production authorization.

## Run independently

From a repository checkout, using the installed ROCm SDK:

```sh
test -n "${BC_WORK_ROOT:?set the project work root}"
test -n "${ROCM_PATH:?select the installed ROCm SDK}"
mkdir -p "$BC_WORK_ROOT/ai-power-screen"
nice -n 19 "$ROCM_PATH/bin/hipcc" -O2 -std=c++17 -Wall -Wextra \
    tools/bigcherry/hardware/ai_probe.cpp \
    -L"$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lhipblaslt \
    -o "$BC_WORK_ROOT/ai-power-screen/ai-probe"
"$BC_WORK_ROOT/ai-power-screen/ai-probe" --self-test
```

The self-test is CPU-only. For real GPU work, use the verified GPU lease,
one selected physical PCI BDF and expected architecture. Do not run during
another workload. Use `PYTHONPATH=tools python3 -m bigcherry.hardware.ai_probe --help`
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
    env PYTHONPATH=tools python3 -m bigcherry.hardware.ai_probe \
    --probe "$BC_WORK_ROOT/ai-power-screen/ai-probe" \
    --profiles engines/radiance/power-profiles.yaml \
    --profile stock --stock-confirmed \
    --bdf "$GPU_BDF" --arch "$GPU_ARCH" --role "$GPU_ROLE" \
    --dtype fp16 --n 768 --seconds "$SCREEN_SECONDS" --pause-ms 0 2 \
    --max-hotspot-c "$SAFE_HOTSPOT_C" \
    --output "$BC_WORK_ROOT/ai-power-screen/stock.json"
```

For a candidate, invoke the same platform command **inside the existing
authorized power transaction's lease and child-lifetime scope**, after the
owner has applied and verified the candidate. Change only `--profile`,
replace `--stock-confirmed` with `--reference <stock.json>`, and choose a
new output file. The operator/transaction must keep the lease through probe
exit and restore. Do not manually leave a candidate applied across leases.
There is intentionally no independent apply command in this first slice;
the ongoing native power owner must supply an authorized test-only
transaction before unattended candidate qualification. Empty placeholder
profiles (e.g. efficient/quiet/performance without rules) are refused, not
treated as successfully tested. The candidate must match each requested
setting; other observed controls and device identity must match the stock
reference. Unsupported settings and absent required readbacks refuse.

Use a single supported dtype initially. Multiple requested types and burst
gaps run in order, stopping at the first nonpass; a supported subset never
silently passes an unsupported request. Choose `--seconds` per case and
a lease longer than total cases plus setup/cleanup. Output files are separate
evidence, not launch inputs, and must not be overwritten. Existing atomic
power-receipt writing is reused. A supervisor may stop the screen at the
explicit safe ceiling or a lower available driver-reported limit. Missing
junction telemetry, failed identity, timeout, runtime error, numerical
mismatch, lease loss, or an interrupted run does not pass. This is a
polled guard, not a thermal control system. Hard driver hangs/SIGKILL cannot
guarantee reaping: verify child exit and retain the existing owner recovery
fence before any restore or new mutation.

## Numerical and identity contract

The native probe uses actual HIP-visible PCI identity and validates the
expected architecture before allocation. It selects a hipBLASLt heuristic
from the installed SDK, records opaque algorithm bytes and versions, and
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
recipe. Compare actual HIP/driver/hipBLASLt versions, opaque heuristic and
workspace for each case. Verify observed controls before, between and after
cases; record the profile digest and reference receipt digest. A profile
label alone is not evidence. The output includes the observed junction
temperature range. This is a short independent arithmetic screen, **not**
proof of factory state, hot steady-state coverage, loaded library content
identity, all matrix instructions, memory residency, driver reset absence
or complete inference stability. A nonexact provider at the stock reference
is unqualified and must not be interpreted as undervolt instability.

The next PWR12 slices should reuse actual native kernel replay for
IQ8R/IQR/attention/GDN/MoE, especially gfx1030 where hipBLASLt is absent;
reuse existing memory/transfer integrity tests with measured safe budget.
Add a nonexact floating corpus only with independently justified tolerance
and a frozen stock reference. Do not introduce another kernel backend
framework or make every profile test every model. Full hot conditioning
under normal cooling, memory temperatures, cold and hot transitions,
driver health, actual production replay/quality and synchronized phase
energy belong to the finalist validation, using existing PWR10/PWR02.
No forced cooling, factory reset, crash-edge search, voltage change or
profile promotion by this tool.

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
verdict. No actual power controls or profile settings were changed. No
temperature conditioning, near-full VRAM, stock-vs-candidate, energy
qualification or engine replay was performed in this slice. PWR12 remains
open for those independent qualification extensions.
