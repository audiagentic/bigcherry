# Vendored HIP wrong-architecture A/B — local validation, 2026-11-16

## Question and scope

Does the locally vendored alternative ROCm SDK prevent the host SIGSEGV when a HIP kernel compiled **only** for gfx1201 is launched on either gfx1100 XTX? This tests **invalid-device-image error handling**, not performance, IQR-v4 numerical correctness, TP3 serving compatibility or substituting another HIP runtime into the installed production release.

The repo has one alternate full vendored runtime at `vendor/rocm/10.1.0rc2`; it contains a HIP compiler and `libamdhip64.so.7`. `vendor/rocm/10.1.0rc2-campaign` contains campaign compiler shims, not a second complete runtime. The installed baseline is `/opt/rocm-7.2.4`.

## Method

- Isolated source, committed with this record: `tools/lab/radiance/hip_wrong_arch/launch_probe.hip`, a single GPU kernel writing integer **42**.
- Independently compiled with each SDK's `hipcc --offload-arch=gfx1201` into binaries under `${BC_WORK_ROOT}/iq6r-v4-qualification/hip-wrong-arch/`. No extra gfx1100 image.
- Each executable has an SDK-specific ELF RUNPATH and a NEEDED reference to `libamdhip64.so.7`; an explicit process-local `LD_LIBRARY_PATH` selects that same SDK. `LD_DEBUG=libs` proved that **all six** tests loaded the expected version-specific HIP runtime.
- Under an exclusive GPU lease, run each binary in an isolated process with **only** `HIP_VISIBLE_DEVICES` set to physical GPU 2 (R9700/gfx1201), GPU 0 (XTX/gfx1100), then GPU 1 (second XTX/gfx1100). Do **not** simultaneously apply `ROCR_VISIBLE_DEVICES` with the physical index; the first matrix mistakenly double-filtered device indices and made GPUs 1 and 2 invisible. The corrected six-case matrix below is authoritative.
- Each process calls `hipGetDeviceProperties`, `hipMalloc`, `hipLaunchKernelGGL`, `hipGetLastError`, `hipDeviceSynchronize` and reads the integer back on success.
- The test affects no production model, system ROCm installation or device files. No GPU lease remains held.

## Verified corrected results

| Binary SDK/runtime | GPU | Runtime-visible ISA | Launch result | Process exit |
| --- | --- | --- | --- | --- |
| ROCm 7.2.4 | physical 2, R9700 | gfx1201 | launch/sync/copy success, result **42** | **0** |
| ROCm 7.2.4 | physical 0, XTX #1 | gfx1100 | **SIGSEGV before hipGetLastError** | **-11** |
| ROCm 7.2.4 | physical 1, XTX #2 | gfx1100 | **SIGSEGV before hipGetLastError** | **-11** |
| Vendor 10.1.0rc2 | physical 2, R9700 | gfx1201 | launch/sync/copy success, result **42** | **0** |
| Vendor 10.1.0rc2 | physical 0, XTX #1 | gfx1100 | HIP returns **200**, `device kernel image is invalid` | **10** (intentionally nonzero on launch error) |
| Vendor 10.1.0rc2 | physical 1, XTX #2 | gfx1100 | HIP returns **200**, `device kernel image is invalid` | **10** |

Evidence: `${BC_WORK_ROOT}/iq6r-v4-qualification/hip-wrong-arch/matrix-results.json`. The run log is under `/scratch/home/.remote-agent/bg-1794819935090.log` and prints the same outcomes.

The two binaries are fingerprinted:
- 7.2.4: `86aa8bbaf443c097be934adb40eb936f74e6cbf755deb005cade10302a043682`
- 10.1.0rc2: `9937a5ed316cf750f3afd62965c3b7700d5e43ec236c0767e42dbce1e539785d`

## Interpretation and recommended action

**Confirmed**: the vendor 10.1.0rc2 HIP runtime corrects the **error behavior** for this independently compiled wrong-ISA repro. The system ROCm 7.2.4 runtime segfaults on both gfx1100 cards; the vendor runtime returns error 200 on both. Both versions successfully execute matching gfx1201 device code. This provides direct, on-host confirmation of the unsafe launch/error-handling behavior identified from upstream HIP source.

**Not established**: a replacement ROCm 10.1.0rc2 runtime will safely run the existing production Radiance build, which was linked and qualified with 7.2.4; an unqualified runtime swap is not permitted. The vendor runtime does not provide a gfx1100 image to a gfx1201-only `libquant.so`. Running full 27B IQR-V4 conversion, model acceptance and TP3 inference on vendor ROCm are separate tasks requiring a managed SDK build, controlled tests and rollback.

**Recommendation**: Keep production pinned to ROCm 7.2.4. Keep the offload ISA preflight that prevents invalid launches. Perform full IQR-V4 conversion with the managed quantizer on the R9700/gfx1201. Use this minimal repro and the six-case matrix as regression evidence for any future SDK upgrade; consider a separate compatibility campaign if evaluating vendor 10.1 for production.

## How to repeat

Build against each SDK's `hipcc --offload-arch=gfx1201` using the committed `launch_probe.hip` and an output binary under the experiment work root. Under an exclusive GPU lease run `run_matrix.py` with `BC_WORK_ROOT` configured. The runner refuses to start without the lease, selects the corresponding runtime locally for each process, records exit codes/launch messages, and writes a JSON receipt. Expect a clean error for both XTXs under vendor ROCm; never interpret this expected launch error as a working XTX kernel.
