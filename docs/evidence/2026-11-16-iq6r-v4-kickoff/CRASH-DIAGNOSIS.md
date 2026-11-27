# IQ6R-V4 managed converter SIGSEGV diagnosis

Date: 2026-11-16. State: **gfx1100/offload mismatch confirmed by matched failure/success; full IQ6R artifact still pending**.

## Observed failure

The managed CI-installed release `bc-radiance-1170.26.0` converter exited from signal 11 (SIGSEGV) after 1,329.55 seconds. The `conversion-status-managed.json` receipt records a failed experimental conversion and no production promotion. The incomplete ~13.45 GB output is **invalid** and was deleted after preserving its crash stderr and receipt; no usable Q6 RAD or quality/TP3 evidence exists.

The preceding older experimental converter also crashed with signal 11, although that run had incorrect `CALIB` inputs and was falling back to rounding. It cannot serve as evidence about the corrected-calibration path or release equivalence.

The managed converter logged `libquant: GPTQ quantises 'blk.23.attn_v' against 1 calibration file(s), K=5120` immediately before log silence. This proves that at least this matrix passed Hessian acquisition and factorization; a buffered concurrent log is **not** a precise crash-site attribution. No missing-calibration fallback was observed in its stderr. The managed release runtime matched its installed manifest SHA.

## Calibration validation independent of converter

The original 18 BF16 shards match the previously recorded SHA-256 hashes. All 257 published source Hessians match pinned upstream LFS SHA-256. Their RADGRAM2 expansion covers 289 converter-visible stems, with full metadata and dimension checks.

A separate `iq6r_v4_verify_grams.py` audit compared **12,593 sampled cells** from the source packed upper triangles and their full row-major RADGRAM2 counterparts, checking transposed positions, all 257 sites and hardlink aliases; all samples were byte-exact. The K=5120 Hessian used for the layer-23 attention inputs has 5,120 finite, strictly positive diagonal values (min 1,097,026.75; max 26,704,031,744). This makes trivial packing, missing-file and non-finite-diagonal explanations unlikely, but does not prove every matrix is positive definite or establish model quality.

## CPU versus GPU and hardware

SIGSEGV is a **host-process** termination; without a backtrace from the actual fault, GPU versus CPU memory ownership cannot be assigned. A snapshot captured with GDB during a single-XTX focused reproduction (interrupted deliberately after 272 seconds, *not* a spontaneous crash) showed a worker inside `libquant.so!lq::factor_impl` via `gptq_factor_ordered`, with ROCm HSA background threads in driver ioctl waits and OpenMP workers mostly idle. This confirms that substantial GPTQ factor work is on the host before GPU trellis execution. Do not mistake this manually interrupted snapshot for the original crashing stack.

At the observed time, host RAM was far from exhausted (approximately 79 GiB available), and the original conversion ended in SIGSEGV rather than a kernel OOM SIGKILL. The R9700 reports zero correctable/uncorrectable RAS memory errors; the XTX cards do not expose comparable counters. These are observations, not a full host-RAM/VRAM hardware diagnostic.

## Focused reproduction and decision tree

A scratch diagnostic recipe retains the standard W8A8 model except that `blk.23.attn_v.weight` uses the exact primary IQ6R-V4 GPTQ recipe rule. `rad-convert --plan-only` accepts this recipe with `--reuse` of the existing read-only 27B W8A8 RAD. It avoids unnecessary requantization and production edits. The bounded debugger runner is `tools/lab/radiance/iq6r_v4_crash_probe.py`; its mode is single XTX, dual XTX, or CPU-only. Every device test requires an exclusive lease. All diagnostic outputs are unqualified, never promoted, and refuse overwrite.

- If the isolated single-XTX conversion crashes with a GDB backtrace in HIP/Trellis, suspect GPU quantization or its host API path. Compare CPU-only and single-/dual-XTX executions, then use targeted memory diagnostics as justified.
- If only dual-XTX fails, investigate device split, per-device pointer ownership and concurrency.
- If CPU-only fails in `factor_impl`, `lq_gptq`, BLAS or OpenMP, isolate host tensor shapes, factor caches, and thread lifetime; a GPU-memory test is not the first response.
- If the focused single-XTX conversion succeeds, the original full-model SIGSEGV may depend on concurrent quantization, resource lifetime or another tensor; capture full-conversion GDB evidence before claiming a cause.

All raw debugger logs and receipts are in the isolated work root under `${BC_WORK_ROOT}/iq6r-v4-qualification/`; do not commit multi-GiB models or calibration data. Production W8A8 remains unchanged.

## Decisive single-XTX reproduction

The isolated focus recipe was accepted by a managed converter plan (only `blk.23.attn_v.weight` at original IQR-V4 GPTQ settings, all other weights reused from read-only W8A8). Under an exclusive lease, GDB reproduced SIGSEGV on one gfx1100 XTX. The last logged operation was successful GPTQ Hessian factorization for `blk.23.attn_v` (K=5120). The crash backtrace was `libamdhip64.so.7` -> `libquant.so!run_rows` -> `gptq_tcq_gpu` -> `gptq_quantise` -> `gptq_quantize` -> `rad::quantise_weight`. Disassembly of managed `libquant.so` at the return PC offset `0x264e20` identifies the preceding instruction at `0x264e1b` as `call hipLaunchKernel@plt`. The host HIP launch itself segfaulted before returning; this is not evidence of a GPU kernel executing and faulting on the device.

## Managed offload architecture mismatch

`llvm-objdump --offloading` of the exact CI-installed managed `radiance_home/quantizers/libquant.so` reports one GPU bundle: `hipv4-amdgcn-amd-amdhsa--gfx1201`. It contains **no gfx1100 offload bundle**. The original runner set HIP_VISIBLE_DEVICES to XTX indices 0,1 and asked the quantizer to use device ordinals 0,1. The single-XTX repro likewise selected gfx1100, whereas available `libquant` kernels were only compiled for gfx1201 (the R9700). This is a concrete arch/device selection defect and the leading explanation for the reproducible `hipLaunchKernel` SIGSEGV. The managed engine recipe declares a gfx1201 GPU target; gfx1100 serving support comes separately from `libr3`. Serving kernel compatibility does not imply the quantizer offload binary supports those GPUs.

Corrective validation: select only the R9700 for quantization, using HIP_VISIBLE_DEVICES=2 with local `--gpus 0`, while retaining the `libr3` serving/layout preflight. Run the identical focused one-projection GDB repro first; then gate any full conversion on a passing GPU architecture match. An alternative managed release with explicitly compiled gfx1100 libquant support would also be valid but requires its own release-quality build. No source/vendor edits, in-place production modifications or unsupported cross-architecture HIP launch are acceptable.

## Corrective R9700 A/B — completed

Using the same CI-managed converter, source checkpoint, verified layer-23 calibration Gram and exact IQ6R-V4 projection rule, the focused reproduction was repeated with HIP_VISIBLE_DEVICES=2 (the **gfx1201 R9700**, local device 0) and libr4d/portable reference layouts. The initial R9700 attempt that accidentally selected libr3 (gfx1100 only) failed cleanly at architecture declaration before writing weights; the corrected libr4d run succeeded. The diagnostic GDB log reports normal process exit after 416.5 seconds; no HIP SIGSEGV or missing-calibration fallback.

The controlled R9700 run produced a valid RAD2 file accepted by managed `rad-info`: 1,038 weights, ~27.94 GiB, with 1,023 weights copied from read-only W8A8 and 15 weights written (only one at the target IQ6R trellis encoding). Layer-23 attention-value projection relative Frobenius weight error was 1.974e-02, which is **not** a whole-model quality acceptance metric. Diagnostic logs, the GDB receipt and the rad-info report are retained. The temporary 28 GiB focus model was then deleted; production W8A8 was never written.

This matched A/B confirms the gfx1100 failure arose from selecting a GPU without a compatible quantizer image. The gfx1201 operation succeeds. It does NOT establish whole-model conversion, TP3 runtime or quality acceptance. The tracked experimental runner now selects the R9700 for GPU quantization and validates the quantizer GPU bundles with llvm-objdump --offloading before launch; layout preflights remain independent for gfx1100/libr3 and gfx1201/libr4d.

Separately, managed libquant does not link OpenBLAS and therefore uses slow scalar CPU Hessian factorization. This explains much of the latency but is not the immediate HIP launch SIGSEGV. A performance improvement needs its own managed build/validation gate and must not be conflated with the device selection repair.

## Independent libr4d selftest report (separate agent, same host)

A separate agent reported a distinct but matching *device-selection* fault: `r4d_selftest` launched `libr4d` (gfx1201-only) with no HIP device pinned, so HIP chose default physical device 0 (gfx1100). GDB showed `main -> device_phase -> run_case -> r4d_rad_relu -> r4d_relu_bf16 -> libamdhip64.so.7` before SIGSEGV. With `HIP_VISIBLE_DEVICES=2`, its `relu_bf16` and `clamp_bf16` numerical rows succeeded. This is **reported by that agent**, not independently re-run as part of the IQ6R qualification. It corroborates the unsupported-device HIP launch mechanism but says nothing about the validity of other selftest rows.

That agent explicitly invalidated the selftest smoke-test portion of its baseline: only `silu_bf16`/`relu_bf16` were inspected and no 201-row replay was performed under correct device pinning. The catalogue outputs were printed before device code and can remain valid. Do **not** report the other ~200 rows, including `kv_store_b16i8` and `attn_prefill_h256_gqa6_b16i8`, as either genuine kernel defects or verified passes. Their properly pinned rerun belongs to the selftest owner, not to the IQ6R conversion experiment.

## ROCm 7.2.4 HIP runtime source inspection

A read-only checkout provides the ROCm source at `tools-src/rocm-systems-7.2.4/projects/clr/hipamd/src/` (read-only). In `hip_platform.cpp`, `ihipLaunchKernel` calls `PlatformState::getStatFunc(&func, hostFunction, deviceId)`; on lookup failure or null function it **reinterpret_casts the host function pointer as a hipFunction_t** and continues into `ihipModuleLaunchKernel`. The lookup traverses `hip_code_object.cpp:StatCO::getStatFunc` and `hip_global.cpp:Function::getStatFunc`, which explicitly return `hipErrorNoBinaryForGpu` if the selected device has no compiled target. This is a concrete unsafe-looking fallback path to examine against the symbolized HIP SIGSEGV, not yet a proven precise faulting expression. The expected behavior for unsupported device image would be a clean HIP error rather than a host SIGSEGV. Do not patch the installed ROCm runtime during the BigCherry experiment.

## Second XTX matched reproduction (2026-11-16)

Physical GPU 1 (second RX 7900 XTX / gfx1100) was selected with HIP_VISIBLE_DEVICES=1 and the local --gpus 0 ordinal, under a fresh exclusive lease. The identical managed rad-convert, isolated blk.23.attn_v IQ6R-V4 rule, calibrated Gram, W8A8 reuse baseline and libr3 layout reproduced SIGSEGV after 400.34 seconds. GDB reported the **same normalized HIP runtime fault** (libamdhip64.so.7 offset consistent with GPU 0), with `run_rows -> gptq_tcq_gpu -> gptq_quantise` in the call chain. No evidence of a device kernel executing was established. The 28 GiB invalid partial diagnostic output was removed; GDB log and receipt remain in isolated scratch. The exclusive lease was released.

Three-card controlled matrix: gfx1100 XTX GPU 0 SIGSEGV; gfx1100 XTX GPU 1 SIGSEGV; gfx1201 R9700 GPU 2 successful RAD2 conversion and rad-info inspection (then discarded as unqualified mixed W8A8/IQ6R diagnostic). Together with the gfx1201-only managed libquant offload bundle this strongly supports unsupported quantizer architecture/device selection rather than defective memory on an individual XTX. It does NOT diagnose the rest of the TP3 system or validate complete IQ6R conversion.

The ROCm source checkout used for static tracing was rocm-systems-7.2.4, Git revision `97f5574fe2fdc7bef44fb01545347912ee9f1779` (clean). `hip_platform.cpp:689-719` contains `ihipLaunchKernel` and the fallback reinterpretation of a host-function pointer to hipFunction_t on `getStatFunc` lookup failure. `hip_global.cpp:149-166` can return hipErrorNoBinaryForGpu when a device image is absent. `hip_module.cpp:440-469` dereferences the resulting DeviceFunc handle during module launch. These lines describe a plausible runtime implementation flaw for an unsupported architecture; we have not symbolized the five internal HIP stack frames to prove the exact failing dereference. The upstream issue should request a clean `hipErrorNoBinaryForGpu`/`hipErrorInvalidDeviceFunction` instead of SIGSEGV, with the three-GPU A/B and GDB traces.

## Vendored ROCm 10.1.0rc2 on-host comparison

A separate minimal gfx1201-only HIP kernel was compiled twice, against installed ROCm 7.2.4 and the complete alternative SDK at `vendor/rocm/10.1.0rc2`. Under a lease, the correctly pinned six-case matrix proved that 7.2.4 segfaults on **both** gfx1100 XTX cards while 10.1.0rc2 returns clean HIP error 200 (`device kernel image is invalid`) on both; both SDKs successfully launch on the matching gfx1201 R9700 and return the correct output. Each executable loaded its own HIP runtime, confirmed by ELF RUNPATH and loader trace. See `ROCM-VENDOR-ABI.md` for the full matrix and reproducible source/runner. This validates safer error handling in vendored 10.1.0rc2 but does not authorize an in-place production runtime swap or make gfx1201-only quantization kernels executable on gfx1100.
