# RSC06 helper kernel configuration audit

The helper loads one kernel DSO using RTLD_NOW | RTLD_LOCAL. It now requires
bc_r4d_configure_helper immediately after dlopen, before ABI queries, optional
open callbacks, registry enumeration, layouts, scratch queries or launches.
Missing, rejected or repeated configuration fails startup. It copies a
size-checked fixed-width struct once, before threads start, from immutable
HelperSettings. There is no helper environment fallback.

## Reachable environment reads

Scope: the helper translation unit, transitive application headers, and the
shipped libr3/libr2/libr4d registry and callbacks. Vendor HIP/HSA and the system
loader retain their runtime environment contracts (device visibility, queues).

The first four reads formerly occupied native/r4d_moe.hip.rw lines
162/164/166/168; the shared trace read was upstream r4d_args.h:194.

| Current file:line | Variable | Reader now | Former helper route |
| --- | --- | --- | --- |
| libr3/r3_moe_settings.h:9 | R3_MOE_NATIVE | Engine only | dlopen libr3 -> MoE W4/FP8 launcher -> native rewrite |
| libr3/r3_moe_settings.h:10 | R3_MOE_FP16 | Engine only | Same launcher; f16 after native |
| libr3/r3_moe_settings.h:11 | R3_MOE_AHALF | Engine only | Same launcher; activation half staging after f16 |
| libr3/r3_moe_settings.h:12 | R3_MOE_PROBE | Engine only | Same launcher; timing ablations after f16 |
| Patched upstream libr4d/r4d_args.h:196 | RAD_R4D_TRACE | Engine only | MoE/quantisation/gather/layout refusal -> R4D_NO -> r4d_trace_on |
| moe-remote/r3_moe_worker.cpp:92 (write) | GPU_MAX_HW_QUEUES | HIP runtime | Derived queue count before HIP; the one sanctioned write |

Paths above are relative to components/, except explicitly upstream paths.
The native launcher now reads typed values at libr3/native/r4d_moe.hip.rw:162.
The diagnostic accessor checks typed configuration before the cached engine
environment read, including on error paths. libr2 and stock libr4d use typed
diagnostics; libr3 alone applies the native gfx11 MoE rewrite.

The helper selects moe_gemm_w4a8, moe_gemm_fp8a8, had_quant_act_fp8,
gated_had_quant_fp8, quant_act_fp8, gated_quant_fp8 and moe_gather_bf16.
It invokes MoE launch/scratch and weight layout/relayout callbacks. Enumerating
the registry does not execute other rows. Shipped kernel plugins have no
plugin-open callback reading application settings. ABI/container helpers and
helper lane kernels contain no environment reads.

These reads are not on the helper call graph:

- moe_remote_hook.cpp worker, shadow, mask, CPU, SHM, row, layer, burst,
  histogram and combine settings; moe_mailbox_ops.hip R3_MOE_WORKER_FUSESTAGE:
  engine-side libr3moe, never loaded by the helper.
- libr3 R3_AR_TIMING/WIRE4 (all-reduce), R3_ATTN_NATIVE (attention),
  R3_GEMV, R3_DEC11, R3_D11_MODE, R3_BT/B/NT/ABL, R3_MXD_KS/BK/NT
  (dense MXFP4): different callbacks; flags are function locals, not load-time
  initializers.
- Upstream RADIANCE_GDN_ANCHOR (recurrent), RADIANCE_MLA_CONFLICT_STATS
  (MLA), R4D_F8_MWLOG (dense FP8), R4D_AR_MAX_BLOCKS (all-reduce),
  R4D_GEMMCFG (dense GEMM resolve). Including r4d_gemm_args.h in the MoE unit
  does not call its resolve function.
- Self-test reads are in separate executables; shelved GDN rewrites are not
  applied.

## Default preservation and engine boundary

git log -p origin/main -- engines/radiance/components/moe-remote/r3_moe_worker.cpp
shows these calls in commit 791710af9's worker split, retained at origin/main:

    setenv("R3_MOE_NATIVE", "1", 0);
    setenv("R3_MOE_FP16", "1", 0);
    setenv("R3_MOE_AHALF", "1", 0);

Thus all three default on. The schema and HelperSettings retain true/true/true,
now delivered to the actual launcher. Explicit environment overrides become
explicit manifest values. Probe defaults to zero and refusal tracing to false,
matching unset environment. Nondefault probe/trace require the experiment
profile. Native/f16/half dependencies and launch-mode packing are unchanged.

An engine does not install helper configuration, so its original four
getenv/atoi semantics, including probe masking, remain. RSC07 owns those
reads, shared refusal tracing, and engine-only plugin/core configuration.

## Application and verification

Upstream edits are exclusively in managed patch rad0010_helper_kernel_config.
It instantiates the config in the row table shared by libr4d/libr3/libr2 and
updates r4d_args.h. Normal, lab and diagnostic source recipes require it.
libr3's owned native rewrite uses r3_moe_settings.h via the existing rewrite
pipeline. Build a matching helper/plugin pair; an old plugin is rejected.

Card-free DSO tests use the real config implementation, loader adapter,
native-settings accessor and refusal-trace accessor: 48 configurations cover
eight flag combinations at each of three probe modes and both trace values.
They check immutable copies, invalid size/value/reinstallation, missing ABI,
zero helper getenv calls and unchanged engine reads. A separate CTest checks
that the actual libr3 binary exports both configuration functions without loading HIP. Manifest tests cover
defaults, explicit false values, diagnostic settings and ranges. TOML scalar
conversion preserves boolean types so overrides satisfy the schema.

The patch stays untested pending hardware identity evidence. Card-free tests
do not establish numerical identity against main.
