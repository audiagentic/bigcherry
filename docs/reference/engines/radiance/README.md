# Radiance

The released **bc-radiance 2020.x** line (see the release list for the current one) incorporates upstream Radiance
**v2.3.0**. Older v2.2.0 and v1.17.0 measurements remain historical comparison evidence. The custom build uses RDNA2 and RDNA3 helpers alongside an RDNA4 engine GPU, with
expert placement adapted to differing GPU memory and bandwidth.

## Patches and extensions at a glance

BigCherry maintains **43 current engine patch packages** and **7 components** for Radiance.
Each description says what the package is for, not that it is enabled or
performance-qualified. **Validated** means package-level evidence exists;
**untested** means qualification is not established; **diagnostic** is for
observability rather than production; **superseded** means a
newer patch replaces it. See the [full catalogue](ENHANCEMENTS.md) and
linked package pages for technical details and evidence.

### Engine patches

The [generated catalogue](ENHANCEMENTS.md) lists every current package, including its state, scope, and evidence links.

### Archived patches

Retired patches are kept unchanged as evidence under [engines/radiance/patches/archive](../../../../engines/radiance/patches/archive/ARCHIVE.md) (reason, successor and evidence per patch in ARCHIVE.md); they are never built or composed.

| Package | What it was for | Final state |
|---|---|---|
| [rad0001_drain_mover_before_arena_free](../../../../engines/radiance/patches/archive/rad0001_drain_mover_before_arena_free/SUMMARY.md) | Drain GPU transfers before freeing memory | `superseded` |
| [rad0003_mtp_full_head_switch](../../../../engines/radiance/patches/archive/rad0003_mtp_full_head_switch/SUMMARY.md) | Optionally draft with the full model head | `rejected` |
| [rad0006_heat_freeze_diagnostic](../../../../engines/radiance/patches/archive/rad0006_heat_freeze_diagnostic/SUMMARY.md) | Freeze expert movement for diagnosis | `superseded` |
| [rad0007_hc_pf2_fp8_down](../../../../engines/radiance/patches/archive/rad0007_hc_pf2_fp8_down/SUMMARY.md) | Unused experimental FP8 HyperConn kernels | `rejected` |
| [rad0034_prefill_concurrency](../../../../engines/radiance/patches/archive/rad0034_prefill_concurrency/SUMMARY.md) | Earlier long-prefill concurrency limit | `superseded` |
| [rad0035_prefill_order](../../../../engines/radiance/patches/archive/rad0035_prefill_order/SUMMARY.md) | Earlier prefill ordering and ageing controls | `superseded` |
| [rad0039_moe_heat_observer](../../../../engines/radiance/patches/archive/rad0039_moe_heat_observer/SUMMARY.md) | Observe expert usage without changing placement; inactive | `untested` |

### Components (extensions)

Components are BigCherry-owned additions built alongside Radiance.

| Component | Simple purpose |
| --- | --- |
| [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) | Selected RDNA4-optimised kernels |
| [bc-iqr](../../../../engines/radiance/components/bc-iqr/README.md) | Owned IQR kernels for RDNA3 and RDNA4, listed ahead of the base libraries |
| [libr2](../../../../engines/radiance/components/libr2/README.md) | RDNA2 expert computation and quantisation |
| [libr3](../../../../engines/radiance/components/libr3/README.md) | RDNA3 matrix, attention and reduction kernels |
| [moe-remote](../../../../engines/radiance/components/moe-remote/README.md) | Runs selected experts on helper GPUs |
| [ngram-helper](../../../../engines/radiance/components/ngram-helper/README.md) | Optional n-gram work on a spare GPU; off by default |
| [serving](../../../../engines/radiance/components/serving/README.md) | Serving configuration validation and manifests |

Individual component settings have their own qualification states. The historical v1.17 MTP replay override is archived; Radiance 2.2 owns
fixed-depth draft, verify and decode pass-key replay.
Mixed-GPU remote-MoE layouts lack matched current-pin end-to-end throughput
and quality evidence here. Full manifest-driven serving startup remains under
qualification.

## Fixed-depth Flash-Next scheduler and replay authority (RUM08)

For the released `flashnext-4gpu` profile the single effective step-timing
authority is the upstream Radiance scheduler, with explicitly selected
`prefill-interleave: 0` and `mixed-step-target-ms: 0`. It runs fixed-depth
MTP (`num-speculative-tokens: 3`). Admission uses the default **arrival**
order; there is no active `prefill-governor` override or second adaptive
interleave/depth controller in this profile. `rad0036` remains composed as
an independently useful **optional long-prefill admission** policy, but
its budget/age/remaining modes are not active for `flashnext-4gpu`.
The archived `rad0013` and `rad0047` are not composed: upstream
`Qwen4exp::pass_key`, `Ctx::pass_key` and the keyed-context issue path
own fixed-shape decode/verify/MTP replay.

The long-Qwen profiles explicitly choose `rad0036` budget/age admission
and `prefill-interleave: 2` for *separate* decode/prefill timing. This
does not constitute two admission owners, and their long-profile behavior
is not qualified by the Flash-Next fixed-depth replay check. New adaptive,
suffix and traversal proposals are outside this profile.

See [RUM08 completion evidence](../../../evidence/2026-11-22-rum08-completion/README.md)
for the upstream key/boundary tests, installed-release hardware trace
and any remaining acceptance gaps. HIP phase traces observe pass kinds
and timings, not replay hashes or actual `RadArgs`; evidence must not
claim argument equality from trace rows alone.

## Current-pin measured prefill

The paired RUM09 release gate measured engine-side prefill on the
**Radiance v2.2.0** Flash-Next four-GPU layout (R9700, two RX 7900 XTX
and RX 6900 XT). Both arms used the same prompts, seed, topology,
262,144-token context, eight sessions and stock power profile.

| Workload | Baseline: v1.17.0 tokens/s | Best recorded: v2.2.0 tokens/s |
| --- | ---: | ---: |
| Document prefill | 3,254 | **3,662** |
| Long-7 prefill | 2,207 | **2,453** |
| Long-8 prefill | 2,099 | **2,430** |

The [paired RUM09 gate](../../../evidence/radiance-latest-bump/RUM09-GATE-D.md)
used two launches per arm in November 2026. These are BigCherry
cross-version measurem