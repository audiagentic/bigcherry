# Retained KV and prefill evidence for Radiance 2.2

The validated production packages `rad0048` through `rad0054` carry forward
the behavior of the shipped cache and prefill implementations. Their promotion
records bind the current package digests and the Radiance 2.2.0 pin
`b6b8751cccb0e834a13ae9e5afca824ee9a487aa`. Each records the historical
behavior and the focused compatibility checks used for the port.

| Packages | Historical behavior | Port check |
| --- | --- | --- |
| `rad0048`, `rad0050` | Validated [rad0040](../../../engines/radiance/patches/rad0040_kv_layout_table/evidence/promotion.json), [rad0037](../../../engines/radiance/patches/rad0037_bf16_fp8_kv/evidence/promotion.json) and [rad0038](../../../engines/radiance/patches/rad0038_bf16_i8_kv/evidence/promotion.json) cover selection, sizing, identity and B16 A/B. | ABI44 layout and fingerprint, distinct byte extents, CLI/YAML selection, allocation and binding, and packed-stream resident fallback. |
| `rad0049` | The old [Flash-Next FP8 control and subject](../2026-11-16-vtp44-generic-kv-layouts/IDENTITY.md) exercised the deployed E4M3 cache. The [historical TP4 attachment](https://codeberg.org/StillDeadcode/radiance/commit/7cf4c005e18aedee67f2667899f264e1be803b8c) supplies the GQA6/page4 row provenance. | Both exact Qwen4exp GQA12/GQA6 page4 attachment paths and native E4M3 store/reader declarations. |
| `rad0051`, `rad0052` | The [gfx1201 device A/B](../2026-11-16-vtp44-generic-kv-layouts/DEVICE-AB.md), [gfx1100 direct selftests](../2026-11-16-vtp44-generic-kv-layouts/rad0042-selftest/README.md), and [Flash-Next TP2 A/B](../2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-AB.md) cover B16 kernels and geometry. | Typed store/reader declarations, exact rows, shared trunk/MTP KV, independent QSA, and the separate native E4M3 DFlash2 cache. |
| `rad0053` | The historical [codec record](https://codeberg.org/StillDeadcode/radiance/commit/c00a650bce641073000068f5b66e594806677712), [27B layout/KL comparison](https://codeberg.org/StillDeadcode/radiance/commit/1823ccd5e9b5e77ebfaf20a37c8329e76a50f725) and [gfx1100 device rows](../2026-11-15-rad53-fp8-prefill-attention/README.md) cover V4 and K6. | Old/new codec source comparison and reference vectors for scale rounding, clamps, nonfinite handling and K6 bit positions. |
| `rad0054` | The shipped lineage records old [E4M3 off versus combined](https://codeberg.org/StillDeadcode/radiance/commit/cdc0ae26df0e87b69bfbae3ae7bdff31be93c001) and [V4 off versus combined](https://codeberg.org/StillDeadcode/radiance/commit/51db508f91f08d6f7bf6959a76126e0304b34e71) tests. The [gfx1201 baseline registry](../2026-11-16-vtp44-generic-kv-layouts/baseline-libr4d-gfx1201-catalogue.txt) lists all nine E4M3/V4/K6V4 QK, PV and combined rows. | Separate CLI/YAML mode parsing, exact page16 row constraints, no prefill selector in decode/DFlash2, and refusal of unsupported packed and page4 combinations. |

The [RUM07 validation record](../../planning/completed/patching-radiance-upstream/RUM07.md#validation)
contains the earlier ABI44 host, reference and object-build details. The
v2.2 ABI44 recheck passed 23 checks over six layout cases and 3,409 checks
over 78 config cases. The current production rebase returned 32 of 32 clean
at the exact v2.2 revision. The no-compile replay fixture applied the seven
packages and confirmed byte-identical reapplication. All seven promotion
records pass G4 with legacy grandfathering disabled.

The historical single-mode coverage differs by cache and geometry. The
old prefill KL reports cover E4M3 and V4 combined mode, and the registry
identifies the other selectable rows. None of this records a new device
measurement for every v2.2 row. Integrated deployment KL, lifecycle and
paired parity remain in [RUM09](../../planning/active/patching-radiance-upstream/RUM09.md).
