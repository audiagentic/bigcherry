# Radiance

Radiance **v1.16.0** is BigCherry's primary upstream target. The custom
build uses RDNA2 and RDNA3 helpers alongside an RDNA4 engine GPU, with
expert placement adapted to differing GPU memory and bandwidth.

## Main additions and patches

| Area | BigCherry change | Owning code or patches |
| --- | --- | --- |
| AMD kernels | RDNA3 attention and GEMMs; RDNA2 expert-kernel subset; selected RDNA4 overrides | [libr3](../../../../engines/radiance/components/libr3/README.md), [libr2](../../../../engines/radiance/components/libr2/README.md), [bc-gfx12](../../../../engines/radiance/components/bc-gfx12/README.md) |
| Remote MoE | Routed experts on helper GPUs, pinned-host mailbox, heat-ranked placement and GPU-side admission/owner generations | [moe-remote](../../../../engines/radiance/components/moe-remote/README.md), [remote hand-off patch](../../../../engines/radiance/patches/rad0002_moe_remote_handoff/SUMMARY.md) |
| HyperConn tuning | Selectable PF2 up-mix launch shape on RDNA4 | [up-mix patch](../../../../engines/radiance/patches/rad0008_hc_upmix_rt_sweep/SUMMARY.md) |
| Serving configuration | Strict TOML loader, manifest reader and engine memory plan; full manifest-driven process startup remains incomplete | [serving](../../../../engines/radiance/components/serving/README.md), [manifest patch](../../../../engines/radiance/patches/rad0009_serving_manifest/SUMMARY.md), [memory-plan patch](../../../../engines/radiance/patches/rad0011_plan_memory/SUMMARY.md) |

There are **10 engine patch packages** and **6 components**. The hand-off
and up-mix patches are recorded as validated; other patches include
diagnostic and unfinished work. See the [full catalogue](ENHANCEMENTS.md)
for states and evidence. Recipe inclusion is not proof of qualification.

## Current-pin throughput

The pinned upstream is Radiance **v1.16.0**. No qualified fresh prefill
tokens/s or aggregate multi-stream client output or engine decode tokens/s
have yet been recorded for this pin. The v1.16.0 release contains a
one- and six-stream latency comparison with v1.14.0, not a matched
unmodified-stock throughput A/B. Do not treat the earlier layout results
as current-pin speedups.

| Model and layout | Prefill tokens/s | Aggregate output tokens/s |
| --- | ---: | ---: |
| [Flash-Next](models/qwen3.8-flash-next.md), R9700 + two RX 7900 XTX + RX 6900 XT | Not qualified | Not qualified |
| [Qwen3.8-27B MXFP4](models/qwen3.8-27b.md), R9700; two-XTX alternative unqualified | Not qualified | Not qualified |
| [Qwen3.6-35B-A3B FP8](models/qwen3.6-35b-a3b.md), R9700 | Not qualified | Not qualified |

Earlier v1.14.0 results, including long-context and concurrent serving
measurements, are retained in the [archive](../../../archive/radiance/v1.14-model-results/).
The Flash-Next four-card layout has a documented quality trade-off.

## Further detail

The [build and serving runbook](RUNBOOK.md) covers composition and
measurement; [GPU layouts](LAYOUTS.md) covers helper card budgets.
Manifest startup is still being integrated against the
[v1 specification](../../../design/serving-manifest-v1/spec.md).
