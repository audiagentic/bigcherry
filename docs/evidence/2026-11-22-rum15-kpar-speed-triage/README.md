# RUM15 compact IQ8R speed triage — 2026-11-22

R9700 gfx1201; native source-built diagnostic libraries; fixed Qwen IQ8R fixture. Timings are stock/candidate/stock A/B/A, 54 matching IQ8R cases, warm weights; positive delta means slower. All candidates passed existing native 35/35 exact-F32-oracle selftests. Timing-only kbench does not itself check numerics.

| Prototype vs production BF16 | Median delta | M33 N1024 K5120 |
| --- | ---: | ---: |
| Packed BF16 row gain in tiled kernel | -0.99% | +31.94% |
| Once-converted shared FP32 row gain | -1.09% | +28.91% |
| Prefetched folded FP32 LDS scales | -0.95% | +28.80% |
| K-parallel invariant gain hoist | +2.26% | +28.56% |
| K-parallel 8-wave allocation | +5.96% | +22.89% |
| K-parallel early U8 prefetch | +3.23% | up to +130.78% |
| Forced alternative narrow/split-K | +8.57% | +183.39% |

The K-parallel dispatch is selected for M=8 and M=33; edits only to the prefill tile cannot directly remove those narrow-shape regressions. K-parallel prefetch showed an extreme outlier; instruction/resource profiling is needed before attributing its cause.

An identical-binary hoist A/A/A control at M8 N12288 K5120 measured 35.91/35.97/35.90 us; the same binary across two library homes measured 35.94/35.91/35.91 us. Library-home identity alone does not explain the older +22.25% M8 cross-build observation; compare code generation and repeated order-balanced runs.

All prototypes remain outside managed patches and serving. None closes the worst-case regression without causing other losses. Next: profile K-parallel ISA/registers/stalls for M33 and M8, capture actual LTX TP1 matrix shapes and compare on both gfx1201/gfx1100 with warm and rotating weights, then validate cache/layout/admission before production selection. No new kernel family, shape workaround, or precision regression was accepted.
