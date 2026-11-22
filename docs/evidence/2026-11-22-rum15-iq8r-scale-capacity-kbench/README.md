# RUM15: checkpoint-scale capacity and native GEMM A/B timing

**22 November 2026. Read-only diagnostics on the merged diagnostic candidate from PR #1351.** The FP32 scale format is numerically validated in C06 but **not admitted to production**. This measurement narrows the RUM15 memory/performance gate; it does not close model-level admission, TP1 parity, reload/refill or cold-streaming benchmarks.

## Checkpoint-derived memory accounting

Read the header, entry and plane tables of the actual LTX-2.5 IQ8R RAD2 checkpoint using the existing \`SelectedImageRows\` metadata parser (\`tools/tests/lab/test_ltx_iq8r_numeric.py\`). **No tensor payloads were read.** Validated for every one of the **1,306** 2-D IQ8R matrices:

- K width is a multiple of the existing 256-element group size;
- original U8 \`scale\` plane has exactly \`N × K/256\` bytes;
- original BF16 \`scale.1\` plane has exactly \`2 × N\` bytes.

The source checkpoint is **39,578,374,144 bytes** (decimal **39.58 GB**). It declares **127,309,824** group scales across the entire model.

| Scale representation | Group-scale plane bytes | Binary MiB |
| --- | ---: | ---: |
| Old folded BF16, 2 bytes/group | 254,619,648 | 242.8242 |
| New folded FP32, 4 bytes/group | 509,239,296 | 485.6484 |
| **Additional scale storage if all resident** | **254,619,648** | **242.8242** |

Across the 960 DiT IQ8R matrices, the potential extra scale storage is **138.0000 MiB**. The other 346 IQ8R matrices contribute **104.8242 MiB**. These are scale-plane bytes only, **not actual per-device allocation**, which also depends on sharding, replication, tiers, allocator alignment and overlap.

The alternate representation preserving the original U8 group scale and BF16 row gain has a **source-plane lower bound** of 142,125,056 bytes (135.541 MiB) across all IQ8R matrices. This is **not a measured GPU layout** or implementable design recommendation: padding, access patterns and arithmetic must be evaluated.

### Per-rank C06 self-attention projection scale delta

For the already tested 48-block video and audio C06 self-attention geometry, calculate each rank's group count directly from the checkpoint's block-0 \`attn1\` and \`audio_attn1\` QKV/output shapes. QKV partitions the 3 projection banks into head-aligned **output-row** shards; output projection partitions the **K columns**. Retain full 256-K groups and scale the block-0 delta by the checkpoint's 48 corresponding blocks.

| Head split | Rank 0 | Rank 1 | Rank 2 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| 8/8/16 | 7.50 MiB | 7.50 MiB | 15.00 MiB | 30.00 MiB |
| 12/12/8 | 11.25 MiB | 11.25 MiB | 7.50 MiB | 30.00 MiB |

One-block increments for split 8/8/16 are **160/160/320 KiB**; for 12/12/8 they are **240/240/160 KiB**. These figures cover **only** the video/audio C06 IQ8R QKV and output weights, not cross-attention, FFN, conditioning, text encoder, VAE, any replicated weights, staging or activations. **Do not use this table as a total TP3 VRAM admission calculation.**

## Same-device, same-fixture kernel timing

The unmodified installed Radiance \`rad-kbench\` performed **stock → candidate FP32 → stock** passes under exclusive GPU leases, separately on R9700 (gfx1201, \`libr4d\`) and XTX (gfx1100, \`libr3\`). Both native libraries use the same pinned ABI and reference plugin. Every run used the same shipped \`kernels.rkb\` fixture (SHA256 \`1ef17fb9cf0a5401f8426b110fcc665c9a70ef6c568b7cf01e8141a17c6a92dd\`), recorded **2026-11-14 for Qwen3.8 27B IQ8R**, **not** the LTX2.5 checkpoint.

- Filter the \`gemm_nt_q\` **operation** (not the kernel-row name); match the **54 IQ8R rows** out of 77 native device-timed cases using the existing \`tools/lab/radiance/kbench_step.py::report_rows\` reader.
- \`--timing-only --max-cases 1 --max-bytes 256 --bench-warm-ms 750 --bench-reps 10 --bench-flush 0\`. A/B/A is interleaved per card, and each ratio compares the FP32 time with the mean of the two stock times for the **identical shape and kernel row**.
- Warm-cache benchmarking (\`bench-flush=0\`), **not cold weight streaming** or model video-generation latency. \`--timing-only\` deliberately makes no correctness claim; that proof is the separate full native C06 and selftest receipts in \`docs/evidence/2026-11-22-rum15-f32-native-candidate/README.md\`.
- Stock/candidate R9700 plugin SHA256: \`f34fa8352c146aeb89ec4f94e990b008a16ab6c2ffb85a55acb22ee5fa51eb39\` / \`9c38079be3b0fb9f316ee5ba364ea402c6cd160c9bc67e44cfd59b26f6f1f906\`.
- Stock/candidate XTX plugin SHA256: \`371fcb408e78e9af18b50e769ce02c393e499331f65fa0a35ba59b20b6ee6436\` / \`e6cd43e1363327c10643953ad5a9fe8a9c5e6be151e9151f8ad9fb5b60953e18\`.
- No layout or plugin was installed into a production prefix; the native candidates were loaded from isolated build outputs. Both exclusive GPU leases were released.

| Metric, 54 matched IQ8R cases | R9700 FP32 vs stock | XTX FP32 vs stock |
| --- | ---: | ---: |
| Median timing change (negative = FP32 faster) | **−0.59%** | **−0.61%** |
| Median change, M ≤ 16 | −0.80% | +0.38% |
| Median change, 17 ≤ M ≤ 64 | +0.87% | −0.42% |
| Median change, M ≥ 65 | −0.82% | −1.72% |
| Median absolute A/A-repeat stock drift | 0.21% | 0.81% |
| Larger isolated slowdowns | **+24.89%**, see below | **+13.71%**, see below |

**Important regression candidates:**

- **R9700, M=33, N=1024, K=5120, group=256, \`gemm_iq8r_nt_m16\`:** FP32 is **+25.35%** in the initial A/B/A run and **+24.89%** in an independently repeated warmer A/B/A run. Stock first/last drift was only **0.35%** and **0.53%** respectively. This is a **repeatable narrow-GEMM performance regression** in the Qwen fixture and should block unconditional production selection until its cost and shape applicability are resolved.
- R9700, M=33, N=12288, K=5120 had an apparent +31.87% first-run slowdown but **55.12% A/A stock drift**, so that observation is **not trustworthy** without isolated confirmation.
- **XTX, M=18 and M=32, N=12288, K=5120:** FP32 was **+13.71%** and **+13.47%** respectively, with stock drift ~0.5%. These were each measured in one A/B/A pass and need repetition before attribution to scale loads, occupancy or register pressure.

An aggregate median improvement does **not** cancel these row-specific regressions, nor establish whole-model video throughput parity. The Qwen benchmark has different weight shapes and operating context than LTX TP3.

## Next RUM15 decision

**Do not promote** the FP32 layout yet. First isolate the repeatable R9700 M=33 small-N slow path in the existing K-parallel/narrow GEMM (inspect register pressure, scale-load path and wave occupancy), check representative **LTX-derived shapes** at M=1/16/64, and repeat the XTX outliers under controlled clocks/cold reads. Compare with a source-factor U8 + BF16 representation only if it reduces both memory overhead and performance cost **without reintroducing scale rounding or a duplicate kernel family**.

Separately measure real rank residency, RUM14 load/reload/refill tiers, old-layout plan rejection and 16/64-row TP1 captured-input numerical parity. The source layout tag is changed by the diagnostic patch, but that alone is not evidence that all stored plans and caches refuse stale buffers.

**Conclusion:** The FP32 format fixes C06 QKV numerics on both architectures and looks broadly competitive in a warm synthetic kernel benchmark. **Its worst-case performance and actual model-level admission remain open.**
