# IQ6R-V4 / RDNA3 (gfx1100) compatibility: card-free preflight

Date: 2026-11-16. **Result: CARD-FREE SOURCE AND BINARY CONTRACT PASS. DEVICE INFERENCE UNQUALIFIED.**

## Exact artifact inspected

- Pinned managed engine: `bc-radiance-1170.26.0`, source `${BC_RADIANCE_SOURCE}` (pinned source revision `20bfb8913e26bc9c2703f4fdf96e0b08`).
- The corresponding *built*, managed RDNA3 plugin: `${BC_RELEASE_BUILD}/radiance_home/kernels/libr3.so` (`BC_RELEASE_BUILD` is the `build_dir` from the managed `install.json`)
- Plugin SHA-256: `109ff26b6e4416db00e9213694816d5423917e051532c381c7140cd67acc92de`.
- Persistent scratch receipt (not a production artifact): `${BC_WORK_ROOT}/iq6r-v4-qualification/rdna3-offline-preflight.json`.

## Actual card-free checks

1. `r3_rewrite.py --check` against the pinned source with both the tracked textual and native gfx1100 rules: `{"failures": {}}`, exit 0.
2. `llvm-objdump --offloading` of the managed `libr3.so`: **gfx1100 HIP image present**.
3. `nm -D` of the same binary: all **20 required exported IQR entry points** present (trunk narrow/tiled/gated GEMMs and scratch hooks, vocabulary/class head, logits, embedding, layout/relayout/unrelayout hooks). They are outside the plugin's stub range; in this build `r3_stub_begin == r3_stub_end`.
4. Pinned `r4d_rows.cpp`: all **seven required upstream IQR row definitions** are present; `r4d_gemm_iqr` is not in `libr3/exclude.txt`.
5. `tools/lab/radiance/iq6r_v4_rdna3_preflight.py` reproduces checks 2-4 without GPU execution and writes its hash/identity receipt. Six card-free tests cover missing image/hook/row, bad stub range, excluded unit and passing input.

**Limitation:** the source-row definitions and exported symbols are necessary but not sufficient to prove that the compiled `libr3` runtime registry will select each row on the real IQ6R-V4 shapes. This check neither loads the plugin into an engine nor launches any HIP kernels.

Historical RDNA3 IQ6R B=6 GEMM numerical and crossover results are recorded under archived VTP09 (both XTX cards, including model-sized N/K); those experiments used a different rebuilt plugin and are **not** a substitute for tests on the current plugin and new full RAD2 model.

## Hardware execution gate — pending an exclusive GPU lease

After the full IQ6R artifact is sealed and checksum-verified, run in the order below, with separate receipts for **physical XTX 0 and 1**. The conversion lease currently belongs to the R9700 job; no card use occurred during this preflight.

| Gate | Mandatory check | Stop condition |
| --- | --- | --- |
| D0 | Verify device identity and gfx1100 offload image of *the exact tested plugin*, explicitly set `HIP_VISIBLE_DEVICES=0` or `=1`, confirm visible HIP ordinal 0 reports gfx1100; record executable/plugin hashes, ROCm version and lease | Any wrong device or image blocks the entire run |
| D1 | Build/run matching pinned `r3_selftest` with managed `libr3.so` and `libref.so` side by side. Exercise `gemm_iqr_nt_m64`, `gemm_iqr_tiled`, `gemm_iqr_gated`, `embed_lookup_iqr`, `logits_gemm_iqr`, `logits_gemm_vocab_iqr`, `logit_rerank_iqr` as supported by the row-specific smoke harness | Any numeric mismatch, failure, implicit fallback or segfault blocks the model |
| D2 | Explicit u4/u5/u6/u7 trellis and RTN embedding encoding verification; rows M=1,4,16,17,32,48,63,64,65,128,256,512,1024,2048 and Qwen27 shapes N=5120/10240 K=5120 wherever legal | Any incorrect decoding/dispatch, untested unsupported shape, silent fallback blocks |
| D3 | Isolated loading of the final fully converted RAD under `libr3` on each XTX, short and long context, draft off. Verify actual resolved kernel names, logits versus BF16 reference, GDN/KV and VRAM | Any missing kernel, memory failure, bad logits or unplanned CPU/libref fallback blocks |
| D4 | Mixed gfx1100 + gfx1100 + gfx1201 TP3, exact rank/manifest agreement, host collective, quality, memory, concurrency and matched W8A8 ABBA throughput | No promotion until all independent Q0–Q9 gates pass |

**Not done:** any GPU numeric selftest for the current managed plugin, full-model IQ6R on either XTX, TP3 load, quality or throughput. Production W8A8 remains unchanged.

## Compiled ABI registry inspection (additional card-free gate)

A second, stronger preflight compiled `tools/lab/radiance/iq6r_v4_rdna3_registry.cpp` against the pinned `abi/rad_abi.h` and used `dlopen`/`rad_kernel_count`/`rad_kernel_at` on the exact managed `libr3.so`, **without** calling `rad_plugin_open`, any HIP runtime functions or device kernels. Result: **587 live rows**, with all **7/7 required IQR rows** present, each with non-null launch, layout and relayout hooks. The rows and priorities were:

| Row | Op | Priority |
| --- | --- | ---: |
| `embed_lookup_iqr` | `embed_lookup` | 50 |
| `logits_gemm_vocab_iqr` | `logits_gemm_vocab` | 60 |
| `logits_gemm_iqr` | `logits_gemm` | 60 |
| `gemm_iqr_nt_m64` | `gemm_nt_q` | 60 |
| `gemm_iqr_gated` | `gemm_nt_q_gated` | 50 |
| `gemm_iqr_tiled` | `gemm_nt_q` | 40 |
| `logit_rerank_iqr` | `logit_rerank` | 50 |

Output receipt: `${BC_WORK_ROOT}/iq6r-v4-qualification/rdna3-registry-audit.json`. This closes the uncertainty about *live compiled registry entries*; it does not prove a selector chooses them for a particular model shape, GPU numerical correctness or model inference. Device tests D0–D4 remain mandatory.
