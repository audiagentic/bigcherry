# Profile campaign: hi134-stage-attribution-01

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1,3` (expected GPU count: 3)
- runtime profile: `production-dual-xtx`
- workload: `hi134-stage-attribution`
- rocprofv3: `version: 1.1.0
        git_revision: 97f5574fe2fdc7bef44fb01545347912ee9f1779
        library_arch: x86_64-linux-gnu
         system_name: Linux
    system_processor: x86_64
      system_version: 6.8.0-85-generic
         compiler_id: GNU
    compiler_version: 13.3.0
        rocm_version: 7.2.4`
- control reps: 3, profile passes: 1
- environment stable: **False** -- control block means spread 29.3% across the run (threshold 5%) -- profiling artifacts are still real and usable diagnostically, but no comparative performance conclusion should be drawn from this run alone

## Controls (unprofiled)

| block | reps | tg mean t/s | tg stddev |
|---|---|---|---|
| A | 3 | 30.30 | 7.65 |
| B | 3 | 29.88 | 3.05 |
| C | 3 | 34.93 | 3.11 |
| D | 3 | 26.87 | 7.01 |
| E | 3 | 26.24 | 3.22 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 4 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 33330 | 1346767.2 | 40.41 | 74.44 | 40 | 128 | 0 |
| `__amd_rocclr_copyBufferRectAligned` | 250 | 860255.6 | 3441.02 | 9494.31 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 1860 | 244469.8 | 131.44 | 824.84 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 3000 | 111239.3 | 37.08 | 68.76 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 41478 | 98766.3 | 2.38 | 3.28 | 16 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 25068 | 83588.1 | 3.33 | 4.44 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 11757 | 68013.1 | 5.78 | 7.36 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 1512 | 65865.6 | 43.56 | 78.96 | 48 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 20307 | 57744.0 | 2.84 | 4.40 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 1512 | 55266.6 | 36.55 | 71.12 | 40 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 3744 | 34939.9 | 9.33 | 11.36 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 3744 | 33318.0 | 8.90 | 11.08 | 16 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 5334 | 25810.3 | 4.84 | 7.40 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 264 | 25065.6 | 94.95 | 129.40 | 24 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 7488 | 22307.7 | 2.98 | 4.08 | 8 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 8811 | 19416.2 | 2.20 | 3.32 | 16 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 6918 | 19243.9 | 2.78 | 3.68 | 16 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 264 | 19177.3 | 72.64 | 89.56 | 144 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 3174 | 18625.4 | 5.87 | 9.76 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 3744 | 17053.4 | 4.55 | 5.64 | 24 | 128 | 0 |
