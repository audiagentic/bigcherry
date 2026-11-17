# Profile campaign: hi134-meta-baseline-01

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1,2` (expected GPU count: 3)
- runtime profile: `production-dual-xtx`
- workload: `hi134-meta-baseline-3gpu`
- rocprofv3: `version: 1.1.0
        git_revision: 97f5574fe2fdc7bef44fb01545347912ee9f1779
        library_arch: x86_64-linux-gnu
         system_name: Linux
    system_processor: x86_64
      system_version: 6.8.0-85-generic
         compiler_id: GNU
    compiler_version: 13.3.0
        rocm_version: 7.2.4`
- control reps: 10, profile passes: 2
- environment stable: **False** -- control block means spread 13.6% across the run (threshold 5%) -- profiling artifacts are still real and usable diagnostically, but no comparative performance conclusion should be drawn from this run alone

## Controls (unprofiled)

| block | reps | tg mean t/s | tg stddev |
|---|---|---|---|
| A | 10 | 36.01 | 5.86 |
| B | 10 | 34.61 | 5.29 |
| C | 10 | 38.79 | 7.16 |
| D | 10 | 33.89 | 3.15 |
| E | 10 | 36.36 | 6.17 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 3 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: True

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 39390 | 1208542.0 | 30.68 | 61.40 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 2196 | 243476.0 | 110.87 | 512.25 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 3000 | 96490.9 | 32.16 | 61.24 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 47922 | 67319.3 | 1.40 | 1.96 | 16 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 28956 | 52174.9 | 1.80 | 2.32 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 13605 | 49405.2 | 3.63 | 4.40 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 1512 | 48415.4 | 32.02 | 63.48 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 1512 | 44668.5 | 29.54 | 60.64 | 40 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 23439 | 38947.2 | 1.66 | 2.32 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 312 | 25808.9 | 82.72 | 101.32 | 24 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 4320 | 22844.8 | 5.29 | 7.64 | 40 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 5754 | 21323.6 | 3.71 | 7.76 | 16 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 4320 | 20003.3 | 4.63 | 5.64 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 8640 | 15739.8 | 1.82 | 2.44 | 8 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 7998 | 14323.6 | 1.79 | 2.28 | 16 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 312 | 14121.8 | 45.26 | 52.36 | 144 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 4320 | 13221.2 | 3.06 | 3.72 | 24 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 10167 | 13033.0 | 1.28 | 1.92 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 3678 | 12356.3 | 3.36 | 6.04 | 24 | 128 | 0 |
| `__amd_rocclr_fillBufferAligned` | 935 | 10711.7 | 11.46 | 11.52 | 16 | 128 | 0 |

## GPU profile: 2

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 3 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: True

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 54540 | 1956700.6 | 35.88 | 61.36 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 4245 | 398806.4 | 93.95 | 570.77 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 66963 | 152485.6 | 2.28 | 3.16 | 16 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 40572 | 124030.3 | 3.06 | 4.24 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 19089 | 105994.6 | 5.55 | 7.24 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 3027 | 103345.7 | 34.14 | 60.36 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 3000 | 97399.3 | 32.47 | 60.28 | 24 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 32823 | 88824.0 | 2.71 | 4.12 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 1512 | 61268.8 | 40.52 | 65.84 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 639 | 54961.3 | 86.01 | 103.32 | 24 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 6048 | 48298.2 | 7.99 | 10.44 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 5904 | 47258.4 | 8.00 | 11.00 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 12096 | 34836.8 | 2.88 | 3.96 | 8 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 6942 | 32532.1 | 4.69 | 6.16 | 16 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 447 | 30546.1 | 68.34 | 89.60 | 144 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 11214 | 29197.0 | 2.60 | 3.60 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 14040 | 28620.8 | 2.04 | 3.16 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 5166 | 26954.5 | 5.22 | 9.32 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 6048 | 24140.1 | 3.99 | 5.16 | 24 | 128 | 0 |
| `void ssm_conv_f32<true, 128ul, 4ul>(float const*, float const*, float const*, in` | 6048 | 20424.4 | 3.38 | 4.60 | 24 | 128 | 0 |
