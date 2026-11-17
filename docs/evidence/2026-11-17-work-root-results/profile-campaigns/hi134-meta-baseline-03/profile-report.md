# Profile campaign: hi134-meta-baseline-03

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1,3` (expected GPU count: 3)
- runtime profile: `production-dual-xtx`
- workload: `hi134-meta-baseline-013-control`
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
- environment stable: **False** -- control block means spread 18.7% across the run (threshold 5%) -- profiling artifacts are still real and usable diagnostically, but no comparative performance conclusion should be drawn from this run alone

## Controls (unprofiled)

| block | reps | tg mean t/s | tg stddev |
|---|---|---|---|
| A | 10 | 28.51 | 5.05 |
| B | 10 | 31.21 | 3.11 |
| C | 10 | 28.42 | 3.89 |
| D | 10 | 27.78 | 4.19 |
| E | 10 | 25.91 | 5.00 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 4 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 45450 | 1835899.6 | 40.39 | 74.40 | 40 | 128 | 0 |
| `__amd_rocclr_copyBufferRectAligned` | 258 | 865287.1 | 3353.83 | 9493.47 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 2532 | 330190.2 | 130.41 | 574.89 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 54366 | 127746.6 | 2.35 | 3.24 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 3000 | 111272.5 | 37.09 | 68.68 | 24 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 32844 | 108991.3 | 3.32 | 4.44 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 15453 | 88986.6 | 5.76 | 7.32 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 26571 | 74800.4 | 2.82 | 4.40 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 1512 | 66966.3 | 44.29 | 78.84 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 1512 | 55595.9 | 36.77 | 71.04 | 40 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 4896 | 45368.7 | 9.27 | 11.24 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 4896 | 43337.4 | 8.85 | 11.04 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 360 | 34161.3 | 94.89 | 129.80 | 24 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 6174 | 30201.6 | 4.89 | 7.36 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 9792 | 28886.2 | 2.95 | 4.04 | 8 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 360 | 26128.2 | 72.58 | 89.44 | 144 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 9078 | 24908.6 | 2.74 | 3.56 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 11523 | 24695.8 | 2.14 | 3.28 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 4182 | 24524.9 | 5.86 | 9.76 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 4896 | 22195.4 | 4.53 | 5.44 | 24 | 128 | 0 |

## GPU profile: 2

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 4 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 43935 | 1773638.5 | 40.37 | 74.32 | 40 | 128 | 0 |
| `__amd_rocclr_copyBufferRectAligned` | 257 | 864702.4 | 3364.60 | 9494.90 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 3573 | 353266.6 | 98.87 | 570.01 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 54075 | 128147.9 | 2.37 | 3.24 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 3000 | 111294.3 | 37.10 | 68.80 | 24 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 32796 | 108749.3 | 3.32 | 4.44 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 15393 | 89232.4 | 5.80 | 7.40 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 26559 | 74694.9 | 2.81 | 4.40 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 1512 | 66874.2 | 44.23 | 78.96 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 1512 | 55554.6 | 36.74 | 71.04 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 543 | 51714.0 | 95.24 | 131.20 | 24 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 4896 | 43558.7 | 8.90 | 11.16 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 4752 | 42113.4 | 8.86 | 11.04 | 16 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 6102 | 29624.8 | 4.85 | 7.16 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 9792 | 28813.9 | 2.94 | 4.08 | 8 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 11328 | 25600.2 | 2.26 | 3.44 | 16 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 351 | 25484.7 | 72.61 | 89.20 | 144 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 9054 | 24698.9 | 2.73 | 3.64 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 4158 | 24355.3 | 5.86 | 9.64 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 4896 | 21847.2 | 4.46 | 5.56 | 24 | 128 | 0 |
