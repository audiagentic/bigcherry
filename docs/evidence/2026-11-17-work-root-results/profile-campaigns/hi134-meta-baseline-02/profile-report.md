# Profile campaign: hi134-meta-baseline-02

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1,2,3` (expected GPU count: 4)
- runtime profile: `production-dual-xtx`
- workload: `hi134-meta-baseline-4gpu`
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
- environment stable: **False** -- control block means spread 12.9% across the run (threshold 5%) -- profiling artifacts are still real and usable diagnostically, but no comparative performance conclusion should be drawn from this run alone

## Controls (unprofiled)

| block | reps | tg mean t/s | tg stddev |
|---|---|---|---|
| A | 10 | 21.24 | 3.68 |
| B | 10 | 20.75 | 4.22 |
| C | 10 | 21.43 | 3.04 |
| D | 10 | 20.18 | 4.42 |
| E | 10 | 22.93 | 3.21 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 3, Agent 4 (expected 4)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 54540 | 1742975.6 | 31.96 | 56.60 | 40 | 128 | 0 |
| `__amd_rocclr_copyBufferRectAligned` | 255 | 647118.6 | 2537.72 | 7030.51 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 3040 | 318168.4 | 104.66 | 532.52 | 24 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 56880 | 159029.4 | 2.80 | 4.44 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 66044 | 142827.2 | 2.16 | 3.16 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 4000 | 118116.2 | 29.53 | 54.04 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 18756 | 99520.4 | 5.31 | 7.32 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 32296 | 81393.7 | 2.52 | 3.96 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 2016 | 71020.2 | 35.23 | 62.20 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 2016 | 68756.1 | 34.11 | 57.40 | 40 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 5952 | 42209.8 | 7.09 | 9.48 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 5952 | 37937.0 | 6.37 | 9.00 | 16 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 7912 | 34973.4 | 4.42 | 6.36 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 11904 | 32428.1 | 2.72 | 3.88 | 8 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 432 | 31510.3 | 72.94 | 98.28 | 24 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 432 | 28574.1 | 66.14 | 89.40 | 144 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 14008 | 27369.4 | 1.95 | 3.12 | 16 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 11024 | 27163.3 | 2.46 | 3.56 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 5072 | 24883.7 | 4.91 | 8.52 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 5952 | 22210.8 | 3.73 | 5.04 | 24 | 128 | 0 |

## GPU profile: 2

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 3, Agent 4 (expected 4)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 60600 | 1938632.3 | 31.99 | 56.72 | 40 | 128 | 0 |
| `__amd_rocclr_copyBufferRectAligned` | 258 | 648050.3 | 2511.82 | 7028.91 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 3376 | 352706.2 | 104.47 | 532.52 | 24 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 62424 | 173630.7 | 2.78 | 4.44 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 72488 | 156065.9 | 2.15 | 3.12 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 4000 | 118287.9 | 29.57 | 54.04 | 24 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 20604 | 109006.6 | 5.29 | 7.28 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 35428 | 88811.1 | 2.51 | 3.96 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 2016 | 71069.3 | 35.25 | 61.72 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 2016 | 68816.8 | 34.14 | 57.32 | 40 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 6528 | 46941.4 | 7.19 | 9.44 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 6528 | 41875.8 | 6.41 | 9.00 | 16 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 8332 | 36558.6 | 4.39 | 6.32 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 13056 | 35526.1 | 2.72 | 3.88 | 8 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 480 | 35001.9 | 72.92 | 98.12 | 24 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 480 | 31924.3 | 66.51 | 89.72 | 144 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 15364 | 29818.5 | 1.94 | 3.12 | 16 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 12104 | 29811.9 | 2.46 | 3.52 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 5576 | 27851.1 | 4.99 | 8.56 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 6528 | 24492.0 | 3.75 | 4.84 | 24 | 128 | 0 |
