# Profile campaign: rd33-profile-01

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1` (expected GPU count: 2)
- runtime profile: `production-dual-xtx`
- workload: `rd33-decode-profiling`
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
- environment stable: **False** -- control block means spread 32.4% across the run (threshold 5%) -- profiling artifacts are still real and usable diagnostically, but no comparative performance conclusion should be drawn from this run alone

## Controls (unprofiled)

| block | reps | tg mean t/s | tg stddev |
|---|---|---|---|
| A | 10 | 62.47 | 12.04 |
| B | 10 | 45.49 | 4.89 |
| C | 10 | 54.44 | 4.37 |
| D | 10 | 50.97 | 11.12 |
| E | 10 | 49.03 | 9.86 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2 (expected 2)
- RCCL activity seen: True

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)` | 11508 | 1449077.8 | 125.92 | 162.00 | 256 | 128 | 568 |
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0, false>(void const` | 35350 | 1445893.9 | 40.90 | 73.28 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0, true>(void const*` | 2830 | 403787.4 | 142.68 | 821.44 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0, false>(void const` | 2018 | 242264.9 | 120.05 | 73.76 | 40 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 41386 | 78980.6 | 1.91 | 3.28 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0, false>(void const` | 2000 | 78341.8 | 39.17 | 65.96 | 24 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 4032 | 59770.7 | 14.82 | 14.84 | 40 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 12726 | 58016.7 | 4.56 | 7.28 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 21882 | 56342.2 | 2.57 | 5.28 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0, false>(void const` | 1008 | 49699.0 | 49.30 | 92.92 | 48 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 4032 | 48838.2 | 12.11 | 16.16 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0, true>(void const*,` | 426 | 45738.4 | 107.37 | 112.96 | 32 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 3, false, false, false, 0, 0, false>(void const` | 1010 | 40805.6 | 40.40 | 66.52 | 32 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 15540 | 40581.8 | 2.61 | 4.80 | 24 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 3936 | 33305.5 | 8.46 | 14.76 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 9360 | 20806.8 | 2.22 | 3.52 | 16 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 4564 | 20308.9 | 4.45 | 6.52 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 8064 | 19070.0 | 2.36 | 4.00 | 8 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 298 | 17711.8 | 59.44 | 81.80 | 144 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 7476 | 17433.8 | 2.33 | 3.76 | 16 | 128 | 0 |

## GPU profile: 2

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2 (expected 2)
- RCCL activity seen: True

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0, false>(void const` | 23230 | 952904.1 | 41.02 | 79.04 | 40 | 128 | 0 |
| `ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)` | 7384 | 408760.1 | 55.36 | 148.56 | 256 | 128 | 568 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0, true>(void const*` | 1296 | 199374.4 | 153.84 | 874.52 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0, false>(void const` | 2000 | 77211.0 | 38.61 | 65.92 | 24 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 27246 | 53761.1 | 1.97 | 3.32 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0, false>(void const` | 1008 | 50328.4 | 49.93 | 93.60 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0, false>(void const` | 1008 | 40850.2 | 40.53 | 70.56 | 40 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 8146 | 38225.5 | 4.69 | 7.20 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 14060 | 37544.6 | 2.67 | 5.36 | 32 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 2592 | 32142.6 | 12.40 | 16.52 | 24 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 2592 | 28790.9 | 11.11 | 14.92 | 40 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 9976 | 26586.0 | 2.66 | 4.68 | 24 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 2592 | 23253.2 | 8.97 | 14.88 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0, true>(void const*,` | 184 | 19569.5 | 106.36 | 109.04 | 32 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 3562 | 16618.5 | 4.67 | 6.76 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 6100 | 14194.5 | 2.33 | 3.60 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 5184 | 12557.4 | 2.42 | 4.00 | 8 | 128 | 0 |
| `void flash_attn_ext_vec<256, 1, (ggml_type)8, (ggml_type)8, false>(char const*, ` | 184 | 11706.1 | 63.62 | 81.60 | 144 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 4792 | 11385.6 | 2.38 | 3.72 | 16 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 2200 | 10869.7 | 4.94 | 9.12 | 24 | 128 | 0 |
