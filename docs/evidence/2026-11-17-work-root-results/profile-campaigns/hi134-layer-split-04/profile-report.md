# Profile campaign: hi134-layer-split-04

- model: `<models-root>/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf`
- lane: `bigcherry-native:control` on `linux-multi`
- devices: `0,1,3` (expected GPU count: 3)
- runtime profile: `hi134-layer-split-control`
- workload: `hi134-layer-split-013-negative-control`
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
| A | 10 | 36.69 | 6.53 |
| B | 10 | 39.49 | 7.03 |
| C | 10 | 42.03 | 2.96 |
| D | 10 | 39.06 | 5.52 |
| E | 10 | 38.54 | 6.76 |

## CPU profile

unavailable -- perf is not usable in this environment yet (see HI133)

## GPU profile: 1

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 4 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 13635 | 1253113.0 | 91.90 | 206.52 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 544 | 331243.4 | 608.90 | 2688.50 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 1000 | 81858.0 | 81.86 | 195.72 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 324 | 69214.5 | 213.62 | 381.20 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 504 | 48004.8 | 95.25 | 218.28 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 504 | 41938.9 | 83.21 | 200.96 | 40 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 16511 | 35274.6 | 2.14 | 3.32 | 16 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 1488 | 32989.4 | 22.17 | 29.72 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 1488 | 28771.0 | 19.34 | 27.04 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 8074 | 23987.6 | 2.97 | 4.84 | 32 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 4689 | 23077.9 | 4.92 | 7.00 | 16 | 128 | 0 |
| `__amd_rocclr_fillBufferAligned` | 30 | 17819.3 | 593.98 | 2776.81 | 16 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 5516 | 13537.9 | 2.45 | 3.72 | 24 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 1488 | 13510.3 | 9.08 | 16.16 | 24 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 1474 | 8701.5 | 5.90 | 8.64 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 3502 | 8427.0 | 2.41 | 3.68 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 2976 | 7604.9 | 2.56 | 4.04 | 8 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 2756 | 7201.2 | 2.61 | 4.00 | 16 | 128 | 0 |
| `void flash_attn_tile<256, 256, 8, 2, false>(char const*, char const*, char const` | 476 | 6379.2 | 13.40 | 19.64 | 144 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 1268 | 5884.7 | 4.64 | 10.12 | 24 | 128 | 0 |

## GPU profile: 2

- capture status: **complete**
- GPU agents seen: Agent 1, Agent 2, Agent 4 (expected 3)
- expected reduction provider: meta
- RCCL activity seen: False

| kernel | calls | total us | mean us | p95 us | VGPR | SGPR | scratch |
|---|---|---|---|---|---|---|---|
| `void mul_mat_vec_q<(ggml_type)8, 5, false, false, false, 0, 0>(void const*, void` | 10100 | 928916.2 | 91.97 | 206.52 | 40 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 0, 0>(void const*, void` | 404 | 248198.4 | 614.35 | 2689.47 | 16 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 2, false, false, false, 0, 0>(void const*, void` | 1000 | 81358.8 | 81.36 | 195.88 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 1, true, false, false, 0, 0>(void const*, void ` | 240 | 51280.9 | 213.67 | 381.36 | 24 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 6, false, false, false, 0, 0>(void const*, void` | 504 | 48448.5 | 96.13 | 217.64 | 48 | 128 | 0 |
| `void mul_mat_vec_q<(ggml_type)8, 4, false, false, false, 0, 0>(void const*, void` | 504 | 42070.2 | 83.47 | 205.52 | 40 | 128 | 0 |
| `quantize_q8_1(float const*, void*, long, long, long, long, long, unsigned int, H` | 12752 | 27617.4 | 2.17 | 3.36 | 16 | 128 | 0 |
| `void gated_delta_net_cuda<128, false, true>(float const*, float const*, float co` | 1152 | 25388.0 | 22.04 | 29.32 | 40 | 128 | 0 |
| `void concat_non_cont<unsigned int, 0>(char const*, char const*, char*, long, lon` | 1152 | 22287.7 | 19.35 | 26.96 | 16 | 128 | 0 |
| `void cpy_scalar<&(void cpy_1_scalar<float, float>(char const*, char*))>(char con` | 6247 | 19252.4 | 3.08 | 4.80 | 32 | 128 | 0 |
| `void rms_norm_f32<1024, true, false>(float const*, float*, int, long, long, long` | 3611 | 17805.1 | 4.93 | 6.92 | 16 | 128 | 0 |
| `__amd_rocclr_fillBufferAligned` | 30 | 17695.1 | 589.84 | 2410.99 | 16 | 128 | 0 |
| `void k_get_rows_float_vec<float>(float const*, int const*, float*, long, long, H` | 1152 | 10821.6 | 9.39 | 16.56 | 24 | 128 | 0 |
| `void k_bin_bcast<&(op_add(float, float)), float, float, float, float const*>(flo` | 4270 | 10394.3 | 2.43 | 3.56 | 24 | 128 | 0 |
| `__amd_rocclr_copyBuffer` | 1159 | 6692.7 | 5.77 | 8.52 | 16 | 128 | 0 |
| `void unary_gated_op_kernel<&(op_silu(float)), float>(float const*, float const*,` | 2711 | 6369.7 | 2.35 | 3.88 | 16 | 128 | 0 |
| `void l2_norm_f32<32>(float const*, float*, int, long, long, long, float)` | 2304 | 5872.1 | 2.55 | 4.00 | 8 | 128 | 0 |
| `void rms_norm_f32<256, true, false>(float const*, float*, int, long, long, long,` | 2126 | 5664.7 | 2.66 | 3.96 | 16 | 128 | 0 |
| `void flash_attn_tile<256, 256, 8, 2, false>(char const*, char const*, char const` | 357 | 4942.7 | 13.85 | 19.68 | 144 | 128 | 0 |
| `void rope_multi<true, false, float>(float const*, float*, int, int, int, int, in` | 974 | 4644.2 | 4.77 | 9.96 | 24 | 128 | 0 |
