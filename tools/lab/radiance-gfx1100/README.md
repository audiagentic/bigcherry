# Standalone Radiance gfx1100 bootstrap (RAD11)

Experimental, **not a complete backend** and not a production switch. The full kernel-by-kernel porting matrix and dual-XTX P2P findings are in [the companion audit](../../../docs/research/radiance-libr4d-gfx1100-audit.md), consolidated into this branch. This is the first implementation slice toward running standalone [Radiance](https://codeberg.org/StillDeadcode/radiance) on two Radeon RX 7900 XTX cards. It uses the source-level plugin ABI of the [GitHub mirror at 6c11671 (Radiance 1.0.5)](https://github.com/AndrzejukPawel/radiance/tree/6c11671bdd31569978e26d14f190926cbd8d724b); newer Codeberg commits **must be checked before deployment**.

## Goals and limitations

The Radiance engine already builds gfx11-generic device bookkeeping. `libr4d` explicitly excludes gfx1100; changing its target regex silently miscomputes WMMA layouts. `libr11` is therefore an independent kernel-library plugin targeting exactly gfx1100, compiled against Radiance's actual installed ABI. The initial slice supplies BF16 `add` and `mul` (including one-row `b` broadcasting). A subsequent independent experiment supplies a restricted BF16 `gemm_nt` band (`1<=M<=16`, `K % 16 == 0`) as **two side-by-side candidates**: a slow scalar FP32 GPU control and a native gfx1100 wave32 WMMA kernel. It does not advertise missing operators.

This slice **cannot serve a model yet**: MiniCPM5-2B BF16 requires 13 model ops plus sampler ops, and tensor parallelism requires exact collectives. Port complete model-vocabulary coverage from `libref` next, with standalone `rad-kbench` correctness tests before TP2 serving. The 29.43-GiB Qwen3.8-27B FP8 Radiance container is a later target; its FP8 GEMMs, attention, GDN and DFlash2 add separate mandatory ops, and FP8 compute must be lowered to RDNA3-supported matrix instructions.

## Build against an installed standalone Radiance

On a ROCm 7.2 Linux host with the standalone engine installed (replace prefix as appropriate):

```sh
cmake -S tools/lab/radiance-gfx1100/libr11 -B /tmp/rad11-build -G Ninja \
    -DCMAKE_PREFIX_PATH=/opt/radiance \
    -DRAD_GPU_TARGETS=gfx1100 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build /tmp/rad11-build
# One selected XTX is sufficient for the first plugin smoke, no P2P used:
ctest --test-dir /tmp/rad11-build -R r11_bf16_smoke --output-on-failure
cmake --install /tmp/rad11-build
```

Select exactly one verified gfx1100 card with `ROCR_VISIBLE_DEVICES=<XTX-index>` before running CTest, then run `ctest --test-dir /tmp/rad11-build -R r11_gemm_smoke --output-on-failure` as well. For CMake's native HIP language, use ROCm Clang rather than the `hipcc` wrapper as `CMAKE_HIP_COMPILER`.

The build creates/installs `libr11.so` under Radiance's `kernels/` plugin directory via `rad_add_plugin`. Never place it in `libr4d` or enable an RDNA4 code object for gfx1100. If Radiance reports an ABI mismatch, rebuild against the exact running version; do not bypass the check.

In a fresh test environment, run `rad-kbench --kernels libr11,libref --report rad11-kbench.md --bench` with matching fixtures and inspect the `add` and `mul` rows for numerical, red-zone, broadcast and performance checks. Uncovered shapes are failures, not assumed successes. Repeat under the ROCm compiler actually used to serve.

## Brutus topology probe (read-only)

Determine which `ROCR_VISIBLE_DEVICES` indices correspond to the **two XTX cards**, not the R9700 or 6900 XT. Example indices are placeholders:

```sh
hipcc -O2 --offload-arch=gfx1100 \
  tools/lab/radiance-gfx1100/peer_probe.hip -o /tmp/r11-peer-probe
XTX_DEVICE_IDS='0,1' # REPLACE with the two verified XTX indices on this host
ROCR_VISIBLE_DEVICES="$XTX_DEVICE_IDS" /tmp/r11-peer-probe
```

The probe prints device architecture, VRAM and peer-capability results. It does **not** enable P2P, attempt bulk transfers, change IOMMU/ACS, or touch the other cards. Peer-capability alone does **not** establish RCCL P2P transport: follow with a separate safe RCCL A/B under the lab queue, `NCCL_PROTO=Simple`, bounded watchdog, small messages first, and compare with BigCherry's existing host-staged exact provider.

Do not use `RADIANCE_FAST_REDUCE=1` or replicate spin-on-peer-flag collectives from the older vLLM fork: its [gfx1100 capture/replay investigation](https://github.com/mkadrlik/vllm-radiance-p2p/blob/main/docs/fast-reduce-mtp-deadlock.md) explicitly rejected them for production. Do not disable IOMMU or override ACS automatically.

## Next slices toward an actual standalone TP2 server

1. **RAD11-A:** port all conventional BF16 elementwise, RMSNorm, embedding, rotary, sampler and fused-gate rows from `libref`. Preserve reference rounders and optional operand placement; compile and run `rad-kbench`.
2. **RAD11-B:** gfx1100 BF16/F16 WMMA fragment policy and dense GEMM, using validated BigCherry GDN/MMQ work as baselines. Explicitly verify gfx11 operand-lane replication; never use `*_gfx12` builtins.
3. **RAD11-C:** portable paged attention and GDN recurrent state/conv with BF16 KV; validate 2B BF16 TP1 first.
4. **RAD11-D:** exact TP2 collectives with a host-staged baseline and conditional RCCL/P2P acceleration only if Brutus proves safe. Prove graph replay and MTP rank agreement. Do not introduce lossy wires initially.
5. **RAD11-E:** MiniCPM5-2B BF16 TP2 end-to-end; then Qwen3.8-27B FP8 (E4M3 -> BF16/F16 compute without permanent dequantised weights), eventually Flash-Next QSA/MoE and DFlash2.

Each slice needs an isolated branch/PR and hardware evidence before merging. This bootstrap does not alter BigCherry's llama.cpp patch set, builds or runtime.

## RDNA3 BF16 WMMA vs independent scalar GPU control

The experimental `gemm_nt` candidates are `r11_gemm_bf16_scalar` (FP32 sequential reduction) and
`r11_gemm_bf16_wmma_gfx1100` (native `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`).
Both compute `Y[M,N] = A[M,K] @ B[N,K].T` with BF16 inputs/output and FP32 accumulation.
Their common registration is restricted to M=1..16, N=1..32768, K=16..8192 divisible by 16,
2D contiguous tensors; no speculative generic GEMM or quantized-weight support is claimed.

**RDNA3-specific difference:** each 32-lane wave computes one 16x16 output tile;
lane `l` feeds A row/B column `l % 16` in both halves of the wave (duplicated BF16 inputs).
Accumulator register `j` corresponds to output row `2*j + l/16` and column `l%16`.
The RDNA4 layout uses `8*(l/16)+j` and is **wrong** for gfx1100.
M/N tails are masked, and the CPU/GPU smoke deliberately uses odd N and asymmetric
matrices to catch transposition and lane-layout failures. Each candidate is independent:
the WMMA path does not call the scalar path.

Source comparisons:
- [AMD RDNA3 WMMA builtin and exact lane mapping](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html)
- [llama.cpp RDNA3 vs RDNA4 dispatch](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mma.cuh)
- [vLLM/AITER gfx1100 Triton W8A8 experience](https://github.com/vllm-project/vllm/issues/51136) (an independent reference, **not** a port of the CDNA-only CK kernel)

```sh
# Offline (no GPU): test the gfx11 lane mapping; does not prove HIP codegen.
python3 -m unittest discover -s tools/lab/radiance-gfx1100/tests -p 'test_*.py'

# After a gfx1100 build as above, on one verified XTX, with the matching installed Radiance:
export ROCR_VISIBLE_DEVICES=0            # example ONLY: replace with verified XTX ID
export RADIANCE_HOME=/opt/radiance:/tmp/rad11-build/radiance_home
ctest --test-dir /tmp/rad11-build -R 'r11_(gemm|bf16)_smoke' --output-on-failure
bash tools/lab/radiance-gfx1100/compare_gemm.sh \
    /tmp/rad11-build /opt/radiance/bin/rad-kbench /tmp/r11-gemm-new-run
```

The comparison runner refuses to overwrite reports, requires successful GPU smoke and
`rad-kbench` completion, and verifies both row names appear. **Presence is not coverage**:
inspect the Markdown report's per-kernel checked/skipped/failing geometry and timings.
The existing recorded Radiance fixture may lack some M/N/K bands, so add dedicated
recorded model fixtures before claiming performance or correctness for those bands.
Benchmark short decode M=1/2/8/16 and K=256..8192 against real model shapes; compare
the two on the same GPU, buffers, ROCm compiler, run order, and thermal/power state.

Next independent candidates to investigate after this numerical gate: gfx11 BF16 WMMA
LDS staging for prefill, direct i8 WMMA for W8A8/W4A8, FP8 E4M3 **packed storage**
with tile-local BF16/F16 conversion, then BF16 and FP8 E4M3 paged KV attention.
RDNA3 does not have RDNA4's native FP8 WMMA or transposed global-load instruction.
**No compiler/GPU performance evidence exists for these new GEMM candidates yet.**
