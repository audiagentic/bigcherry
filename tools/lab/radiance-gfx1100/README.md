# Standalone Radiance gfx1100 bootstrap (RAD11)

Experimental, **not a complete backend** and not a production switch. The full kernel-by-kernel porting matrix and dual-XTX P2P findings are in [the companion audit](../../../docs/research/radiance-libr4d-gfx1100-audit.md), consolidated into this branch. This is the first implementation slice toward running standalone [Radiance](https://codeberg.org/StillDeadcode/radiance) on two Radeon RX 7900 XTX cards. It uses the source-level plugin ABI of the [GitHub mirror at 6c11671 (Radiance 1.0.5)](https://github.com/AndrzejukPawel/radiance/tree/6c11671bdd31569978e26d14f190926cbd8d724b); newer Codeberg commits **must be checked before deployment**.

## Goals and limitations

The Radiance engine already builds gfx11-generic device bookkeeping. `libr4d` explicitly excludes gfx1100; changing its target regex silently miscomputes WMMA layouts. `libr11` is therefore an independent kernel-library plugin targeting exactly gfx1100, compiled against Radiance's actual installed ABI. The first slice supplies **only** BF16 `add` and `mul` (including one-row `b` broadcasting) with strict dtype, shape and stride validation. It intentionally does not advertise missing operators.

This slice **cannot serve a model yet**: MiniCPM5-2B BF16 requires 13 model ops plus sampler ops, and tensor parallelism requires exact collectives. Port complete model-vocabulary coverage from `libref` next, with standalone `rad-kbench` correctness tests before TP2 serving. The 29.43-GiB Qwen3.8-27B FP8 Radiance container is a later target; its FP8 GEMMs, attention, GDN and DFlash2 add separate mandatory ops, and FP8 compute must be lowered to RDNA3-supported matrix instructions.

## Build against an installed standalone Radiance

On a ROCm 7.2 Linux host with the standalone engine installed (replace prefix as appropriate):

```sh
cmake -S tools/lab/radiance-gfx1100/libr11 -B /tmp/rad11-build -G Ninja \
    -DCMAKE_PREFIX_PATH=/opt/radiance \
    -DRAD_GPU_TARGETS=gfx1100 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
cmake --build /tmp/rad11-build
# One selected XTX is sufficient for the first plugin smoke, no P2P used:
ctest --test-dir /tmp/rad11-build -R r11_bf16_smoke --output-on-failure
cmake --install /tmp/rad11-build
```

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
