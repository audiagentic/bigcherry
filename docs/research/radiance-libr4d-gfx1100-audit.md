# Radiance libr4d RDNA3 (gfx1100) kernel audit

Status: source-level assessment only; **no gfx1100 compile, hardware benchmark, or correctness qualification performed**.
Date: 2026-10-09. Target: BigCherry/Brutus **2 x Radeon 7900 XTX (gfx1100), TP2**, PCIe4 x8 topology.
Sources: [Radiance mirror at 6c11671 (2026-10-04)](https://github.com/AndrzejukPawel/radiance/tree/6c11671bdd31569978e26d14f190926cbd8d724b/libr4d), [its CMake](https://github.com/AndrzejukPawel/radiance/blob/6c11671bdd31569978e26d14f190926cbd8d724b/libr4d/CMakeLists.txt), [kernel plugin ABI](https://github.com/AndrzejukPawel/radiance/blob/6c11671bdd31569978e26d14f190926cbd8d724b/docs/PLUGIN.md), [RDNA3 WMMA](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html), [RDNA4 WMMA](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html).

**Source vintage warning:** this third-party GitHub mirror is Radiance 1.0.5, not necessarily the latest upstream Codeberg libr4d. Refresh source commit and manifest before implementation. Mirror's top-level LICENSE is Apache-2.0; verify current upstream licenses and attribution before incorporating any code.

## Interpretation

- **L**: relatively low ISA risk, but uncompiled and untested; usually scalar/index/sampling kernels.
- **M**: portable algorithm with dependencies on gfx12 helper headers, fragment packing, atomics, or state layout.
- **H**: high effort; RDNA4 WMMA/layout or unsupported native FP8 path, requiring a gfx11 implementation.
- **X**: architecture may permit the algorithm, but **blocked by this machine's measured/no-P2P topology** until a separate transport is built.
- **D**: 4/8-rank-only; defer for TP2.

Counts: L=11, M=14, H=26, X=5, D=3; **59 HIP translation units** (CMake also has host C++ declarations and helper headers). The CMake regex must preserve uppercase `N` in `*_Nrank_*`. A low-risk grade is **not** a claim that a .hip compiles as-is on gfx1100.

## Complete HIP translation-unit audit

| CMake translation unit (each under `libr4d/`, suffix `.hip`) | Grade | gfx1100 requirement |
|---|:---:|---|
| `r4d_attn_paged_h256_gqa6` | H | gfx12 WMMA in included prefill/decode bodies; port fragment/staging, test FP8 and BF16 KV |
| `r4d_attn_paged_h128_gqa4` | H | same included attention bodies; h128/GQA4 geometry and drafter window tests |
| `r4d_attn_paged_gqa8` | H | same included attention bodies; head 128/256, GQA8 instantiations |
| `r4d_attn_paged_h256_gqa12` | H | wrapper of GQA8 parametrised attention; QSA/GQA12 tests |
| `r4d_attn_vit_h72_bf16` | H | r4d_dt16 gfx12 BF16/F16 WMMA; gfx11 fragment replication and h72 tail |
| `r4d_vit_bf16` | H | r4d_gdn_wmma helper; port BF16 WMMA and fused vision kernels |
| `r4d_gdn_chunk_scan_k128_v128_c64_bf16` | H | gfx12 BF16 WMMA, persistent register state and LDS schedule; compare BigCherry 1253 |
| `r4d_gdn_conv_w4_h128_bf16` | M | mostly convolution/normalisation; isolate shared WMMA header, verify state cache |
| `r4d_gdn_kkt_solve_k128_c64_bf16` | H | gfx12 WMMA Gram/triangular solve; implement gfx11 fragment conversion |
| `r4d_gdn_recurrent_update_k128_v128_bf16_fp32state` | H | WMMA/state layout and speculative rollback; exact FP32 state tests |
| `r4d_gdn_gated_rmsnorm_h128_bf16` | M | scalar normalisation, but shared gfx12 helper must be isolated |
| `r4d_hc_bf16` | H | hyperconnection fused ops, gfx12 WMMA plus FP8 conversion and shared layouts |
| `r4d_ngram_ids` | L | integer/token lookup; check atomics and ranges |
| `r4d_ple_gate_bf16` | L | elementwise gate; portable HIP after common-header guards |
| `r4d_ple_conv_bf16` | L | convolution; portable HIP after common-header guards |
| `r4d_ar_entry` | X | cross-rank rendezvous/scratch depends on peer mapping; host/RCCL transport redesign |
| `r4d_ar_gather_2rank` | X | two-rank peer-memory gather; blocked if peer access unavailable |
| `r4d_ar_oneshot_2rank_exact` | X | two-rank direct P2P allreduce; replace with host-staged/RCCL for Brutus |
| `r4d_ar_oneshot_2rank_wht6` | X | P2P-dependent, lossy 6-bit wire; only trial after transport and quality gate |
| `r4d_ar_oneshot_Nrank_exact` | D | 4/8-rank only; out of scope for TP2 hardware |
| `r4d_ar_twoshot_Nrank_exact` | D | 4/8-rank only; out of scope for TP2 hardware |
| `r4d_ar_twoshot_Nrank_ti8` | D | 4-rank only, lossy wire; out of scope for TP2 hardware |
| `r4d_gemm_bf16_nt_m16` | M | forwarder to m64; update selection once gfx11 m64 works |
| `r4d_gemm_bf16_nt_m64` | H | gfx12 BF16 WMMA and fragment layouts; gfx11 duplicated 16-element operands |
| `r4d_gemm_bf16_nt_tiled` | H | gfx12 WMMA, LDS and tile scheduling need gfx11 tuning |
| `r4d_gemm_w4a16_nt_m64` | H | gfx12 F16 WMMA plus 4-bit weight relayout; reshape duplicated operands |
| `r4d_gemm_w8a16_nt_m64` | H | gfx12 F16 WMMA plus 8-bit weight relayout |
| `r4d_gemm_w2a8_nt` | H | gfx12 i8 WMMA and packed W2 draft-head quantisation |
| `r4d_gemm_w4a8_nt_m64` | H | gfx12 i8 WMMA, offline-packed fragments; repack for gfx11 |
| `r4d_gemm_w8a8_nt_m64` | H | gfx12 i8 WMMA, fragment replication and scale handling |
| `r4d_gemm_w8a8_tiled` | H | gfx12 i8 and iu4 WMMA intrinsics, LDS layout and scheduling |
| `r4d_gemm_w4a8_tiled` | H | gfx12 i8 and iu4 WMMA intrinsics, LDS layout and scheduling |
| `r4d_gemm_w4a8_prefill` | H | gfx12 i8 WMMA, prefill tile/load schedule |
| `r4d_gemm_mxfp4a8_nt_m64` | H | gfx12 FP8 WMMA unavailable on gfx1100; dequant to BF16/F16 or requant i8 |
| `r4d_gemm_fp8a16_nt_m1` | H | gfx12 BF16 WMMA on E4M3 weight; decode FP8 weights on load |
| `r4d_gemm_fp8a8` | H | gfx12 FP8 WMMA unavailable on gfx1100; dual dequant and gfx11 BF16 GEMM |
| `r4d_quant_act_fp8` | M | E4M3 pack/conversion and fragment order; gfx11 software conversion fallback |
| `r4d_fused_quant_fp8` | M | FP8 conversion and fused peer/norm paths; split transport and conversion |
| `r4d_gram_accum_fp8` | H | FP8 WMMA helper; BF16 conversion and gfx11 WMMA |
| `r4d_qsa_block_key_bf16` | L | reduction/key compression; guard shared common helpers |
| `r4d_qsa_score_bf16` | H | BF16 WMMA helper; gfx11 layouts, sparse score correctness |
| `r4d_qsa_select_bf16` | M | sorting/top-k/indexing; concurrency and scratch/layout tests |
| `r4d_qsa_tail_store_bf16` | L | bounded tail copy/index state; verify cross-step invariants |
| `r4d_qsa_work_bf16` | L | bounded worklist construction; verify dynamic shapes |
| `r4d_sample_chain_f32` | L | scalar fused sampling; RNG/ordering deterministic comparison |
| `r4d_model_bf16` | M | small ops, but includes gfx12 FP8 WMMA header; isolate unused utilities |
| `r4d_sample_stages_f32` | L | sampling stage kernels; deterministic tie-break/seed semantics |
| `r4d_quant_act_i8` | M | basic quantisation but output layout is gfx12 WMMA fragment order; redesign packing |
| `r4d_had_quant_act_i8` | M | Hadamard and quantisation portable, gfx11 fragment repacking |
| `r4d_rmsnorm_had_quant_i8` | M | RMSNorm+rotation portable, gfx11 fragment repacking |
| `r4d_ar_ln_had_quant_i8` | X | P2P fused collective unavailable without peer mapping; split host AR from fusion |
| `r4d_gated_had_quant_i8` | M | elementwise+rotation portable, gfx11 fragment repacking |
| `r4d_qk_norm_rope_gate` | L | elementwise QK norm, RoPE and gate; compare tolerances |
| `r4d_gdn_gated_norm_had_quant_i8` | M | GDN gated norm+rotation portable; gfx11 fragment repacking |
| `r4d_dequant_w4_bf16` | M | unpack is portable but consumes gfx12 offline weight layout |
| `r4d_rowtopk_bf16` | M | top-k math portable but FP8 helper header and tie handling require audit |
| `r4d_dflash_conv_t2_g16_bf16` | L | convolution elementwise; verify speculative geometry |
| `r4d_dflash_select_bf16` | L | candidate selection; acceptance and tie-breaking correctness |
| `r4d_moe` | H | FP8 WMMA and gfx12 sparse/expert GEMM; use BF16 gfx11 path or BigCherry MMQ |

## Shared non-translation-unit blockers

1. `CMakeLists.txt`: `rad_plugin_gpu_targets(R4D_GPU_TARGETS "^gfx12")` returns on gfx11. Do **not** simply change this to gfx11; unsupported opcodes/layouts can silently miscompute.
2. `r4d_gdn_wmma.h`, `r4d_dt16.h`: gfx12 `__builtin_amdgcn_wmma_*_gfx12` consumes **8 x 16-bit elements/lane**, whereas RDNA3 gfx11 WMMA consumes **16 with lane-half replication**. Reimplement fragment loads/stores, A/B interleaving, accumulator mapping; verify with independent 16x16 probes.
3. `r4d_fp8_wmma.h`: gfx12 FP8 WMMA has no RDNA3 equivalent. RDNA3 supports F16/BF16 and IU8/IU4 dense WMMA but **not native FP8 WMMA**. E4M3 storage is viable, decode exactly into BF16 prior to WMMA (or evaluate i8 requantisation as explicitly lossy); retune tile geometry and split-K. The header's prose suggesting an unsuffixed gfx11 FP8 variant is not a supported capability per AMD/Clang's published RDNA3 builtin list.
4. `r4d_common.h`: `__builtin_amdgcn_global_load_tr_b128_v8i16` is gfx12-only. Use regular global loads + lane shuffle/LDS transpose. Replace/guard gfx12 barrier assumptions and verify compiler emissions with `llvm-objdump`.
5. `r4d_fp8_frag.h`, `r4d_fp8_layout.cpp`, `r4d_layout.cpp`: fragment permutation and packed artifacts are coupled to gfx12, **even when the consuming kernel contains no explicit gfx12 builtin**. Have gfx11-specific weight packer, invertible unrelayout, and descriptor constraints.
6. All-reduce `r4d_ar_*` and fused `ar_ln_had_quant_i8`: peer-mapped scratch and device flags assume working GPU-to-GPU P2P. BigCherry PGC12/PGC13 report **no P2P on Brutus**; retain RCCL/host-staged exact provider. **Do not infer an RDNA3 hardware-wide prohibition:** a separate dual-XTX TRX50 installation reports reliable RCCL PCIe P2P with `NCCL_PROTO=Simple` after its own IOMMU/ACS configuration (see field report below). That same installation reports that the *custom spin-based* one-shot GPU all-reduce is **not viable**, despite functioning peer transport. Fused norm can be ported independent of transport. WHT6/ti8 changes numerics, require quality verification; do not turn on by default.
7. Device ABI and dispatch `r4d_rows.cpp`, `r4d_shapes.cpp`, `r4d_fuse.cpp`, `r4d_plugin.h`: declare **only tested gfx1100 shapes**. The plugin loader checks actual code objects and prioritises installed plugin order. Host-only libref is usable for testing but production should not require `--debug-accept-reference-kernels`.

## BigCherry integration / non-duplication

- `1253_nro04_gfx1100_bf16_chunked_gdn` is **validated** in patch.toml (README wording appears stale). Compare numerics and profiler evidence before proposing another GDN scan.
- `1334_hip_sparse_flash_attn`, `1350_mmq_few_tile_streamk`, and `1237_rd30_moe_mmq_compact_grid` are **validated**. Reuse baselines, do not substitute R4D attention/MMQ without A/B.
- `1250_nro01_allreduce_q8_wire` remains **evaluated**, and PGC12/PGC13 already cover transport/phase-aware routing and topology calibration; coordinate rather than duplicate.
- BigCherry production uses GGUF Qwen3.8 Flash-Next IQ4_XS or 27B Q8_0, unlike Radiance's `.rad` block-FP8, W4A8/MXFP4 layouts. A ported R4D GEMM is **not** a plug-in replacement for llama.cpp MMQ unless encoding, activation format, scales and tensor ordering match. Keep a standalone Radiance backend track distinct from a BigCherry patch track.
- QFP43 speculative draft acceptance regression is P0; this research makes no production changes and must not displace that fix.

## Implementation slices (one branch and PR each; do not change main directly)

1. **ABI + cheap coverage:** out-of-tree `libr11`/gfx1100 kernel plugin with independent `rad_plugin_info()`, explicit `gfx1100` code-object selection, low-ISA-risk L rows. Use reference backend only for test comparisons; publish supported op/shape registry. Runtime dispatch must fail closed on unsupported shapes.
2. **WMMA correctness harness:** isolated 16x16 gfx11 BF16/F16/IU8 WMMA with independent CPU oracle, layout bijection, and `llvm-objdump` opcode check. Implement separate gfx11 fragment-policy headers; never compile gfx12 intrinsics for gfx11. Move BF16 GEMM and GDN KKT/scan next, one family at a time.
3. **Attention + QSA:** port paged h256 GQA6/GQA12 (then GQA8 and h128), BF16 KV first; add FP8 cache E4M3 -> BF16 input decode. Compare masking, ragged tails, causal/sliding-window, split-KV and 200K-context index behavior; then QSA indexer and sparse attention.
4. **Dual-GPU transport:** measure `hipDeviceCanAccessPeer`, peer copy and cross-rank flags for each XTX pair. No P2P => stage via host/RCCL exact allreduce, measure transfer/phase crossover per PGC12/13. WHT6 only as opt-in lossy experiment with 2-rank symmetry and output-quality guard.
5. **Quantised GEMM:** gfx11 W4A8/W8A8 integer WMMA with separate offline packing, compare to BigCherry's validated MMQ; then FP8/MXFP4 via FP8 decode to BF16 and gfx11 BF16 WMMA. Avoid doubling persistent FP8 weights into BF16 (dual XTX has 48 GiB VRAM total and needs KV/scratch).
6. **Model/serving qualification:** start Qwen3.8-27B FP8 model in Radiance TP2, not 114 GiB Flash-Next. Trace every resolved operator and missing plugin band; compare greedy outputs and speculative acceptance against known reference. Flash-Next/GDN/QSA and draft selectors only after complete op coverage.

## Exit gates per slice

- Compiler: target `gfx1100` on the tested ROCm/Clang stack; zero gfx12-only builtins; `r4d_selftest`, `rad-kbench` and independent CPU/golden reference; disassemble and record ISA and VGPR/LDS/occupancy.
- Correctness: all supported shapes and tails, no NaN/Inf, bounded error for approximate paths, bit/greedy identity wherever promised; draft acceptance, state replay/rollback, and two-rank agreement.
- Performance: ABBA on dual XTX at 8K/24K/98K prefill and decode/MTP, record latency p50/p90, transfer cost, full-model t/s, peak VRAM, disallow >0.5% unrelated prefill regression and require reproducible gains (>=3% margin for provider selection); no fabricated expected speedups.
- Deployment: separate optional gfx1100 plugin, explicit enable/disable and diagnostics, no production switch before hardware evidence.

## gfx1100 deployment evidence discovered 2026-10-09

Source: [mkadrlik/vllm-radiance-p2p main fc022019](https://github.com/mkadrlik/vllm-radiance-p2p/tree/fc022019051dbedc794572d46a6306585a61a89a); [README](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/README.md); [engineering notes](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/AGENTS.md); [capture deadlock postmortem](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/docs/fast-reduce-mtp-deadlock.md). This is a **vLLM-Radiance 0.5.7 / vLLM 0.26.0** derivative with `gfx1100` AITER/Triton and custom HIP sources, **not an RDNA3 build of current standalone Radiance/libr4d**. Different model file formats and runtime.

Verified from source (performance and machine behavior are **the fork author's reports**, not reproduced on Brutus):

- Production configuration: dual RX 7900 XTX, TP2, Qwen3.8-27B AWQ-INT4 + MTP, `RADIANCE_FAST_REDUCE=0`, `NCCL_PROTO=Simple`. Reported 23.8–28.9 t/s single stream in this serving environment; do **not** compare to BigCherry Q8_0/MTP 78–80 t/s without matching quant/model, hardware and sampling constraints. Authors say all-reduce rather than weight bandwidth dominates TP2 at batch 1.
- Reported **PCIe P2P works for standard RCCL** on their TRX50 workstation when IOMMU is disabled and ACS barriers absent. This **does not prove** Brutus can activate direct P2P: verify BIOS/PCI topology, root complex, ROCm driver, peer-capability flags and measured peer paths. Do not recommend changing system-wide IOMMU or PCIe ACS in production without security and stability review; ACS overrides can undermine isolation.
- Custom all-reduce `build/patches/radiance_ar_ext.hip` uses IPC peer scratch, fine-grained BAR, per-block device counters, peer flag spin. Eager asymmetric all-reduce during CUDA graph warmup deadlocked; capture-only gating passed toy tests but live replay still livelocked. `RADIANCE_FAST_REDUCE=0` is a deliberate production safety decision, NOT a claim that RCCL P2P fails.
- [Unmerged `feat/rdna3-ar-flush-protocols` commit 64d425d](https://github.com/mkadrlik/vllm-radiance-p2p/tree/64d425d891b98a0822728771bb8c9953122db39b) contains [Phase 0 measurements](https://github.com/mkadrlik/vllm-radiance-p2p/blob/64d425d891b98a0822728771bb8c9953122db39b/docs/PHASE0_RESULTS.md) and [decision spec](https://github.com/mkadrlik/vllm-radiance-p2p/blob/64d425d891b98a0822728771bb8c9953122db39b/docs/rdna3-one-shot-ar-spec.md): a plain global store + `s_waitcnt vmcnt(0)` was ordered, but mutual polling and payload pushes starved peer visibility in graph replay (phase 1 fails). Their RCCL reference timings: 30 µs @ 10 KiB, 34 µs @ 80 KiB, 76 µs @ 640 KiB; entirely **different topology/stack** from Brutus. The branch records `NOT VIABLE` for its one-shot protocol and leaves the default off.
- Reusable implementation sources: [`build/patches/patch_gfx1100.py`](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/build/patches/patch_gfx1100.py) (architecture dispatch, AITER allow-list, W8A8 Triton GEMM), [`build/patches/patch_gdn_wmma.py`](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/build/patches/patch_gdn_wmma.py) (GDN triangular solve), [`build/aiter-configs/gfx1100-GEMM-A8W8.json`](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/build/aiter-configs/gfx1100-GEMM-A8W8.json) (shape tuning), [`build/patches/router_gemm.hip`](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/build/patches/router_gemm.hip) and [`docs/triton-gfx1100-gap-analysis.md`](https://github.com/mkadrlik/vllm-radiance-p2p/blob/fc022019051dbedc794572d46a6306585a61a89a/docs/triton-gfx1100-gap-analysis.md). The `Dockerfile.gfx1100` layer's upstream base has drifted to RDNA4 and the author says rebuilding the complete pinned source-wheel stack may be necessary. Its prebuilt image cannot be assumed bit-reproducible from that Dockerfile.
- Kernel portability implication: RDNA3 W8A8 GEMM/WMMA work **already exists** in the vLLM family. The fastest validation track is to isolate its gfx1100 HIP/AITER programs as reference baselines for compatible *formats*, then port the latest standalone libr4d algorithms through gfx11-specific fragment policy rather than duplicating entire vLLM or treating the old fork as a current libr4d port.

**Immediate Brutus experiments (nondisruptive first):** snapshot `rocminfo`, PCIe BDF/root ports and `lspci -vv` ACS flags, IOMMU state, `hipDeviceCanAccessPeer`, topology/RCCL path selected, and peer-copy feasibility in a dedicated lab slice. Then ABBA `RCCL_PROTO=Simple` or `NCCL_PROTO=Simple` as appropriate to provider, exact collective against BigCherry's host path, with bounded timeout and no startup stress/bandwidth sweep. Keep graph-spin fast AR disabled; do **not** toggle IOMMU/ACS as part of automated testing. Any transport selection must be topology-specific and phase-aware per PGC12/PGC13.

## Source review limitations

Static source review only. Source version is older than current Codeberg upstream and live source cannot be verified through GitHub. No hipcc/ROCm GPU execution in this environment. Rows are porting estimates, **not** verified gfx1100 runtime compatibility.
