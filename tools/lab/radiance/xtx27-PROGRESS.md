# XDP01–XDP06 progress — 2026-11-09

## Implemented and checked

- [x] Experimental worktree `the isolated XDP worktree`, branch `exp/xtx27-prefill-opt`, no changes to primary checkout.
- [x] Fail-closed dual-XTX Radiance benchmark runner using MEN03, measured binary/model/corpus provenance, dry-run default, fixed two-rank no-P2P setup, FP8 KV and drafting off.
- [x] `--embedding-placement ram` explicitly selected to preserve VRAM for long-context KV; VRAM embedding remains a separate potential short-context experiment, not a capacity default.
- [x] Existing MXFP4 build tested on both XTX. One-rep WHT6 smoke at ~704/2,305/6,679 prompt tokens: 2,085/2,581/2,534 TTFT-derived tok/s.
- [x] Two-rep matched 8192-context run at ~6,680 actual prompt tokens: exact wire, batch2048 **2,115 tok/s**; WHT6 batch2048 **2,512 tok/s**; WHT6 batch4096 **2,513 tok/s**. WHT6 has known lossy behaviour, not yet quality qualified.
- [x] Isolated per-op diagnostic: rank-0 aggregated MXFP4 `gemm_nt_q` ~2,311 ms, `all_reduce` ~821 ms, GDN chunk scan ~215 ms across prefill + 32 decode steps. Profiling forces synchronisation and is not comparable as a speed score.
- [x] Capacity-first plan XDP02, full-attention KV arithmetic with explicit layer types, TP2 prefix and single-card R9700 suffix, stage-owned state only (not implemented).
- [x] At hypothetical 1,048,576 FP8 tokens, 64 XTX TP2 layers needs 16 GiB attention KV per XTX; 40 XTX + 24 R9700 layers needs 10 GiB per XTX + 12 GiB R9700. R9700 stage weight estimates are illustrative, not runtime-resident proof.
- [x] Saved capacity envelope under `${BC_WORK_ROOT}/xtx27-prefill/1m-kv-envelope.json`; benchmark evidence under the same scratch parent, no benchmark evidence committed to Git.
- [x] 10 `unittest` card-free tests pass; Ruff check clean.
- [x] Our GPU lease released after all engines exited; no GPU processes remained.
- [x] XDP03 new BF16-based mixed-quantization plan and dry-run recipe generator, matched MXFP4 control, W4 FFN middle-prefix profile, optional R9700 suffix FP8. Local original BF16 model checkpoint/18 shards verified; no download.
- [x] Emitted matched BF16-origin baseline/mixed recipe and config-fingerprinted manifests to isolated scratch `quant-lab/` (no `.rad` conversion yet); `--write` only, default read-only.
- [x] Card-free measured-input-constrained selector tests: stage-owned logical FP8 KV reservation, per-card VRAM reserve and exact bounded memory/latency DP. Accepts `selected` generator profile only with measured evidence; no actual placement or runtime quant switching.
- [x] 31 `unittest` card-free tests pass (including 21 new XDP03 tests); Ruff clean.
- [x] Once RAD49 lease cleared, acquired exclusive GPU lease and ran pinned `rad-convert --plan-only` on the BF16 control and mixed W4 recipes; both PASS, 1,024 planned weights, 289 quantized/735 as-is, estimated 16.50 GiB (control) vs 16.43 GiB (mixed), no .rad files written.
- [x] Ran actual gfx1100 `libr3` shape N17408 K5120 kernel timings: at M512 W4A8I8 3226.6 µs minimum vs MXFP4 tiled 931.6 µs (W4 3.46x slower); M64 W4A16 best 368.3 vs MXFP4 tiled 308.7 µs; M1 W4A16 best 76.9 µs vs MXFP4 gated 103.4 µs / decode 146.9 µs (not matching end-to-end paths). Raw logs isolated in `quant-lab/` scratch.
- [x] STOP broad W4 prefill conversion under existing kernels (insignificant memory saving, severe kernel slowdown); runtime quality/KL not measured. GPU probes ended cleanly and exclusive lease released.

- [x] XDP04: isolated HIP gfx1100 native W4A4 IU4 WMMA microkernel + one `xtx27_w4a4.py` CLI for plan, CPU synthetic quant quality, stand-alone compile and one-leased GPU six-case suite. No Radiance patch, model conversion or runtime integration.
- [x] gfx1100 HIP compiler succeeded (warning-free); initial off-diagonal GPU mismatch traced to gfx11 interleaved accumulator/transposed W-first layout and fixed. Native W4A4 unsigned-weight/signed-activation group128 zero correction now passes all outputs of six GPU oracle geometries, K up to 1024, error zero. Lease auto-released.
- [x] 45 card-free XDP tests pass, including 14 new XDP04 cases; Ruff clean; scratch build binary/manifest and GPU logs in `${BC_WORK_ROOT}/xtx27-prefill/w4a4`.
- [x] Synthetic outlier caution: W4A4 relative L2 0.374 vs W4A8 0.193 on random 16x32x256 sampled matrices; this is not model KL. No production quality or speed claims.

- [x] XDP05 detailed design/decision log: cooperative 8-wave tiled gfx11 native IU4 × IU4 WMMA, 2×4 / 4×2 / 8×1 geometries (32×64, 64×32, 128×16), tile-major operand prepack, signed-A4/unsigned-W4 per-group scales/zero correction. No production modification or new worktree.
- [x] Added tiled source to existing lease-guarded `xtx27_w4a4.py --kernel tiled` runner, with distinct source/binary hash manifests, nine shape cases × three tilings = **27 complete CPU-oracle GPU tests**; all passed exactly including tails/non-aligned dimensions.
- [x] Actual dense 27B projection (M512,N17408,K5120) per-tile experiments scan all 8,912,896 outputs for non-finite and CPU-verify 768 stratified/random samples; every sample passed exactly. ISA confirmed native gfx11 `v_wmma_i32_16x16x16_iu4`, VGPR 86–91, LDS 6–9 KiB, no reported spills.
- [x] Same-card 3-rep MXFP4 tiled median **922.7 µs** vs best native W4A4 tiled median **1,747.0 µs** (1.89× slower). W4A4 4×2=1,839 µs and 8×1=2,213 µs. W4A4 excludes new activation quantization, repacking, TP/serving; code arithmetic correct but both performance and real-model quality are **NOT qualified**. Logs saved in `${BC_WORK_ROOT}/xtx27-prefill/w4a4`; leases released.
- [x] XDP05 NO-GO (D1–D3): do not convert BF16 to broad W4A4, patch production libr3 or promote dynamic format policy. Prioritize proven MXFP4 path and TP all-reduce/CPU coordination profiling and optimization. Card-free checks and doc catalog updated.

- [x] XDP06 read-only RDNA3 upstream survey recorded: hipfire genuine gfx1100 IU4 128x128 (8/16 wave), XTX 1,061-us kernel report at N17408 K5120 M512 (unpaired), group128 A4 MSE candidates and fused producer; Strata FP16/BF16 WMMA and K-quant MMQ (Blackwell FP4 W4A4 is not RDNA3); hipEngine tuning, llama.cpp/Lucebox small-row specializations, Quark W4A16 GPTQ nibble repacking. Full sources, transfer limits, legal/quality caveats and design decisions in XDP06.md.
- [x] XDP06 reopens only a **bounded** independent 128x128 native IU4 research kernel (16-wave low-live-acc first); current 32x64 XDP05 remains no-go for serving. No full conversion, production integration or GPU launch from the source survey.

- [x] XDP06 implemented **independent** gfx1100 128x128/16-wave native IU4 correctness candidate with separate source/binary manifests under existing single tester --kernel wide. 16,384 exclusive outputs/workgroup (card-free proof), 544 CTAs at M512 N17408; rolled K16 schedule lowered compiled VGPR 235 to 147 with zero reported spills, 16KiB LDS and real IU4 ISA. All 53 card-free XDP tests pass; hardware numerical suite and speed are **PENDING GPU lease**. Full design/resource decision in XDP06.md.

- [x] XDP06 second *same-binary* output-store ablation: compile-time direct vs LDS-transposed/coalesced row-major final store (32x129 skewed scratch reused after group K loop, 4 slices, +8 barriers). Card-free proves 4096 unique producer/consumer LDS cells per slice and 16384 unique final outputs. Actual gfx1100 compiler: direct 147 VGPR/66 SGPR/16384 LDS B, coalesced 149 VGPR/68 SGPR/16512 LDS B; both 0 spill/private. One lease-guarded wide suite checks 12 shapes x 2 modes=24; full 27B via --wide-store direct|coalesced. 55 XDP card-free checks PASS. Hardware speed/correctness **NOT YET TESTED** while RAD49 GPU lease is held.

## Still required

- [ ] Bench 32K-256K **native** model context with actual resident KV/GDN/graph/runtime accounting; quality, TTFT and decode measured across context and sequences.
- [ ] Exact/WHT6 correctness/KL gate and longer repeated, rotated A/B to qualify speed improvement.
- [x] Tiled native IU4 W4A4 feasibility vs MXFP4: **tested and rejected for now** (XDP05, 1.89× slower GEMM-only). Native gfx11 IU4 instruction is real, but standalone correctness does not imply viable model throughput.
- [ ] XDP06 gated follow-up: independently test 128x128 native gfx11 IU4 16-wave output tile, full-output oracle, ISA/occupancy, 3-rep matched XTX A/B against XDP05 and MXFP4; include per-group A4 quantization/fused producer and model KL before any promotion. Require a measured complete path faster than 923 µs at M512/N17408/K5120 and real-model KL parity. Preserve the correct XDP05 tile/oracle code as a research fixture.
- [ ] If native W4 becomes viable, convert BF16->MXFP4 matched control and mixed candidate and run full end-to-end KLD, TP2 throughput, memory and KV residency gates. This conversion was deliberately not started.
- [ ] Obtain comparable measured per-layer W4/MXFP4/FP8 perf, peak weights and reserves; run the XDP03 GPU-aware selector with real evidence and measure selected profile end-to-end before admitting or promoting any dynamic policy.
- [ ] Native gfx1100 MXFP4 GEMM and TP all-reduce isolated kernel experiments with correctness; identify safe speed/capacity trade-off.
- [ ] Full-layer R9700 stage + activation handoff microharness, memory planner admission and KV/GDN state ownership proof. Current Radiance helper infrastructure is MoE-only and does not implement this split.
- [ ] Future 1M checkpoint availability, true geometry, architecture and training/rope scaling support; 1M on current local Qwen3.8-27B is outside its declared 262,144-token native limit.
- [ ] Production changes only after measured gates and review, no push/PR.

Raw measurement definitions: TTFT-based tokens/s, *not* pure prefill kernel throughput. Strata Flash-Next MoE benchmarks are not matched to this dense 27B run.
