---
id: RAD15
order: 15
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-20T12:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Engine-side MoE decode rows: the worker GEMVs as kernel-plugin rows for gfx1100 (libr3) and gfx1201 (bc-moe-gfx12)

## Description

The fast FP8 and W4 decode GEMVs (`lane_moe_gemv_h2`, `lane_moe_gemv_w4h2`; RAD14) run only inside the moe-remote worker. The engine's own MoE decode still uses libr4d's one grouped-GEMM row per format, `moe_gemm_fp8a8` / `moe_gemm_w4a8` (16/32-row WMMA tiles, NOTUNE). Radiance resolves every op by scanning the `--kernels` plugins in order: the first plugin with a matching row wins, and an op's schema is fixed by the first plugin that declares it (core/plugin/registry.h, select.cpp at 7a7707a). So the GEMVs can serve the engine with NO engine patch, as extra `moe_gemm_q` rows restricted to small decode steps, with libr4d serving everything else.

Two targets:
- gfx1100: rows in libr3, already a plugin that overrides libr4d rows. Speeds every XTX engine: Qwen3.8-27B on two XTX, the 2xXTX Flash-Next profile, and an XTX-primary plus R9700-worker layout. Measured on one XTX for a 16-row 35B request: grouped GEMM 260 us, packed-f16 GEMV 149 us.
- gfx1201: a new component `bc-moe-gfx12` listed before libr4d, on native `v_dot4_f32_fp8_fp8`. On the R9700, libr4d's grouped GEMM is 135 us against our f16 GEMV's 115 us on captured requests, and synthetic 16-row decode is near the DRAM floor (GPT: about 30% headroom at 16 rows; 103 us against a ~72 us floor at 4 rows), so this row is go/no-go on measurement.

## Steps

1. Fix `tools/lab/radiance/kbench_step.py` attribution: join each graph instance to its exact selected row and geometry; exclude vision and prefill; count MTP separately.
2. R9700 baseline: engine-alone `moe_gemm_q` gate/up and down under rocprofv3 on live decode (prefix cache off), 1 and 4 streams. Confirm whether gfx1201 has FP8 WMMA (ISA and libr4d disassembly).
3. gfx1100 rows in libr3: `moe_gemm_q` rows for fp8a8 and w4nl64a8h, schema-identical to libr4d's, constrained to small decode steps; the GEMVs adapted to the op's operands (sorted/off, gate/up part pitch, scratch). `--debug-selection` must show the row for decode and libr4d for prefill.
4. A/B with rad-kbench (`--kernels libr3:libr4d:libref --op moe_gemm_q --all-rows`) and on captured requests; then XTX engine stream A/B (27B and Flash-Next 2xXTX).
5. gfx1201 row (`bc-moe-gfx12`, dot4) only if step 2 shows at least 20% headroom.
6. Topology A/B: XTX tp2 engine with the new libr3 rows plus the R9700 as worker, against the current R9700 engine plus XTX workers.
7. Qualification: fix the broken W4 CPU reference in the worker harness first; then CPU reference, KL/top-1, MTP acceptance at 1/2/4/6 streams, and at least 10% end-to-end decode before default-on.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

`--debug-selection` shows the new row for decode-sized `moe_gemm_q` and libr4d for prefill; rad-kbench time against libr4d at the model's shapes; captured-request replay on identical inputs; KL unchanged against the same engine on libr4d rows; MTP acceptance unchanged; matched stream A/B at least 10% decode.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No engine patch: plugin rows only (the plugin hierarchy is absolute, first match wins). Risks: the schema must match libr4d's exactly or the plugin is refused at load; the W4 packed layout is libr4d-private (verify byte identity); selection is by host-side geometry only (no device-to-host sync for routing statistics). Related: RAD14 (worker GEMVs), RCS05 (backend seam), RCS10 (calibrated crossovers).

2026-10-21 findings (rad-kbench at real shapes on the R9700, joined per graph instance by kbench_step.py, PR #351):
- Flash-Next decode, 4 rows (1 stream x MTP 3): the engine's MoE GEMM is NOT a significant cost (the XTX workers own the cold
  experts). Large int8 dense GEMMs run at 535-588 GB/s (N=16384/12288 at K=2560, N=2560 at K=6144), 85-93% of the measured
  630 GB/s cold read: no headroom. The LM head (logits_gemm_i8, 248320 x 2560) runs at 572 GB/s: none. Headroom is in
  hc_read (mHC stream read, hc_read_e4m3: 27.8 us a call x 97 = 2.7 ms a step, 16%; it streams ~6.5 MB of E4M3 mixing
  matrices in two launches, ~38% of bandwidth, so ~1.5 ms a step is recoverable) and in the small GEMMs (N=512/1280,
  K=640: 5-10 us, 180-330 GB/s, launch-bound; fusion is the lever).
- So for Flash-Next the gfx1201 rows to write are hc_read (one launch, both matrices streamed at bandwidth) and fused small
  GEMMs, not moe_gemm_q. For the 35B, moe_gemm_q stays the target (step 2 baseline still to run).

2026-10-21 prefill map (rad-kbench --max-tok 4096 on the R9700, Flash-Next, one 4096-token chunk, libr4d at real shapes):
hc_read 97 x ~1.28 ms = 124 ms (41-44 TOPS, ~11% of the int8/fp8 matrix peak: two products with a 320-wide inner rank);
int8 dense GEMMs (gemm_i8a8_tiled) 216 calls, 130 ms at 155-195 TOPS (~half of peak); hc_write 32 ms; gdn_chunk_scan
29 ms; moe_gather 17 ms. (gdn_conv_update's 5.3 ms a call is the decode kernel timed at 4096 tokens; prefill uses
gdn_conv_prep, which kbench did not time here, so that row is excluded.) Prefill target: an hc_read row for large step
sizes on WMMA (the bc-gfx12 plugin, same A/B harness), then the int8 tiled GEMM.
2026-10-21 hc_read follow-up: the bc-gfx12 plugin skeleton (PR to follow, branch feat/bc-gfx12-hc-read) is selected by
radiance for decode-sized hc_read and, delegating to libr4d, times identically (27.6 us at M=4, 37.0 at M=16), so the A/B
harness is rad-kbench --kernels libbcgfx12:libr4d:libref --op hc_read on the real container. Reading libr4d 1.9.0's
launcher shows the grid work is already done upstream (column tile narrowed to fill the card, 81 blocks at one row, a
measured "sharp optimum"; a tiled split-K path from kDecMinRows). The remaining gap is mostly the latency-bound norm
prologue (~6.7 us of kernel A by upstream's own note), so a realistic gain is ~3-5 us a call (~0.4 ms a step, ~2%), not
the ~1.5 ms first estimated. Deprioritised behind the prefill map and the 35B MoE baseline.

2026-10-20 bc-gfx12 hc_read prefill row, stage 1 (delegate-only, M >= 17): proves plugin selection/layout plumbing ahead of a WMMA redesign (same two-stage approach as the decode row and moe_gemm_q). rad-kbench --kernels libbcgfx12:libr4d:libref --op hc_read on the real Flash-Next container, --max-tok 4096: libbcgfx12 now runs 2 of its 3 declared rows (decode + prefill), libr4d's own hc_read_e4m3 row runs 0 of 429 cases (fully delegated through ours), 140 of 140 cases pass. Merged (#358).

2026-10-20 PF2 K-split scaling cliff, root-caused then refuted: hc_read's PF2 form scales near-linearly 512->2048 rows but 2048->4096 costs 2.36x instead of 2x. Hypothesis was hc_down_kchunk's K-split heuristic (ks) using the wide form's 128-row tile while PF2's own down kernel uses a 64-row tile, going 2->1 right at this boundary. Tested by patching a throwaway scratch build of libr4d (BC_HC_PF2_KS_FORCE env override in both hc_pf2_launch and its scratch sizer hc_part_bytes; never touched the real pinned tree) and forcing ks=1/2/3/4 at M=4096: full hc_read call got slightly WORSE with higher ks (1267->1289->1295->1307 us), and the down-product kernel timed in isolation (stage-stop instrumentation) was unchanged (565 vs 562 us at ks=1 vs 2). Conclusion (confirmed with GPT dev-agent review): K-split count is not the cause; natural grid is 64 workgroups either way (32x2 at 2048, 64x1 at 4096). Per-stage breakdown at 2048->4096: prep 99->176us (1.8x, fine), down-product 183->566us (3.1x, the sole outlier), up+mix 215->475us (2.2x, close to fine), quant flat. Root cause inside hc_down_pf2_kernel unconfirmed (needs register/occupancy profiling; rocprofv3 crashes radiance's AQL backend on this box). Disposition: do not pursue a PF2 K-split patch; proceed with the FP8 WMMA hc_read redesign (down+up products) as the higher-value prefill target, per GPT's recommendation.

2026-10-21 hc_read prefill FP8 WMMA: scope correction. The bc-gfx12 plugin model (reuse libr4d's EXPORTED hooks, delegate the rest) does not fit this op: the down-product's activation operand (N, built by libr4d's internal grouped-rmsnorm + low-rank-gate + stream-mean prologue) is not an exposed operand and has no export. A plugin row would have to reimplement hc_read's entire ~3700-line math stack to get N, not just swap the matmul. Right-sized approach: a radiance engine PATCH (same system as rad0001-rad0006) modifying libr4d's hc_pf2_launch directly -- fuse FP8 (E4M3) quantisation of N into the existing prep kernel (hc_pf2_prep_kernel), add an FP8 WMMA down-product kernel, gated by a flag, keep the up-product and epilogue as they are.

Hardware-verified building blocks:
- __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 exists and runs correctly on the R9700 (gfx1201); rocWMMA 7.2.4 already has a working specialization (wmma_impl.hpp:1053).
- The SAFE path is rocwmma::fragment's high-level API (fragment<matrix_a/matrix_b/accumulator,...>, load_matrix_sync, mma_sync, store_matrix_sync), not the raw builtin with hand-rolled lane mapping -- empirically proved correct via 5 independent single-nonzero-element probes. matrix_a row_major, matrix_b col_major, both ldm=K.
- W_down's existing stored layout (row-major [lr=320,HN=10240]) is byte-identical to what matrix_b's col_major load wants with ldm=HN -- no weight data rearrangement needed.
- GPT dev-agent design review confirms the builtin/tile shape independently but its kernel skeleton uses the raw builtin with a lane-fragment layout it explicitly flags as unverified; the rocwmma::fragment path was used instead.

2026-10-21 standalone kernel validated correct (tools/lab/radiance/hc-down-fp8-validate.hip, at the real shape M=64/N=320/K=10240, hc-down-fp8-isolate.hip for a narrower large-stride check): first version failed hard (max rel diff 18380) from two real bugs this harness caught -- (1) a __shared__ buffer declared inside a per-wave loop was actually one block-wide allocation, so all 8 concurrent waves smashed each other's partial sums; (2) the 128-wide scale-block width (KB) was used AS the WMMA K-step (16), so the loop only ever touched K at stride 128, consuming just 1,280 of 10,240 K elements (1/8 of the data) while silently 'succeeding'. After both fixes: against the quantised-operand reference, max rel diff 1.4e-3 (fp32 accumulation-order noise, i.e. correct); against the original fp32 operands, RMS relative error 3.58%, matching GPT's 2-4% heuristic estimate for double-FP8 quantisation error. Run on the R9700 (ROCR_VISIBLE_DEVICES=2).

Not started: the actual radiance patch (radNNNN_*, package with patch.py/toml/README/SUMMARY/tests) wiring this validated kernel into libr4d's hc_pf2_launch with the fused activation quantisation. Next session: author the patch package (rad0001-0006's shape), wire the kernel in behind a default-off flag, then rad-kbench correctness at rows 17/32/64/.../4096, then a Flash-Next clone of qwen35b-helper-kld.json's quality_only harness (same thresholds: max_kld_mean_increase 0.02, median 0.01, p99 0.2, top1_drop 1.0, ppl_ratio 0.05).

2026-10-21 hc_read prefill FP8 WMMA: second, deeper scope finding. hc_pf2_prep_kernel does NOT build or store N (the gated/normalised/meaned activation) anywhere -- it only computes the per-(row,stream) rmsnorm scale (scaleout) and decodes the WEIGHT into fragment order (plane/udec). N itself is built inline, per-K-step, inside hc_down_pf2_kernel's own fetchA/stageA closures: raw h decoded, multiplied by the per-stream norm scale (a single scalar for the WHOLE stream width, not per-128-of-K) and the per-channel gate weight, staged as bf16 into a double-buffered LDS tile (kTAP-wide, kTKT=64-element slices), consumed immediately by frag()+mma(). There is no full-precision N buffer to attach an extra FP8-cast-and-scale write to, and no existing per-128-of-K activation scale to reuse (the weight's own scale IS per-128, but the kernel's native streaming granularity is 64, half that).

Consequence: an FP8 down-product needs either (a) a new standalone quantisation pass that computes N in full (replicating fetchA's per-stream-scale/gate-weight math) and writes (Aq,Ascale) to scratch before the down-product runs, or (b) restructuring hc_down_pf2_kernel's slice loop to defer quantisation by one 64-wide slice, accumulating a running max over two slices (128 elements) to match the weight's existing scale granularity. Both are real changes to an already highly-tuned, double-buffered pipeline that explicitly guarantees bit-identical reduction order against the wide and per-row kernel forms (r4d_selftest checks this invariant) -- the bf16 fallback path must keep it exactly.

Disposition: this is materially bigger than 'swap the matmul' and risks silently-wrong production inference output if rushed -- the exact failure mode the standalone validation harness already caught twice (a shared-memory race, and a loop silently consuming 1/8 of K) in a much simpler synthetic test. Not attempting the in-place r4d_hc_bf16.hip patch without the same kind of hardware-iterative validation the standalone harness used. Next session: build a SEPARATE standalone harness that first replicates fetchA's N-construction math (reading real h/scaleout/w from a real hc_read call, e.g. via a throwaway scratch rebuild of libr4d that dumps N to a file) to get a ground-truth N to validate option (a) or (b) against, before writing either into the real kernel.

2026-10-21 ground-truth N extracted and quantised, real data not synthetic. Built a throwaway debug build of libr4d (scratch source copy, BC_HC_NDUMP_ROW env hook added to hc_down_pf2_kernel's stageA closure, dumping one real row's full N (10240 floats, post norm-scale and gate-weight, pre-WMMA) to a file during a live rad-kbench hc_read call on the real Flash-Next container; never touched the pinned tree). Confirmed non-degenerate (all 10240 values populated, range -23..+29).

Real (not approximated) E4M3 round-trip on this ground-truth data, per-128-of-K amax/448 scale (tools/lab/radiance/ndump-quant-check.hip, using the actual float8_t cast rather than a hand-rolled approximation of E4M3's non-uniform step): RMS relative error 2.53%, max abs error 0.78, 94 of 10240 codes saturated (0.9%). Matches GPT's 2-4% heuristic and the synthetic validation harness's 3.58% end-to-end (which also includes weight quantisation). The saturation rate is a real signal worth tracking in the eventual KL check -- a small fraction of channels may see outsized error.

This directly answers the gap from the prior scope finding (no existing per-128 activation scale, no stored N buffer): the ndump technique proves out the ground-truth-extraction half of the plan (option (a)/(b) from the prior note) and can be reused to validate the eventual restructured kernel's actual output against this same real row, not just synthetic data. Still not started: the kernel-side restructure (deferred quantisation across two 64-wide slices to match the weight's 128-wide scale, or a separate quantisation pre-pass) and the actual radiance patch package.

2026-10-21 hc_read prefill FP8 WMMA: third scope detail, found before writing any kernel code (analysis only, no source touched this round). The per-128-of-K activation amax (needed for the FP8 scale) is not a simple per-thread running carry across hc_down_pf2_kernel's two closures (fetchA/stageA). Thread layout: kKV=kTKT/8=8 threads share each row's 64-wide slice (kTKT=64), each owning a fixed 8-element window (k8a) that does not change across calls; k0 advances by kTKT=64 each slice, so a single thread's two consecutive slices cover two DISJOINT 8-wide windows 64 apart within one 128-wide scale group (its own 16 of 128 elements), not an overlapping or complete view. Getting the group's amax right requires a real cross-thread reduction (across the kKV=8 threads at a given row, over 2 consecutive slices = 16 threads' worth of 8-element windows) at the point the second slice of a group completes, not just local register state -- a wave/workgroup-level reduction added to an already carefully-scheduled double-buffered loop (sA, barriers, prefetch-ahead bp[]).

Disposition: stopping live modification of r4d_hc_bf16.hip here. This is correctness-critical, intricate GPU kernel engineering (a single misplaced barrier or thread-mapping error reintroduces exactly the silent-wrong-output failure mode the standalone harness already caught twice) that deserves dedicated, higher-effort, iterative-on-hardware work, not a low-effort pass layered on an already long session. Next concrete step, precisely specified: add the cross-thread (kKV=8 threads x 2 slices = 16 threads) amax reduction at the point a 128-group's second slice's stageA call completes, validate the derived (scale, quantised codes) against tools/lab/radiance/ndump-quant-check.hip's offline recomputation on the SAME ground-truth row (bit-identical scale, matching quantised codes) before touching the WMMA swap itself. The ndump/ndump-quant-check tooling built this session is reusable for that validation.

2026-10-21 cross-thread amax reduction, the specific blocker from the prior note, validated and resolved. tools/lab/radiance/hc-pf2-quant-reduce.hip isolates the bookkeeping (kKV=8 threads sharing each row's 64-wide slice, a tree reduction over those 8 threads at the point a 128-wide group's second slice completes, via LDS) against the offline ground-truth recomputation on the real dumped row (ndump.bin): 0 mismatches across all 80 scales and 10240 codes, bit-for-bit exact.

tools/lab/radiance/hc-down-fp8-e2e.hip then chains this proven quantiser into the proven WMMA down-product kernel (hc-down-fp8-validate.hip), on the real ground-truth N (weight side still synthetic -- no real-weight dump built yet): 2.58% RMS error against a reference built from the same real N and the synthetic weight's own quantised reconstruction, matching the standalone activation-quantisation measurement (2.53%) almost exactly. The two previously-separate, separately-proven pieces compose correctly on real data with no compounding surprise.

State of the FP8 WMMA down-product, activation side: fully validated (ground truth -> quantise -> WMMA -> correct, on real data). Remaining before this can go into r4d_hc_bf16.hip as a radiance patch: (1) a real-weight ground-truth extraction (same ndump technique, for W_down instead of N) to validate the weight side the same way: (2) porting the proven quantise_n + hc_down_fp8 kernels into the actual double-buffered, prefetch-ahead hc_down_pf2_kernel structure (this validation used separate, simpler kernels -- the real kernel's tight scheduling still needs care, though the core math/bookkeeping is now proven); (3) the patch package itself (patch.py/toml/README/SUMMARY/tests) and the KL harness run. Given the preceding two blockers are now resolved, this is materially closer to ready than the prior note's 'needs dedicated engineering time' assessment -- though still real work, not a one-line change.

2026-10-21 second-model search for the bc-gfx12 moe_gemm_q decode row's quality evidence: two real candidates tried and fetched, both genuine architecture dead ends (not missing metadata, not disk/bandwidth issues):
- Gemma 4 E4B (official google/gemma-4-E4B-it, 16GB bf16 safetensors, downloaded and converted via rad-convert --arch gemma4): rad_arch_declare refuses outright -- 'this checkpoint uses per-layer embeddings, shared KV layers, a double-wide MLP or attention biases (the E2B/E4B shape); this plugin serves the 26B-A4B and 31B shape'. The radiance gemma4 plugin does not implement the E4B variant's specific architecture features.
- Qwen3-30B-A3B-FP8 (official Qwen/Qwen3-30B-A3B-FP8, 31GB fp8_e4m3 compressed-tensors, downloaded and converted via rad-convert --arch qwen35moe): declares further than Gemma4 did (device/recipe/quant all matched -- fp8_e4m3 is exactly what qwen35moe@fp8_e4m3 expects) but fails at 'the container does not describe the linear layers (key heads 0, value heads 0, key width 0, value width 0, conv width 0)' -- qwen35moe requires hybrid linear-attention (delta-net-style) layers that this plain-generation Qwen3 MoE does not have. Genuine architecture-family mismatch, not a metadata gap (a rope.dimension_count fix got past an earlier, different error, but this one is structural).

Both downloads (47GB combined) fetched, tested, and cleaned up (disk back to 140GB free on ${BC_DATA_ROOT}).

Finding: radiance's architecture plugin set here is narrowly tuned to the specific model families already in use (qwen35/qwen35moe needs hybrid linear attention like the 35B-A3B and Flash-Next have; qwen4exp is Flash-Next's own; gemma4 excludes the public E4B shape). The already-tested 35B-A3B and Flash-Next containers use non-standard version naming (3.6, 3.8) suggesting pre-release/non-public checkpoints this project has specific access to -- finding a DIFFERENT public MoE model that both fits an existing radiance architecture plugin and uses fp8a8-quantised experts is a real search problem, not a quick second download. Disposition: stopped after two genuine, well-tried attempts rather than keep guessing at a third; this needs either owner guidance on which model family to try next (one they know fits), or accepting single-model evidence for this specific case (an off-by-default, easily-reverted flag), or extending a radiance architecture plugin (real engine work) to broaden what's convertible.

2026-10-21 OWNER DECISION: single-model (35B) evidence accepted for BC_GFX12_MOE_GEMV, after two genuine, well-tried second-model attempts (Gemma 4 E4B, Qwen3-30B-A3B-FP8) both hit real architecture mismatches in radiance's plugin set, not quick-fixable. Both downloads (47GB) fetched, tested, confirmed as genuine dead ends, and deleted; owner: 'im ok with 35b only'. Flag flipped to default on (component.toml status validated, kernel source default inverted), citing the existing decode-time check (99.06% top-1, KL mean 3.0e-3/p99 5.7e-2 against libr4d) as the accepted, now single-model-sufficient evidence. Off switch (BC_GFX12_MOE_GEMV=0) retained.

## Change Log

- 2026-10-20T12:00:00+00:00 (created-by): Created by agent
- 2026-10-21T03:56:32.097581+00:00 (updated-by): Updated: section:notes
- 2026-10-21T03:59:24.015049+00:00 (updated-by): Updated: section:notes
- 2026-10-21T04:03:10.125847+00:00 (updated-by): Updated: section:notes
- 2026-10-21T04:04:49.861673+00:00 (updated-by): Updated: section:notes
- 2026-10-21T04:16:43.437145+00:00 (updated-by): Updated: section:notes
- 2026-10-21T04:56:46.675319+00:00 (updated-by): Updated: section:notes
- 2026-10-21T05:00:27.165068+00:00 (updated-by): Updated: section:notes
