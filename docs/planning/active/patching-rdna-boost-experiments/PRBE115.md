---
id: PRBE115
order: 0
plan: patching-rdna-boost-experiments
state: done
created-at: '2026-09-29T23:14:07.176425+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Q8_0 F32-activation MTP-verify kernel for widths 2-5 preserving stock numerics

## 2026-10-12 authoritative audit: terminal rejection of raw-F32 MTP verify widths

**Disposition: DONE.** Patch `1301_prbe115_q8_f32_mtp_widths` is already `rejected`; no new weight-hoist-only kernel, experiment queue, configuration, scheduler or dispatcher is warranted. Preserve validated `1241` on gfx1100 dense Q8_0 **width 1** and the stock Q8_1-activation MMVQ path for widths 2–5. This supersedes the October 8 open-ended single-dequant proposal. Reopen only on materially new correctness-preserving algorithm and measured critical-path evidence.

### Source and numerical contract

- `1241/patch.py::_LOOP_BODY_NEW` specializes `mul_mat_vec_q<..., f32_act=true>`. For each output column `j`, `vec_dot_q8_0_f32` reads raw F32 `y_f32+j*stride_col_y+kbx*qk`, unpacks Q8_0 weights and accumulates F32. `ggml_cuda_mmvq_q8_0_f32_decode` uses the existing launch geometry. The host gate skips `src1_q8_1` allocation and `quantize_row_q8_1_cuda` only for dense Q8_0, width 1, gfx1100 and `!forced.requested()`. The original forced-autotune, MoE, other-type and architecture controls remain intact.
- `1301/patch.py::_GATE_NEW` widens that gate to `1<=ne1<=BIGCHERRY_Q8_F32_MAXCOLS` (maximum 8) with optional gfx1201; `_LAUNCH_NEW` instantiates each width. It neither shares unpacked weights across columns nor preserves the Q8_1 quantizer. The inherited `BIGCHERRY_PATCH_HIT` uses `std::call_once` per width: route evidence, **not** per-step kernel launches, work or transfer counts.
- Stock Q8_0×Q8_1 MMVQ first quantizes F32 activations, then accumulates integer products and scales. Raw-F32 MMVQ instead accumulates products with unquantized activations. For one 32-value block with integer weight 1 at index 0 (zero elsewhere) and activations `[0.001,1,0,...]`, a max-abs/127 Q8_1 reference rounds index 0 to zero while raw F32 retains 0.001 times weight scale. Hoisting the weight dequantization outside the column loop **cannot repair this semantic difference**. This is a mathematical host counterexample, not a reproduced HIP output defect.
- Repeated per-column weight unpack and F32 accumulation can outweigh the removed quantization launch. Do not infer a net win from a missing launch; preserve quantizer/weight work accounting, graph buffer lifetime, MTP draft acceptance, and cross-card reductions. The two XTX cards have no normal P2P; this mechanism does not eliminate their tensor-split reduction.

### Measured BigCherry evidence and consolidation

`QFP21` and `1301/SUMMARY.md` report Qwen3.8-27B Q8_0 dual-XTX, built-in MTP4, 10K/32K ABBA with actual width-2/3/5 route hits. MAXCOLS 2: neutral (~73.1 vs 73.0/73.2 t/s at 10K; ~72.0 vs 72.1/72.3 at 32K). MAXCOLS 3: neutral. MAXCOLS 5: ~64.7–64.8 vs 73.0–73.4 at 10K (~12% regression), ~67.8–67.9 vs 72.0–72.1 at 32K (~6% regression). These historical observations are not four-session confidence intervals or proof about a *different* kernel. The earlier raw-F32 widening changed MTP acceptance (0.90101→0.95580) and greedy output. `1241/SUMMARY.md` separately reports +4.56–4.67% width-1 raw decode (~+3.2% control-adjusted); it is not a verify-width or gfx1201 result.

PRBE115 owns terminal 1301 mechanism disposition. QFP21 owns the historical run results; PRBE113 retains the existing per-width quantizer/kernel profiling dependency. No new profile queue or second MMVQ implementation is created; QFP15, Radiance Flash-Next/n-gram/prefill, MoE handoff, GDN, AllReduce/wire and release work remain untouched. Last independent PRBE115 change was 2026-10-08 06:47 UTC; 1301 patch source moved 2026-10-09; no PRBE115/1301 independent change, PR or queued experiment in the preceding 12 hours (cutoff 2026-10-12 06:08 UTC). Recent BCOP155–167 address other capabilities.

### Bounded re-entry / validation gate

1. Before hardware, a candidate must **preserve stock Q8_1 scale, rounding, saturation, integer accumulation and output conversion**; test 32-element packed blocks including tiny activations, signs, ties, saturation, zero, odd tails and widths 1–5. A raw-F32 weight-hoist-only variant fails this numerical requirement by construction. Follow with backend-ops, fixed-draft full-vocabulary/logits and MTP acceptance comparisons; no false claim of bit-identity from greedy text alone.
2. Reuse a completed, uncontended PRBE113 trace, not an overlapping Radiance GPU queue. Attribute actual non-overlapped `quantize_row_q8_1_cuda` launch time, allocation and bytes per verify width, plus MMVQ and wall time. If eliminating **all** attributable quantization cannot theoretically improve E2E decode by at least 3%, terminate. Concurrent rank kernel-time sums do not establish critical path.
3. Only if both gates pass, implement one default-off change within existing 1241/MMVQ ownership with stock fallback; no new cache/selector. Test `!ids`, forced autotune, gfx1100/gfx1201 gates, gfx1030 negative, graph replay, multi-ubatch, repeated same-process requests and expected work/transfer counts.
4. Promotion requires independent gfx1100 and gfx1201 controls at 8K/32K/98K, widths 1–5, >=4 sessions per architecture and >=10 paired ABBA rounds/session, CI95-low >=3% E2E decode and <=1% regression on width-1/prefill/other-type controls, with full-vocab and identical fixed-draft MTP acceptance. Otherwise close.

### Current upstream and other engines

The newest upstream `mmvq.cu` change since October 1 is [llama.cpp #27828](https://github.com/ggml-org/llama.cpp/pull/27828) (merged October 10): Pascal/NVIDIA `MUL_MAT_ID` routing, not RDNA Q8_0 verify-width. [llama.cpp #21151](https://github.com/ggml-org/llama.cpp/issues/21151) describes a Vulkan RDNA3 K-quant f32-dequant alternative, not a HIP Q8_0/MTP implementation or transferable measurement. [ROCm AITER](https://github.com/ROCm/aiter) lists gfx1100/gfx1201 experimental, with no verified drop-in GGUF Q8_0/Q8_1 replacement for this lane. Adopt nothing.

**Actually validated this audit:** disposable Python max-abs/127 host mathematical model, 3/3 single-column and 2/2 two-column non-equivalence examples; 2/2 historical throughput arithmetic checks. Not an exact HIP rounding/reduction simulation. **Not run:** repository pytest, HIP compile, inference, rocprof or GPU benchmark. New measured improvement: none. Terminal BCOP168.


## Description

GPT ranked candidate #1 (2026-09-30): 1241 proves F32 activation is valuable at width 1, but widening the old RD33 to ne1<=8 changed MTP numerics (acceptance 0.90101->0.95580, greedy divergence), so the widened gate was rejected. Measure stock Q8_1 quantize + MMVQ time by width and its fraction of decode, then design a dedicated verify-width kernel that keeps stock numerical semantics and MTP acceptance. Other ranked candidates to revisit after the profile capture: MMVQ geometry/nwarps for widths 2-5, tensor-split AllReduce latency/payload (after 0860), GDN recurrent/MTP-verify kernel (1254 only helps prefill: K>1, one sequence, n_tokens>K+64), MTP host/sampling/launch overhead.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Backend-reference per width 2-5, full-vocab/logprob tolerance, exact MTP acceptance parity, 4-session performance on 27B Q8_0 dual XTX.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Gated on the profile-first item; do not start kernel work before the per-width decode fraction is known. Combo acceptance change was explained by widened 1241, not 1206 or 1245.

2026-10-04 result: patch 1301_prbe115_q8_f32_mtp_widths (env-gated BIGCHERRY_Q8_F32_MAXCOLS / _RDNA4) widened 1241's Q8_0 F32-act gate to ncols 1..8 (per-column vec_dot_f32_decode, weights re-dequantized per column). Activation proven (PATCH_HIT ncols=2,3,4) on the Flash-Next deployment candidate. Quick screens (24K): width4 48.8 vs 48.3/49.3 ms/step; +RDNA4 48.2 vs 49.3/48.3 -> neutral; acceptance dropped (171/250, 169/256 vs ~174/242) - the same numerics shift this item already recorded for the rejected widening (repeated without reading this item first). Parked. GPT (RV4215): one bounded try of a hoisted single-dequant variant (gate: kernel -20%, verify4 E2E >= 2%), compared via a replayed draft stream so the verify-width mix is controlled; code requested in req_ee409a9b21e74e86. See patching-qwen-flash-next and QFN01 notes.

## Change Log

- 2026-10-08 (triage): Kept pending at P2. Decode shortlist rank #3 behind PRBE113 and PRBE111. 1301_prbe115_q8_f32_mtp_widths state=rejected (patch.toml) after acceptance/numerics drift; no verbatim widening. Profile exact Q8_0 width2..5 share first, then consider fresh single-dequant kernel only with controlled replayed draft stream and identical acceptance, >=2% E2E and kernel -20% gate.

- 2026-09-29T23:14:07.176425+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261003_141341_experimental-decode-kernel-pat_2923
- 2026-10-03T14:13:44.858022+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T15:21:38.477056+00:00 (updated-by): Updated: section:notes
