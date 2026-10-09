# 1241 (RD33 / AMD-MMV-001): dense Q8_0 single-column decode without activation quantization

## Scope

Adds an `f32_act` MMVQ path and `vec_dot_q8_0_f32` helper that dequantizes Q8_0 weights and dots them against the original F32 activation, skipping Q8_1 activation quantization. Eligibility is deliberately narrow: dense/non-MoE Q8_0, `ne1 == 1`, gfx1100/RDNA3_0, and no forced autotune candidate.

The earlier PRBE26 experiment widened the gate to `ne1=1..8`. Hardware evidence showed that this changed the MTP verification path (n_max=4 verify width 5), moved acceptance from 0.90101 to 0.95580, and produced tg2048 -5.65%. That widening has been removed; 1241 now targets plain/single-column decode only.

## Numerics

The path intentionally changes arithmetic from Q8_1-quantized activation/int8-style accumulation to original-F32 activation/F32 accumulation. Stock bit/logprob parity is therefore not the correctness oracle. Promotion requires backend/CPU-reference accuracy for Q8_0 `ne1==1`, plus end-to-end quality guards. MTP verify (`ne1>1`) must remain on the stock path.

## Hardware evidence

Dual gfx1100 / 2x Radeon RX 7900 XTX, Qwen3.8-27B-Q8_0, `-sm tensor`:

- Backend Q8_0 MUL_MAT reference testing previously passed within CPU-reference tolerance, including n=1 shapes.
- Plain non-MTP order-balanced 8-pair A/B: tg512 +4.52% (CI +4.36..+4.67), tg2048 +4.42% (CI +4.33..+4.52), 8/8 pairs faster; pp1024 -0.32%, pp4096 -0.13%.
- The superseded 1..8 gate materially perturbed MTP: acceptance 0.90101 -> 0.95580 and tg2048 -5.65%; this is the reason for narrowing.
- Narrowed-gate activation evidence under `BIGCHERRY_PATCH_TRACE=1` shows only `ncols=1`. Subject hits for ncols 2..8 are forbidden.

## Promotion scope

The supportable performance claim is limited to dual-gfx1100 dense Q8_0 plain non-speculative decode. MTP/speculative verification is a non-activation/control lane, not a positive performance lane. Other architectures, quantizations, MoE/ID matmuls, and wider MMVQ batches are out of scope.

Promotion still requires contract-produced validated evidence: subject-hit/control-miss activation, explicit subject non-activation for ncols 2..8/MTP verify, width-1 backend-reference correctness and broader quality guard, and contract-admitted order-balanced plain-decode performance evidence.
## Promotion evidence (2026-09-30)

Campaign `t-1241n1-gfx1100-s1..s4` (artifacts: `tools/lab/native-vs-patched/runs/rd33n1-campaign/`), 27B Q8_0 positive on GPUs 0,1 with `-sm tensor`, Q6_K 4B single-GPU control:

| session | positive tg128 | pairs faster | Q6_K control |
|---|---|---|---|
| s1 | +4.67% | 10/10 | +1.49% |
| s2 | +4.56% | 10/10 | +1.34% |
| s3 | +4.67% | 10/10 | +1.44% |
| s4 | +4.66% | 10/10 | +1.42% |

- Contract verdict (s4 aggregate): pass. Correctness: 50/50 vs CPU reference. Activation: `BIGCHERRY_PATCH_HIT ... ncols=1` on subject only; none on control; none for ncols 2..8.
- 3-arm reference ladder (s4, tg128 t/s, plain decode): stock 23.115, baseline 23.164, baseline+patch 23.999 (+3.8% vs stock, +3.6% vs baseline).
- Exploratory A/Bs (non-admitted, corroboration only): MTP n_max=4 acceptance identical to control (0.90101) with tg +0.6%; plain decode +4.4% (8/8 pairs).
- Caveats: the +1.4% Q6_K control movement is a systematic build-layout effect, not subtracted by the contract policy; scope limited as in SUMMARY.md. Activation proof is trace-marker based, not kernel profiling.
