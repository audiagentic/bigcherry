# 1355_hc_post_gate_fuse

QFP35 residual fusion against llama.cpp b11474, composed after production patches 1313 and 1344.

Default: **on**. Control: `BIGCHERRY_HC_POST_GATE_FUSE=0`.

Build experiment `hc-post-gate-fuse`, then run one binary with full process separation:

```bash
BIGCHERRY_PATCH_TRACE=1 \
AB_ENV="BIGCHERRY_HC_POST_GATE_FUSE=0" \
tools/lab/flash-next/queue-env-ab.sh qfp35-hcpost <build-run-id> 8192 24576 98304
```

A = fused/on; B = composed production/off. Require the patch-hit marker in A, complete prefill timing separation to
claim a win, unchanged decode control/acceptance, and identical greedy md5 A/B at all three depths. Because this is
intended to reproduce the same F32 expressions and HC reduction order, any greedy mismatch is a failure pending
fidelity diagnosis, not an accepted equivalence result.

## Promotion record

Promotion evidence, profile-evidence tier (QFP18), pin b11474, 2026-10-10.

Model: Qwen3.8-Flash-Next UD-IQ4_XS (production profile: 2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT, ctx 245760, f16 KV, ub512)
Model: Gemma 4 26B (tensor split, no draft; no-regression only, the path does not engage)
Model: Qwen3.6-35B (tensor split, no draft; no-regression only, the path does not engage)

Mechanics: offline tests and patch-lint pass; composition with the production set clean.
Activation: BIGCHERRY_PATCH_HIT marker appears only in the fused arms on Flash-Next; 0 hits on Gemma 4 26B and Qwen3.6-35B (the patch only matches Qwen4Exp hyper-connection gates: SCALE -> SIGMOID -> SCALE consumed inside DSV4_HC_POST).
Identity: greedy text identical in every run at 8K / 24K / 98K on Flash-Next, and on Gemma 4 26B and Qwen3.6-35B.
A/B: one binary (b-metamem-hcf1 = production + 1355), fused (default) against BIGCHERRY_HC_POST_GATE_FUSE=0: decode +0.7% at 8K, +0.5% at 24K, +0.5% on a second request, pooled over the rounds below; single rounds overlap. Prefill unchanged.
No-regression: Gemma 4 26B and Qwen3.6-35B, tensor split, no draft, default against off: text identical, prompt and decode speed unchanged (the path does not engage); Qwen3.8-27B on two RX 7900 XTX (prod27b-ab.sh, run nr27-1355): no change, text identical.

Scope: profile-scoped. The mechanism exists only in Qwen4Exp graphs, so default-on cannot change other models.

## Flash-Next, first three ABBAs per depth (2026-10-09)

Decode t/s, fused against off; fused arm in both position patterns (r1: runs 1 and 4; x1, x2: runs 2 and 3).

| Depth | r1 | x1 | x2 |
|---|---|---|---|
| 8K | 87.3 / 89.7 vs 87.6 / 87.1 | 87.3 / 90.2 vs 83.2 / 84.8 | 85.4 / 87.9 vs 85.8 / 86.2 |
| 24K | 72.5 / 74.1 vs 72.5 / 71.9 | 72.1 / 73.6 vs 71.8 / 72.6 | 72.1 / 73.8 vs 70.7 / 72.6 |
| 98K | 70.7 / 71.6 vs 70.8 / 70.4 | 71.6 / 70.4 vs 68.9 / 70.2 | 70.8 / 69.8 vs 70.1 / 70.9 |

## Flash-Next, more rounds (2026-10-10, `queue-promotion-followups-2.sh` step 6)

Longer decode (about 1,400 accepted of 1,950 drafted), four ABBAs at each depth, two on a second request.

| Workload | Fused, mean decode | Off, mean decode | Change |
|---|---|---|---|
| 8K, 8 runs per arm | 94.7 t/s | 94.0 t/s | +0.7% |
| 24K, 8 runs per arm | 83.8 t/s | 83.4 t/s | +0.5% |
| 24K second request, 4 runs per arm | 64.6 t/s | 64.3 t/s | +0.5% |

Draft acceptance is the same in both arms (for example 1381-1384 of 1987-1996 at 24K). The arms' ranges overlap in
every workload; the fused mean is higher in all three.

## Other models (step 7)

| Model | Prompt t/s, default / off | Decode t/s, default / off | Text | Marker hits |
|---|---|---|---|---|
| Gemma 4 26B | 3443.4, 3435.9 / 3456.0, 3508.0 | 93.7, 93.4 / 93.4, 93.0 | identical | 0 |
| Qwen3.6-35B | 4060.0, 4077.8 / 4070.3, 4072.0 | 116.9, 117.0 / 116.2, 117.0 | identical | 0 |

## Decision

Promoted as an output-identical small decode win on Flash-Next with no regression anywhere. The claim rests on the
pooled result across the rounds above, not on complete separation in one ABBA.

## Native llama.cpp baseline

With `BIGCHERRY_HC_POST_GATE_FUSE=0` the gate runs as the separate SCALE, SIGMOID, SCALE nodes that native llama.cpp
b11474 emits for this graph, feeding the composed 1313/1344 consumer, on the same binary. The off arm is therefore
the unfused behaviour for the code this patch changes; the rest of the build is the BigCherry production baseline in
both arms.
