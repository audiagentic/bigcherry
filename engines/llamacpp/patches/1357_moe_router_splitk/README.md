# 1357_moe_router_splitk

Not promoted. Mechanism is in SUMMARY.md: the marked MoE router matmul runs through a split-K F32 GEMM with a
fixed-order reduction instead of rocBLAS SGEMM. Off by default; `BIGCHERRY_MOE_ROUTER_SPLITK=1` turns it on.

## Evidence

- Gate (kernel census of the released build, run `census-rel1-d24576`, Flash-Next, 75 chunks of 512 tokens): the
  router SGEMM runs once per layer per chunk on each target card, 0.165 ms per call on an RX 7900 XTX
  (`Cijk_..._MT64x64x8`) and 0.432 ms on the R9700 (`Cijk_..._MT16x16x16`). On the R9700, 3300 of its 3600 calls are
  followed directly by `topk_moe_cuda`, the rest by the unfused top-k kernels.
- Weight: `ffn_gate_inp.weight` is F32 `[2560, 512]` in all 48 layers of the UD-IQ4_XS model.

## Still to show

Offline mechanics and patch-lint, then on hardware: the activation marker, kernel time of the two new kernels against
SGEMM, an ABBA on one binary with `tools/lab/flash-next/queue-env-ab.sh` (flag on / off), and identical greedy output
at 8K, 24K and 98K. A changed output rejects the patch: expert selection must not move.

## Promotion record

Promotion evidence, profile-evidence tier (QFP18), pin b11474, 2026-10-10.

Model: Qwen3.8-Flash-Next UD-IQ4_XS (production profile: 2x RX 7900 XTX + R9700 tensor split, MTP drafter on the RX 6900 XT, ctx 245760, f16 KV, ub512)
Model: Qwen3.6-35B-A3B (MoE, 256 experts; tensor split, no draft)

Mechanics: offline tests for 1357 pass (4 tests), patch-lint clean, composition with the production set clean (CI offline-patch and compile-only on PR #92).
Activation: BIGCHERRY_PATCH_HIT patch=1357_moe_router_splitk with the flag on only. Flash-Next: k=2560 experts=512 tokens=512 parts=8 (runs rse1-*). Qwen3.6-35B: k=2048 experts=256 tokens=1661 parts=8 (run om-q36-1357-2-B); absent in the flag-off arms. Gemma 4 26B (no router): absent in both arms.
Identity: not bit-identical on Flash-Next (the router scores are summed in another order and feed a discrete top-k choice of experts), so the greedy text differs. Against a CPU f32 reference at 8K, 24 probes, no drafter (run rsk1-fid): production top-1 23/24, mean TV 0.0831; with 1357 top-1 23/24, mean TV 0.0799. On Qwen3.6-35B and Gemma 4 26B the greedy text is identical with the flag on and off.
A/B: one binary (b-metamem-rse1 = production + 1357 + 1358), BIGCHERRY_MOE_ROUTER_SPLITK=1 against default (off), Flash-Next at 24K, four requests, ABBA each (queue-decode-acceptance.sh): prefill +3.2% pooled (+3.1%, +2.7%, +4.5%, +2.5%), consistent in all four.
No-regression: Qwen3.6-35B-A3B and Gemma 4 26B, tensor split, no draft, ABBA off against on (queue-promotion-followups-2.sh step 8): text identical, prompt and decode speed unchanged.

Scope: profile-scoped. The flag stays default off; the patch only takes a matmul that build_moe_ffn marks as the router (F32, at least 64 experts, 64 tokens and 512 of K).

## Flash-Next at 24K, off against on, four requests

| Request | Prefill t/s, off -> on | Decode t/s, off -> on | Draft acceptance, off -> on |
|---|---|---|---|
| 1 | 1308.2 -> 1348.3 (+3.1%) | 72.4 -> 72.6 (+0.2%) | 58.2% -> 58.8% |
| 2 | 1312.1 -> 1347.8 (+2.7%) | 73.7 -> 58.6 (-20.4%) | 59.0% -> 43.6% |
| 3 | 1282.5 -> 1340.8 (+4.5%) | 68.8 -> 69.5 (+1.0%) | 58.1% -> 55.8% |
| 4 | 1307.6 -> 1340.5 (+2.5%) | 64.5 -> 68.2 (+5.7%) | 51.2% -> 54.5% |
| pooled | +3.2% | 69.8 -> 67.2 (-3.8%) | 56.6% -> 53.2% |

Decode is equal or better with the patch in three of four requests; the pooled loss is request 2. The same request
lost 20.6% decode when 1350 was switched OFF in the 2026-10-09 run of this check, so it moves that much whenever the
text changes, in either direction; it is recorded here as a known limit of the evidence, not explained away.

Earlier two-build and one-binary runs (2026-10-09, runs rsk1-*, rse1-*): prefill +2.2% to +2.7% on top of 1358 at
8K / 24K / 98K.

## Other models, off against on

| Model | Prompt t/s, off / on | Decode t/s, off / on | Text | 1357 marker |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 4072.6, 4161.0 / 4115.7, 4095.9 | 116.6, 116.9 / 116.5, 116.9 | identical | on arm only |
| Gemma 4 26B | 3286.9, 3474.8 / (flag on, no router) | 93.7, 93.8 / unchanged | identical | none |

The prompt there is about 1,700 tokens, so these are no-regression figures, not a gain.

## Native llama.cpp baseline

With `BIGCHERRY_MOE_ROUTER_SPLITK` unset the router matmul goes to rocBLAS SGEMM exactly as native llama.cpp b11474
sends it, on the same binary. The off arm is therefore the native behaviour for the code this patch changes; the
rest of the build is the BigCherry production baseline in both arms.
