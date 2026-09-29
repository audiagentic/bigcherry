# Testing — PRBE52 adaptive MTP wiring

## Lane applicability
Wires 1255's adaptive controller into per-sequence MTP behind `--spec-draft-n-min-adaptive` / `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE`; zero remains disabled. It changes the effective draft cap and consumes real accepted-token feedback. On dual gfx1100 Qwen3.8-27B-Q8_0 with MTP `n_max=4` (verify batch 5 columns), opt-in can directly affect decode/MTP scheduling and throughput; prefill is a control lane. **Applicable to this lane: yes when adaptive mode is enabled.**

## Hardware-free
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1268_prbe52_adaptive_mtp_wiring`
- Unit mechanics: apply/idempotence; missing anchors fail closed; CLI floor validation; disabled value preserves fixed MTP; per-sequence reset/isolation; effective cap follows controller; accepted count updates the correct sequence; composition includes required 1210 and 1255.

## Activation
With `BIGCHERRY_PATCH_TRACE=1` and adaptive mode enabled, require exact marker: `BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING depth=<N> seq=<N>`. It must not fire when adaptive mode is disabled.

## Correctness
Dual 7900 XTX/gfx1100, `-sm tensor`, Qwen3.8-27B-Q8_0, MTP `n_max=4`. Compare control/subject MTP logprobs; max abs diff <= `5e-4`; report any greedy-token divergence. Require draft-acceptance parity to standard server-bench control `0.90101`, recording accepted/drafted counts, adaptive depth trace, and ratio.

## Performance
Order-balanced `bigcherry ab-benchmark --server-config` A/B with adaptive wiring/opt-in as the only variable; report `pp1024`, `pp4096`, `tg512`, `tg2048`, raw pair order/deltas and Mann-Whitney. Prefill should be treated as control; decode/MTP is the target. Correctness divergence voids performance conclusions.

## Promotion
Promotion requires a `PRBE52-ADAPTIVE-MTP-WIRING` contract campaign yielding `patch-verify-evidence` validated-evidence with activation, correctness, depth/acceptance accounting and statistical performance results. Ad-hoc A/B is diagnostic only.
