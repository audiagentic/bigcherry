# Testing — NRO06 adaptive MTP controller

## Lane applicability
1255 adds only the pure `bigcherry_nro06_adaptive_mtp` depth-controller state machine beside MTP; it is intentionally not instantiated or wired into runtime. Therefore it cannot change gfx1100 `-sm tensor` Qwen3.8-27B-Q8_0 MTP `n_max=4` decode, verify batch 5, or prefill by itself. **Applicable to this lane: no — not applicable to this lane unless composed with runtime wiring such as 1268.**

## Hardware-free
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1255_nro06_adaptive_mtp_depth`
- Unit-test reset floor/cap, depths, exact climb thresholds, drop pressure `max(depth*5,20)`, full/partial/zero acceptance, no drop below floor, `n_draft<=0`, long mixed traces, apply/idempotence and missing-anchor fail-closed behavior.

## Activation
1255 has no runtime activation marker because it is deliberately unwired. If standalone runtime activation is ever added, propose exact marker: `BIGCHERRY_PATCH_HIT patch=1255_nro06_adaptive_mtp_depth path=adaptive_mtp_controller`. Do not add/fire it merely for compilation; activation must mean the controller influences draft depth. In current standalone tests, absence is expected.

## Correctness
Standalone: runtime correctness is N/A because behavior is unchanged. When composed with wiring, require MTP logprob max abs diff <= `5e-4` vs control, report greedy divergence, and require draft-acceptance parity to standard control `0.90101` with accepted/drafted counts.

## Performance
Standalone performance is N/A: an order-balanced `bigcherry ab-benchmark --server-config` would intentionally measure no runtime variable. For a wired composition, use one-variable order-balanced A/B and report `pp1024`, `pp4096`, `tg512`, `tg2048` with Mann-Whitney.

## Promotion
1255 alone should be promoted only on its controller/mechanics contract evidence if such a contract is defined; runtime claims require the wiring patch's contract campaign. Promotion evidence must be `patch-verify-evidence` validated-evidence, never ad-hoc A/B.
