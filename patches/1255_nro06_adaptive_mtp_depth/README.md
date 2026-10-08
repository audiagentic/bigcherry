# 1255_nro06_adaptive_mtp_depth

Plan `NRO06`; state `untested`; pure controller, runtime wiring lives in 1268.

The controller is deterministic and uses accepted/drafted counts only. Each request resets to depth `clamp(3, floor, cap)`; every 32 drafted tokens it drops one level at <=60% acceptance, climbs one level at >=72%, and otherwise holds. The 60-72% dead band prevents oscillation. No clock, allocation address, process-global mutable state, or previous-request state participates.

Rationale for the 2026-10-08 policy change: Flash-Next hardware showed floors 1/2 lose at 8K while floor 2 gains at 98K; floor 3 is near-neutral. Starting at 3 removes the shallow cold start, while sustained low acceptance can still select depth 2 at long context. This is a hardware-directed hypothesis pending the new A/B; the policy itself is source-deterministic.

Fixed `draft-mtp` remains unchanged unless 1268 is composed and `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE>0`.
