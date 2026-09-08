# 1251_nro02_allreduce_fused_residual

Plan `NRO02`; state `untested`; requires `1250_nro01_allreduce_q8_wire`.

This draft materializes only the finish-kernel arithmetic required for residual fusion. It does **not** yet extend the Meta communication ABI, match a graph pattern, or clear an ADD compute flag. Those operations can silently remove required work if the matcher is wrong, so they remain blocked on negative graph fixtures.

Two helpers are added: one for ordinary exact/BF16-style finish arithmetic and one for Q8_0 rank blocks. They preserve the selected wire representation's arithmetic and add a same-type residual. NRO02's future control and subject must use the same wire mode; Q8 compression is not part of the NRO02 causal claim.

Before validation-ready state, implement: optional communicator fused-add entry point, exact Meta matcher (reshape-only chain, one consumer, F32/same shape, mirrored residual/output), per-rank tensor mapping, provider-success gating, and skip-node restoration. See NRO02 plan and `TESTING.md`.
