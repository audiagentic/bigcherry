# Testing — NRO06 adaptive MTP

Unit-test the pure state machine before integration. Cases: reset floor/cap, floor>cap clamp, depths 1..8, exact climb thresholds, full accept reset of drop pressure, partial/zero acceptance pressure, one-level drop, no drop below floor, no update for `n_draft<=0`, and long mixed traces.

After integration, add two independent sequence traces to prove no shared state; reset/new-request tests; fixed-MTP regression test; deterministic final-token parity versus target/reference; requested/accepted work accounting.

Performance comparison must include best relevant fixed depth, not only one weak baseline, and several content classes.
