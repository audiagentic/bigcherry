# default-on

Decides whether a generic BIGCHERRY_FEATURES set (e.g. `hip-q81`, `sched-async`) can be on by default: the same
release binary with every runtime flag at its default (A) vs the set on (B), ABBA, on models other than the one it was
tuned on. Default-on requires no regression in prefill/decode beyond noise and identical greedy text (or f32-reference
agreement at near-ties) on every model checked. Plan: QFP22 (owner 2026-10-05: name sets by scope; default on what
does not regress).
