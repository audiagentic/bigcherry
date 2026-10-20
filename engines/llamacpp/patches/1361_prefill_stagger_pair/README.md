# 1361_prefill_stagger_pair: out-of-phase prefill pair staggering (QFP49)

## Summary

`BIGCHERRY_PREFILL_STAGGER=<half tokens>` lets a two-ubatch decode call (A and
B, each `half` tokens) execute the tensor-split Meta backend one subgraph out
of phase, so one pair's compute can overlap the other's AllReduce.

## Outcome

evaluated (shelved, owner decision 2026-10-12): never ran a real prompt
correctly end to end (last hardware run asserted in
`bc_1361_copy_input_pinned`; earlier runs declined on growing inputs or
produced wrong text). No engine run showed any real sum overlapping compute.
Kept on main, default off, in no build, as the record of the mechanism; the
standalone probe and toy-model harness say the staggering idea itself is
sound even though this implementation is not finished.

## Detail

See `SUMMARY.md` in this directory for the full mechanism (the two
graph/allocation sets, the preflight requirements, and what bounds the
mechanism's value from outside-the-engine probes) and the owner's shelving
rationale.
