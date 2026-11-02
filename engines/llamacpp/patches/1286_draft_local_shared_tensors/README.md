# 1286_draft_local_shared_tensors

## Summary

Copies the target tensors a DFlash/DSpark draft borrows (`tok_embd`, `output`, `output_s`) onto the draft's own device when they live in the target's tensor-split (meta) buffer, so DFlash/DSpark run with a `-sm tensor` target. Without it the draft aborts with "pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation".

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
