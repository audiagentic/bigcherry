# 1286_draft_local_shared_tensors

**Status:** untested
**Plan item:** none

## What it does

Copies the target tensors a DFlash/DSpark draft borrows (`tok_embd`, `output`, `output_s`) onto the
draft's own device when they live in the target's tensor-split (meta) buffer, so DFlash/DSpark run with a
`-sm tensor` target. Without it the draft aborts with "pre-allocated tensor (output.weight) in a buffer
(Meta()) that cannot run the operation". Layer-split and single-device targets copy nothing.
