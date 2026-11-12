# rad0025_multiseq_pipelined_prefill

**Status:** untested
**Plan item:** VTP31

Scope: `qwen35_fp8` and `qwen35_w8a8`'s pipelined prefill (rad0020), for a step of several whole prefill sequences.

## Summary

A step that prefills several prompts at once is cut between sequences, at the boundary nearest half the rows, and runs
as the pipelined step: each block's all-reduce for one piece beside the other piece's work. `RadBatch` gains a host
copy of the step's bounds for the cut. Same bits as the serial step.

## Outcome

Untested (not promoted). Built for VTP31: multi-sequence prefill steps ran serially with ~70% of device time in
collectives at three ranks.

## Detail

See README.md.
