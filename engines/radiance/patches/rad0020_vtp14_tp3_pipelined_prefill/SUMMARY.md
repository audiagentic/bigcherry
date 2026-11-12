# rad0020_vtp14_tp3_pipelined_prefill

**Status:** untested
**Plan item:** VTP14

Scope: the fp8 Qwen plugin's pipelined prefill at `--tp 3` under `--tp-ownership`.

## Summary

`qwen35_fp8` pipelines a prefill step of 1024 rows or more: the rows run as two pieces, and each block's all-reduce
for one piece runs on the second lane beside the other piece's work. Upstream turns this on only at two ranks, so
native TP3 prefilled serially while the TP2 baseline pipelined. This patch also turns it on at three ranks when an
ownership profile is in force.

## Outcome

Untested (not promoted). The pipelined step computes the serial step's bytes, so output is unchanged; the effect to
measure is time to first token on long prompts.

## Detail

- Two edits in `arch/qwen35_fp8/qwen35_fp8.cpp`: the `pipe_on` condition, and the note an int8 trunk prints when it
  stays serial (its boundary norms write the E4M3 pair only).
- The all-reduces keep their serial declaration at `overlap` 1; selection picks their kernels by world, size and
  peer access, as for the serial step.
