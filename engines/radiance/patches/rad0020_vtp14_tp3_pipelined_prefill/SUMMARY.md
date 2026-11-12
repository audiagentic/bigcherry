# rad0020_vtp14_tp3_pipelined_prefill

**Status:** untested
**Plan item:** VTP14

Scope: the fp8 Qwen plugin's pipelined prefill as configuration: `--pipelined-prefill auto|on|off` and
`--pipelined-prefill-min-rows N`.

## Summary

`qwen35_fp8` pipelines a wide one-sequence prefill step: the rows run as two pieces, and each block's all-reduce for
one piece runs on the second lane beside the other piece's work. Upstream fixes it at two ranks from 1024 rows. This
patch makes the mode and the narrowest step options; `auto` adds three ranks under `--tp-ownership`.

## Outcome

Untested (not promoted). The pipelined step computes the serial step's bytes, so output is unchanged. MXFP4 TP3,
4096-token prompts: time to first token −5% (VTP14.md).

## Detail

See README.md.
