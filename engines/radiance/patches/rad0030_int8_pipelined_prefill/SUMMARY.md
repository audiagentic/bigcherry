# rad0030_int8_pipelined_prefill

**Status:** untested
**Plan item:** VTP37

Scope: tensor-parallel prefill of a model whose trunk norms write rotated int8 codes (the w8a8 Qwen plugin on kernel
libraries that serve the `_had_quant_i8g` forms).

## Summary

The pipelined prefill (two pieces a step, every block's all-reduce and the next norm on a second lane) now runs on a
rotated-int8 trunk when `--pipelined-prefill on` is given. Before this such a trunk always prefilled serially, because
the second lane's norm issued its op with the E4M3 code pair instead of the int8 pair the trunk reads. Under `auto`
(the default) nothing changes.

## Outcome

Untested (not promoted). Built for VTP37; measurements are in the plan item.

## Detail

See README.md.
