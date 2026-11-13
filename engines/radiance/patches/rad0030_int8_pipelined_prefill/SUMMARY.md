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

Untested (not promoted). VTP37, w8a8 Qwen3.8-27B on XTX, XTX and R9700 (TP3), 2x ABBA against the build without it:
one-prompt time to first token -17% to -21%, four 4096-token prompts -16%, 4/6/8 streams +7.4% / +7.8% / +8.4%.
A pipelined step is not bit-identical to the serial step at every prompt length (as on an E4M3 trunk): see README.md.

## Detail

See README.md.
