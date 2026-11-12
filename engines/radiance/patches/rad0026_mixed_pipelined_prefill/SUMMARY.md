# rad0026_mixed_pipelined_prefill

**Status:** untested
**Plan item:** VTP31

Scope: `qwen35_fp8` and `qwen35_w8a8`'s pipelined prefill (rad0020, rad0025), for a mixed step of decode rows and
several prefill sequences.

## Summary

A step that carries decode rows beside two or more prefill sequences is pipelined too. The first piece holds the
decode sequences and the prefill sequences before the cut, the second the rest. Each block issues the serial path's
own decode and prefill launches over its piece. Same bits as the serial step.

## Outcome

Untested (not promoted). Built for VTP31: after rad0025, mixed steps (most of the prefill at 8 streams) still ran
serially.

## Detail

See README.md.
