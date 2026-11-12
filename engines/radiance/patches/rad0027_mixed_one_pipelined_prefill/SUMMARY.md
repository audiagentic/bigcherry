# rad0027_mixed_one_pipelined_prefill

**Status:** untested
**Plan item:** VTP31

Scope: `qwen35_fp8` and `qwen35_w8a8`'s pipelined prefill (rad0020, rad0026), for a mixed step of decode rows and one
prefill sequence.

## Summary

A step that carries decode rows beside one prefill sequence is cut inside that sequence and pipelined. The first piece
holds the decode rows and the sequence's first rows, the second the rest. Each block issues the serial path's decode
launches, then rad0020's one-sequence prefill pieces at an offset row.

## Outcome

Untested (not promoted). Built for VTP31: a long prompt beside generating streams ran its later chunks serially.

## Detail

See README.md.
