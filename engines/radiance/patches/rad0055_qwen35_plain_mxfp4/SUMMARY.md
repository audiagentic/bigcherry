# rad0055_qwen35_plain_mxfp4

**Status:** validated
**Plan item:** RUM06

Scope: Restore plain MXFP4 admission; host-only verification completed, hardware admission pending.

## Summary

Restore only plain MXFP4 from the 2.2 retired-weight classifier, reusing
rad0019's existing q|k|v|z row-part and uneven tensor-parallel span behavior.
Retain all other explicit format refusals.

## Outcome

Exact frozen-composition patch replay and libr3 native rewrite pass.
No GPU runtime, KLD or throughput claim is made.

## Detail

[Patch implementation](patch.py) and [validation notes](README.md).
