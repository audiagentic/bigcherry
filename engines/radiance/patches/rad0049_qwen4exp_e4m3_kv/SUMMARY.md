# rad0049_qwen4exp_e4m3_kv

**Status:** validated
**Plan item:** RUM07/RUM11

Scope: Restore native E4M3 Flash-Next cache rows on the ABI43 base.

## Summary

The patch restores the deployed H256/GQA12 and GQA6/page4 E4M3 rows, store and attention wiring without aliasing the format to I8.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The Qwen4exp declarations keep the E4M3 format and geometry explicit at both store and attention selection.

[Patch implementation](patch.py).

RUM11: a separate H256/page16 E4M3 fused-store row admits variable local head counts and rotary widths, with byte-identity selftests for 6/1 and 12/2 heads. This does not change the deployed Flash-Next page-4 row. Hardware selftest and TP3 startup/KLD qualification remain outstanding.
