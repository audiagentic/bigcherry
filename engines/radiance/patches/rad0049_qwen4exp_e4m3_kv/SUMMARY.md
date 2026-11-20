# rad0049_qwen4exp_e4m3_kv

**Status:** validated
**Plan item:** RUM07

Scope: Restore native E4M3 Flash-Next cache rows on the ABI43 base.

## Summary

The patch restores the deployed H256/GQA12 and GQA6/page4 E4M3 rows, store and attention wiring without aliasing the format to I8.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.

## Detail

The Qwen4exp declarations keep the E4M3 format and geometry explicit at both store and attention selection.

[Patch implementation](patch.py).
