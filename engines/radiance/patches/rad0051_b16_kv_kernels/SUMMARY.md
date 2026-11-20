# rad0051_b16_kv_kernels

**Status:** validated
**Plan item:** RUM07

Scope: Port paired B16F8 and B16I8 kernels to the ABI44 runtime contract.

## Summary

The patch adds typed B16 stores/readers and preserves separate QSA storage and shared Qwen4exp trunk/MTP KV ownership.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.

## Detail

The kernel declarations bind the BF16-key layout and each value codec so a cache store has a matching attention reader.

[Patch implementation](patch.py).
