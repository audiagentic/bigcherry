# rad0051_b16_kv_kernels

**Status:** validated
**Plan item:** RUM07

Scope: Port paired B16F8 and B16I8 kernels to the ABI44 runtime contract.

## Summary

The patch adds typed B16 stores/readers and preserves separate QSA storage and shared Qwen4exp trunk/MTP KV ownership.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The kernel declarations bind the BF16-key layout and each value codec so a cache store has a matching attention reader.

[Patch implementation](patch.py).
