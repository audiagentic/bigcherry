# rad0053_v4_kv_layouts

**Status:** validated
**Plan item:** RUM07

Scope: Restore fp8_v4 and k6_v4 on the shared ABI44 packed-KV descriptor.

## Summary

The patch ports the historical V4/K6 stores, readers and geometry rows using descriptor-owned offsets and preserves the tested standalone and fused scale arithmetic.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The shared layout descriptor makes fused and standalone stores agree on the byte offsets used by attention.

[Patch implementation](patch.py).
