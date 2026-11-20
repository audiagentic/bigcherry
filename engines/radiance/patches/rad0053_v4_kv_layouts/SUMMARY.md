# rad0053_v4_kv_layouts

**Status:** validated
**Plan item:** RUM07

Scope: Restore fp8_v4 and k6_v4 on the shared ABI44 packed-KV descriptor.

## Summary

The patch ports the historical V4/K6 stores, readers and geometry rows using descriptor-owned offsets and preserves the tested standalone and fused scale arithmetic.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.

## Detail

The shared layout descriptor makes fused and standalone stores agree on the byte offsets used by attention.

[Patch implementation](patch.py).
