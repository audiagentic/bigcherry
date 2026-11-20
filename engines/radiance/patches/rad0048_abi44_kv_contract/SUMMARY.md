# rad0048_abi44_kv_contract

**Status:** validated
**Plan item:** RUM03/RUM07

Scope: Define the local ABI44 KV layout contract and canonical packed-format descriptors.

## Summary

The patch appends and fingerprints the local KV layout/capability fields and defines one descriptor table for V4, K6V4, B16F8 and B16I8.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.

## Detail

The ABI layout fingerprint covers the appended KV capability fields. Consumers use the same descriptors for geometry checks and layout sizes.

[Patch implementation](patch.py).
