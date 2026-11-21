# rad0048_abi44_kv_contract

**Status:** validated
**Plan item:** RUM03/RUM07

Scope: Define the local ABI44 KV layout contract and canonical packed-format descriptors.

## Summary

The patch appends and fingerprints the local KV layout/capability fields and defines one descriptor table for V4, K6V4, B16F8 and B16I8.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The ABI layout fingerprint covers the appended KV capability fields. Consumers use the same descriptors for geometry checks and layout sizes.

[Patch implementation](patch.py).
