# rad0050_b16_kv_runtime_contract

**Status:** validated
**Plan item:** RUM07

Scope: Port B16 cache configuration and resident allocation to ABI44.

## Summary

The patch carries packed-layout selection through CLI/YAML, capability admission, model declaration, sizing, fallback and binding; unsupported streamed writes remain resident.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The runtime validates the packed layout before allocation and uses the same declared layout for cache binding.

[Patch implementation](patch.py).
