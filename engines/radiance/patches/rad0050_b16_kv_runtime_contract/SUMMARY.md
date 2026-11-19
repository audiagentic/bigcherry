# rad0050_b16_kv_runtime_contract

**Status:** validated
**Plan item:** RUM07

Scope: Port B16 cache configuration and resident allocation to ABI44.

## Summary

The patch carries packed-layout selection through CLI/YAML, capability admission, model declaration, sizing, fallback and binding; unsupported streamed writes remain resident.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.
