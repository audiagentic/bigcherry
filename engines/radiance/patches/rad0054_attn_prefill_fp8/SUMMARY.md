# rad0054_attn_prefill_fp8

**Status:** validated
**Plan item:** RUM07

Scope: Restore independent E4M3 prefill precision modes for historical rows.

## Summary

The patch restores off, QK, PV and combined prefill modes on the supported Qwen35 H256/GQA6/page16 E4M3, V4 and K6V4 rows; decode and unsupported geometries do not accept the prefill selector.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.
