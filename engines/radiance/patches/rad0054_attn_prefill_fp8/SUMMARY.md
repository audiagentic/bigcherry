# rad0054_attn_prefill_fp8

**Status:** validated
**Plan item:** RUM07

Scope: Restore independent E4M3 prefill precision modes for historical rows.

## Summary

The patch restores off, QK, PV and combined prefill modes on the supported Qwen35 H256/GQA6/page16 E4M3, V4 and K6V4 rows; decode and unsupported geometries do not accept the prefill selector.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

The precision selector applies to prefill and is limited to kernels whose QK and PV readers support the chosen format.

[Patch implementation](patch.py).
