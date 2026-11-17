# rad0037_bf16_fp8_kv

**Status:** validated
**Plan item:** VTP43

Scope: one new `--kv-cache-dtype` value, `bf16_fp8`: bf16 keys and E4M3 values in a paged cache.

## Summary

`--kv-cache-dtype bf16_fp8` stores a token's key as bf16 and its value as E4M3 (saturating, scale 1.0), 3 * head_dim
bytes a token and head against 4 * head_dim for bf16 and 2 * head_dim for fp8. Keys keep bf16's quality; values take
fp8's half size. It is served by libr4d's h256/gqa6 prefill (plain and KV-split) and decode kernels, a `kv_store` row and
the fused `qk_norm_rope_gate_kv_store` row; the Qwen3.5 FP8 plugin declares it (`RAD_KVQ_B16F8`) and probes it for that
geometry only. Other geometries, the e4m3 product legs (`prefill_fp8`), streaming and gfx11 non-native paths are refused
at resolve or startup, never run on a wrong layout.

## Outcome

Untested. See README.md.

## Detail

See README.md.
