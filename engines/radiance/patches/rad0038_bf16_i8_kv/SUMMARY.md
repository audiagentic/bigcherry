# rad0038_bf16_i8_kv

**Status:** untested
**Plan item:** VTP43

Scope: one new `--kv-cache-dtype` value, `bf16_i8`: bf16 keys and int8 values with one f16 scale per 32 value dims.

## Summary

`--kv-cache-dtype bf16_i8` stores a token's key as bf16 and its value as int8 with an f16 scale per 32 dims (the
smallest f16 with 127 * scale >= the group's absmax; a zero group stores scale 0), 49 * head_dim / 16 bytes a token and
head (784 at head_dim 256) against 4 * head_dim for bf16 and 3 * head_dim for `bf16_fp8`. It exists because the E4M3
value of `bf16_fp8` flips exact-copy tokens at long context (VTP43). It is served by libr4d's h256/gqa6 prefill (plain
and KV-split) and decode kernels, a `kv_store` row and the fused `qk_norm_rope_gate_kv_store` row; the Qwen3.5 FP8 plugin
declares it (`RAD_KVQ_B16I8`) and probes it for that geometry only. Other geometries, the e4m3 product legs,
streaming and gfx11 non-native paths are refused at resolve or startup.

## Outcome

Untested. See README.md.

## Detail

See README.md.
