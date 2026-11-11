# rad0018_vtp14_n2_tp_ownership

**Status:** untested
**Plan item:** VTP14 (N2)

Scope: `--tp-ownership qwen38-27b-6-6-12`, an asymmetric three-rank ownership of the dense Qwen3.8-27B (BF16 plugin). It
puts twice the heads and the largest feed-forward share on the 32 GiB R9700.

## Summary

Splits the dense Qwen3.8-27B unevenly over three ranks. Query heads go 6/6/12, KV heads 1/1/2, delta-net heads 4/4/8
(K) and 12/12/24 (V), and the feed-forward 5120/4864/7424. Each rank reads its own spans of the container's weights
through radiance's shard-span ABI.

## Outcome

Card-free only so far. The patch applies to v1.17.0 after rad0016 and is idempotent. It has not been compiled or
run on hardware yet.

## Detail

See README.md and docs/planning/active/run-dense-27b-virtual-tp/VTP14.md.
