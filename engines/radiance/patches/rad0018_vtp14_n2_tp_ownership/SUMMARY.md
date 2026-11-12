# rad0018_vtp14_n2_tp_ownership

**Status:** untested
**Plan item:** VTP14

Scope: `--tp-ownership SPEC` (`kv=..;ffn=..[;q=..][;lk=..;lv=..]`), an uneven tensor-parallel split, one count a
rank, checked against the loaded model (BF16 Qwen plugin). No model or split is built in; the split is manifest
configuration.

## Summary

Lets a layout put more heads and feed-forward rows on a larger card. Each rank reads its own spans of the container's
weights through radiance's shard-span ABI; `geom_from` refuses a spec whose shares do not match the model.

## Outcome

On cards, with rad0016 and rad0017, native TP3 serves the 64-layer Qwen3.8-27B BF16 correctly at
`kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24` over (XTX, XTX, R9700): about 23 tok/s single stream (smoke, not a
benchmark).

## Detail

See README.md and docs/planning/active/run-dense-27b-virtual-tp/VTP14.md.
