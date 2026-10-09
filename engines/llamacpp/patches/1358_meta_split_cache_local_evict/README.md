# 1358_meta_split_cache_local_evict

Not promoted. Mechanism and the indirect evidence are in SUMMARY.md.

## Still to show

One binary (production + 1358), ABBA with `tools/lab/flash-next/queue-split-cache-evict.sh`: A = default (on),
B = `BIGCHERRY_META_SPLIT_CACHE_EVICT=0`, at 8K, 24K and 98K twice: prefill and decode speed, identical greedy text,
the marker in A only. Then the same on Qwen3.8-27B (two XTX tensor split), since the patch is not model-specific.
