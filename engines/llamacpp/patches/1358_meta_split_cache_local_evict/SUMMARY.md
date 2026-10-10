# 1358_meta_split_cache_local_evict

**Status:** validated
**Plan item:** QFP41

## What it does

The Meta backend memoises each tensor's split state under its address, with a copy of the tensor to detect an
address that was recycled for another tensor. On a mismatch upstream clears the whole cache; this patch erases only
the stale entry. On by default; `BIGCHERRY_META_SPLIT_CACHE_EVICT=0` restores the whole-cache clear.

## Why

Graph tensor addresses are recycled between graph shapes (prefill chunk, decode step, draft verify). One stale entry
then throws away the memoised state of every other tensor of the current graph and the recursive walk recomputes it
on the host, between chunks.

## Scope

Tensor-split (Meta) runs on any model. Host-side only: no kernel, tensor placement or result changes. Every remaining
entry is still compared with its own tensor copy when it is looked up.

## Activation

`BIGCHERRY_PATCH_TRACE=1` logs `BIGCHERRY_PATCH_HIT patch=1358_meta_split_cache_local_evict mechanism=local-evict
entries=` once, the first time a stale entry is evicted.

## Evidence

Indirect so far: the change shipped inside 1328, and a two-build ABBA (run `aux3off-r1`, `aux3off-r2`: production
`b-main2` against `b-aux3` = production + 1328 with expert offload off, Flash-Next, ctx 245760, f16 KV) gave
identical text at 8K / 24K / 98K and prefill +2.1% / +3.3% / +3.0%, every run of the second build above every run of
the first. That comparison does not isolate this edit from the rest of 1328; the one-binary A/B with the off switch
does.
