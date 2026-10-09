#!/bin/bash
# 1358 local eviction of stale Meta split-state cache entries: build production + 1358 (experiment
# meta-split-cache-evict), then ABBA on one binary, A = default (on), B = BIGCHERRY_META_SPLIT_CACHE_EVICT=0
# (upstream's whole-cache clear), at 8K, 24K and 98K twice: prefill and decode speed, greedy identity, marker per arm.
# Run from the lab tree with the branch checked out.
# Usage: queue-split-cache-evict.sh [tag]     (default tag sce1; build name b-metamem-<tag>)
R=/mnt/data/bigcherry-work/runs
tag=${1:-sce1}
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BIGCHERRY_PATCH_TRACE=1
rm -rf $R/b-metamem-$tag.log $R/metamem-$tag* $R/$tag-*
EXPERIMENT=meta-split-cache-evict CTX_LIST=245760 DEPTH=8192 bash tools/lab/flash-next/queue-meta-mem.sh $tag | grep -E "^timing|SERVER_FAILED|QUEUE_EXIT"
grep -hE "^BUILD_(EXIT|BINARY)" $R/b-metamem-$tag.log | cut -c1-160
grep -hE "error:" $R/b-metamem-$tag.log | sort -u | head -8 | cut -c1-220
E="BIGCHERRY_META_SPLIT_CACHE_EVICT=0"
for r in 1 2; do
    AB_ENV="$E" bash tools/lab/flash-next/queue-env-ab.sh $tag-r$r b-metamem-$tag 8192 24576 98304 | sed "s/^d/r$r d/" | grep -E "^==|^r[0-9]|^md5|SERVER_FAILED|blocked"
done
echo "marker (A B B A at 24K): $(for d in $R/$tag-r1-d24576/*/; do cat $d/*.server.log | grep -c "patch=1358"; done | tr "\n" " ")"
