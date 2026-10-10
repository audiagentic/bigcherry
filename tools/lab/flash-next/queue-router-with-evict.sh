#!/bin/bash
# 1357 (router split-K) and 1358 (stale split-state cache entry eviction) together: build production + both
# (experiment router-splitk-with-cache-evict), then on that one binary, at 8K, 24K and 98K, twice:
#   both-vs-neither: A = both on, B = both off          -> the combined prefill gain
#   both-vs-evict:   A = both on, B = 1358 only         -> what 1357 adds on top of 1358
# 1358 is on by default; 1357 is switched on for the whole run and off again in the B arms.
# Greedy text: 1357 changes it, so A and B differ here by design; the md5 lines show whether each arm is stable.
# Run from the lab tree with the branch checked out.
# Usage: queue-router-with-evict.sh [tag]     (default tag rse1; build name b-metamem-<tag>)
R=/mnt/data/bigcherry-work/runs
tag=${1:-rse1}
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BIGCHERRY_PATCH_TRACE=1
rm -rf $R/b-metamem-$tag.log $R/metamem-$tag* $R/$tag-*
EXPERIMENT=router-splitk-with-cache-evict CTX_LIST=245760 DEPTH=8192 bash tools/lab/flash-next/queue-meta-mem.sh $tag | grep -E "^timing|SERVER_FAILED|QUEUE_EXIT"
grep -hE "^BUILD_(EXIT|BINARY)" $R/b-metamem-$tag.log | cut -c1-160
grep -hE "error:" $R/b-metamem-$tag.log | sort -u | head -8 | cut -c1-220
export BIGCHERRY_MOE_ROUTER_SPLITK=1
for r in 1 2; do
    AB_ENV="BIGCHERRY_MOE_ROUTER_SPLITK=0 BIGCHERRY_META_SPLIT_CACHE_EVICT=0" bash tools/lab/flash-next/queue-env-ab.sh $tag-none$r b-metamem-$tag 8192 24576 98304 | sed "s/^d/both-vs-neither r$r d/" | grep -E "^==|^both|^md5|SERVER_FAILED|blocked" | cut -c1-150
    AB_ENV="BIGCHERRY_MOE_ROUTER_SPLITK=0" bash tools/lab/flash-next/queue-env-ab.sh $tag-evict$r b-metamem-$tag 8192 24576 98304 | sed "s/^d/both-vs-evict r$r d/" | grep -E "^==|^both|^md5|SERVER_FAILED|blocked" | cut -c1-150
done
echo "markers at 24K, both-vs-neither r1 (A B B A): 1357 $(for d in $R/$tag-none1-d24576/*/; do cat $d/*.server.log | grep -c "patch=1357"; done | tr "\n" " ") 1358 $(for d in $R/$tag-none1-d24576/*/; do cat $d/*.server.log | grep -c "patch=1358"; done | tr "\n" " ")"
echo ROUTER_WITH_EVICT_DONE
