#!/bin/bash
# 1357 split-K router GEMM (QFP36): build production + 1357 (experiment moe-router-splitk), then ABBA on one binary,
# A = production, B = BIGCHERRY_MOE_ROUTER_SPLITK=1, at 8K, 24K and 98K twice: prefill and decode speed, greedy
# identity, and the activation marker per arm. Run from the lab tree with the branch checked out.
# Usage: queue-router-splitk.sh [tag]     (default tag rsk1; build name b-metamem-<tag>)
R=/mnt/data/bigcherry-work/runs
tag=${1:-rsk1}
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BIGCHERRY_PATCH_TRACE=1
rm -rf $R/b-metamem-$tag.log $R/metamem-$tag* $R/$tag-*
EXPERIMENT=moe-router-splitk CTX_LIST=245760 DEPTH=8192 bash tools/lab/flash-next/queue-meta-mem.sh $tag | grep -E "^timing|SERVER_FAILED|QUEUE_EXIT"
grep -hE "^BUILD_(EXIT|BINARY)" $R/b-metamem-$tag.log | cut -c1-160
grep -hE "error:" $R/b-metamem-$tag.log | sort -u | head -8 | cut -c1-220
E="BIGCHERRY_MOE_ROUTER_SPLITK=1"
for r in 1 2; do
    AB_ENV="$E" bash tools/lab/flash-next/queue-env-ab.sh $tag-r$r b-metamem-$tag 8192 24576 98304 | sed "s/^d/r$r d/" | grep -E "^==|^r[0-9]|^md5|SERVER_FAILED|blocked"
done
echo "marker (A B B A at 24K): $(for d in $R/$tag-r1-d24576/*/; do cat $d/*.server.log | grep -c "patch=1357"; done | tr "\n" " ")"
grep -h "patch=1357" $R/$tag-r1-d24576/*/*.server.log | sort | uniq -c | head -4
