#!/bin/bash
# QFP17/QFP22 1332: where does chunked prefill time go? rocprof kernel stats of one prefill fill (~32K tokens) for
# ub512 unchunked, ub512 chunk256, ub1024 chunk256, ub1024 chunk512 (240K f16, flashnext profile), then kernel time grouped by family.
# Also: the no-MTP single-token decode crash case under GGML_SCHED_DEBUG_REALLOC=1 (graph realloc / topology changes).
# Usage: chunk-prof.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export BIGCHERRY_FEATURES=flashnext CTX=245760 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 TS=0.31,0.27,0.42 EXTRA_OT='^token_embd\.weight$=CPU'
export DEPTH=20480
for arm in 512:0 512:256 1024:256 1024:512; do
  ub=${arm%%:*} c=${arm##*:}
  BIGCHERRY_QSA_CHUNK=$c UB=$ub B=$ub bash "$s" "$bin" "$root/ub$ub-c$c" prefillprof 2>&1 | grep -E "^prefillprof:|SERVER_FAILED" | sed "s/^/ub$ub c$c: /"
done
python3 - "$root" <<'PY'
import csv, glob, sys, re
root = sys.argv[1]
fam = [("flash_attn", r"flash_attn|fattn"), ("mask build (fill/repeat/set_rows/add)", r"fill|repeat|set_rows|k_bin_bcast|add"),
       ("concat", r"concat"), ("matmul (mmq/mmvq/mmf)", r"mul_mat|mmq|mmvq|mmf"), ("allreduce/copy", r"cpu_root|allreduce|cpy|copy"),
       ("gdn/ssm", r"gated_delta|ssm|gdn"), ("indexer/topk", r"indexer|top_k|topk|argsort")]
for d in sorted(glob.glob(f"{root}/ub*")):
    files = glob.glob(f"{d}/rocprof/**/*kernel_stats.csv", recursive=True)
    tot = {}
    total = 0.0
    for f in files:
        for row in csv.DictReader(open(f)):
            name = row.get("Name", "")
            ns = float(row.get("TotalDurationNs", 0) or 0)
            total += ns
            key = next((k for k, rx in fam if re.search(rx, name, re.I)), "other")
            tot[key] = tot.get(key, 0) + ns
    print(f"== {d.split('/')[-1]}  total GPU kernel time {total/1e9:.2f} s (summed over GPUs)")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"   {k:42s} {v/1e9:7.3f} s  {100*v/max(total,1):5.1f}%")
PY
# crash case diagnostics: no-MTP 1-token decode after a chunked prefill
GGML_SCHED_DEBUG_REALLOC=1 NO_MTP=1 DECODE_N=8 DEPTH=24576 BIGCHERRY_QSA_CHUNK=256 UB=512 B=512 \
  bash "$s" "$bin" "$root/crash-realloc" timing 2>&1 | grep -E "^timing:|SERVER_FAILED" | sed "s/^/crash-case: /"
grep -hiE "realloc|graph.*(nodes|leafs)|reserve" "$root/crash-realloc/timing.server.log" | tail -15 | sed "s/^/realloc: /"
# correctness after the compact causal filter: no-MTP decode (the old crash case) and MTP serving identity, both chunk on/off
here=$(cd "$(dirname "$0")" && pwd)
bash "$here/chunk-nomtp.sh" "$bin" "$root/nomtp" 2>&1 | sed "s/^/nomtp: /"
QUICK_DEPTH=24576 bash "$here/quick-ab.sh" "$bin" "$bin" "$root/mtp-d24k" BIGCHERRY_QSA_CHUNK=256 2>&1 | sed "s/^/mtp-d24k: /"
md5sum "$root"/mtp-d24k/*/*.greedy.txt | awk '{print $1}' | sort | uniq -c | sed "s/^/mtp-d24k greedy: /"
