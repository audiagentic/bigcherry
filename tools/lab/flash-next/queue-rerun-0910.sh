#!/bin/bash
# Re-run after the 0910 deadlock fix (af459ed9 hung every server at ggml_init): smoke, DFlash sweep (27B + 1286), default-on 3-model check, 1301 width sweep.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
true
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-release3 bigcherry:stock:linux-multi stock-none gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-r1286c bigcherry:stock:linux-multi retest-1286 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 BUILD b-r1301c bigcherry:stock:linux-multi retest-1301 gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT smoke-0910 tools/lab/default-on/smoke-0910.sh @b-release3 $R/smoke-0910
VIS=0,1,2,3 SCRIPT q21-dflash2 tools/lab/dflash/probe-27b.sh @b-r1286c $R/qfp21-dflash2/out (plain|mtp5|q21-dflash-.*) 3
VIS=0,1 SCRIPT don2-27b tools/lab/default-on/xmodel-ab.sh @b-release3 $R/default-on2-27b hip-q81,sched-async /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf -sm tensor -ts 1,1 -ub 512 -b 2048 --spec-type draft-mtp --spec-draft-n-max 4
VIS=0 SCRIPT don2-35b tools/lab/default-on/xmodel-ab.sh @b-release3 $R/default-on2-35b hip-q81,sched-async /mnt/data/llm-models/qwen3.6-35B-A3B/gguf/Qwen3.6-35B-A3B-UD-IQ3_S.gguf -ub 512 -b 2048
VIS=0 SCRIPT don2-4b tools/lab/default-on/xmodel-ab.sh @b-release3 $R/default-on2-4b hip-q81,sched-async /mnt/data/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf -ub 512 -b 2048
VIS=0,1 SCRIPT q8w2b-27b tools/lab/flash-next/prod27b-ab.sh @b-r1301c @b-r1301c $R/qfp21-1301-27b-w2b BIGCHERRY_Q8_F32_MAXCOLS=2
VIS=0,1 SCRIPT q8w3b-27b tools/lab/flash-next/prod27b-ab.sh @b-r1301c @b-r1301c $R/qfp21-1301-27b-w3b BIGCHERRY_Q8_F32_MAXCOLS=3
JOBS
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo "== smoke"; cat $R/smoke-0910.log | grep -E "^(smoke|help|ok|FAIL)"
echo "== dflash"; grep -E "^[a-z0-9-]+: (prompt|SERVER_FAILED)" $R/q21-dflash2.log
md5sum $R/qfp21-dflash2/out/*.greedy.txt | awk '{print $1}' | sort | uniq -c
for m in 27b 35b 4b; do echo "== default-on $m"; grep -E "^d[0-9]|^ +[0-9]" $R/don2-$m.log; done
for w in 2b 3b; do echo "== 1301 MAXCOLS ${w%b}"; grep -E "^d[0-9]|SERVER_FAILED" $R/q8w$w-27b.log; done
echo ALL_JOBS_DONE
