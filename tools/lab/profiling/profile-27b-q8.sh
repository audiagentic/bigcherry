#!/bin/bash
# rocprofv3 attribution for Qwen3.8-27B Q8_0 on dual gfx1100 (-sm tensor), plan from dev-gpt-agent
# (2026-10-01): 64-token decode timeline, one 2048-token prefill ubatch, and SQ counter run.
# Usage: profile-27b-q8.sh <llama-server path from a BUILD row> <out-dir>
# llama-bench is taken from the same build's bin/. Runs under the queue's host + GPU locks.
set -u
server=$1 out=$2
lb=$(dirname "$server")/llama-bench
model=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
rp=${BC_HIP_PATH:-/opt/rocm}/bin/rocprofv3
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1
common=(-m "$model" -ngl 99 -sm tensor -fa on -b 2048 -r 1 -o jsonl)
mkdir -p "$out"
# BUILD rows build llama-server only; add llama-bench to the same build tree (incremental).
if [ ! -x "$lb" ]; then
  cmake --build "$(dirname "$(dirname "$server")")" --target llama-bench -j > "$out/build-llama-bench.log" 2>&1     || { echo "llama-bench build failed"; tail -20 "$out/build-llama-bench.log"; exit 1; }
fi
"$lb" "${common[@]}" -p 0 -n 32 -ub 512 > "$out/warmup-tg.jsonl"
"$rp" --runtime-trace --stats --output-format csv --output-directory "$out/tg" \
  -- "$lb" "${common[@]}" --no-warmup -p 0 -n 64 -ub 512 > "$out/tg.jsonl" 2> "$out/tg.stderr"
tg_rc=$?; echo "TG_EXIT=$tg_rc"
"$lb" "${common[@]}" -p 2048 -n 0 -ub 2048 > "$out/warmup-pp.jsonl"
"$rp" --runtime-trace --stats --output-format csv --output-directory "$out/pp2048" \
  -- "$lb" "${common[@]}" --no-warmup -p 2048 -n 0 -ub 2048 > "$out/pp2048.jsonl" 2> "$out/pp2048.stderr"
pp_rc=$?; echo "PP_EXIT=$pp_rc"
"$rp" --pmc SQ_WAVES SQ_BUSY_CYCLES GRBM_GUI_ACTIVE --output-format csv --output-directory "$out/tg-pmc" \
  -- "$lb" "${common[@]}" --no-warmup -p 0 -n 64 -ub 512 > "$out/tg-pmc.jsonl" 2> "$out/tg-pmc.stderr"
echo "PMC_EXIT=$?"  # counters are best-effort
du -sh "$out"
echo PROFILE_DONE
[ "$tg_rc" -eq 0 ] && [ "$pp_rc" -eq 0 ]
