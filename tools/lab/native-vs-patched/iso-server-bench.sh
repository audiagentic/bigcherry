#!/usr/bin/env bash
# usage: iso-server-bench.sh <llama-server> <label> <port> <bench-configs> <extra server args...>
set -u
BIN=$1; LABEL=$2; PORT=$3; CFG=$4; shift 4
MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
LOG=/home/audumla/iso-$LABEL.log
HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1 LLAMA_SERVER_ENABLE_SHUTDOWN=1 setsid "$BIN" -m $MODEL -ngl 99 --fit off --flash-attn on --ubatch-size 512 --batch-size 2048 --threads 8 --parallel 1 --port $PORT "$@" < /dev/null > $LOG 2>&1 &
SP=$!
for i in $(seq 1 150); do grep -q "listening" $LOG && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED"; tail -n 5 $LOG; exit 1; }; sleep 2; done
cd /mnt/vault/development/llmhosts/llamacpp
python3 bench/run_bench.py --bench-type server-bench --server-url http://127.0.0.1:$PORT --model Qwen3.8-27B-Q8_0 --bench-configs $CFG --toggles '{"repetitions":2}' 2>&1 | grep -E "_tps|tps|Log:"
kill -INT $SP; wait $SP 2>/dev/null
echo DONE $LABEL
