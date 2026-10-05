#!/bin/bash
# QFP22 1332: localise the no-MTP crash after a chunked prefill (dense 4-token tail ubatch, segfault in
# ggml_backend_meta_synchronize). Three chunk-256 runs: gdb backtrace, scheduler realloc debug, 1326 off.
# Usage: chunk-nomtp-diag.sh <llama-server> <out-root>
set -u
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 DECODE_N=64 DEPTH=24576 CTK=f16 CTV=f16 BIGCHERRY_QSA_CHUNK=256
run() {  # <name> ; env from caller
  echo "$1: $(bash "$s" "$bin" "$root/$1" timing 2>&1 | grep -E '^timing: prompt|SERVER_FAILED' | tr '\n' ' ')"
}
bin=$1 root=$2
WRAP="gdb -q -batch -ex run -ex bt --args" run gdb
grep -nE "^#[0-9]+ |SIGSEGV|SIGABRT|GGML_ASSERT" "$root/gdb/timing.server.log" | head -40
GGML_SCHED_DEBUG_REALLOC=1 run realloc
grep -nE "realloc|needs_realloc|graph_reserve|not reserved" "$root/realloc/timing.server.log" | tail -30
BIGCHERRY_SCHED_ASYNC_INPUTS=0 run async-off
