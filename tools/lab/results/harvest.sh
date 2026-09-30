#!/bin/bash
# Pull compact summaries of every finished Brutus lab run into tools/lab/results/runs/ (committed).
# Run from the controller checkout after runs land; re-running refreshes all summaries.
set -eu
host=${BC_BENCH_HOST:-audumla@10.10.100.10}
runs=${BC_RUNS_DIR:-/mnt/data/bigcherry-work/runs}
remote_out=/mnt/data/bigcherry-work/results-summary
here=$(cd "$(dirname "$0")" && pwd)
ssh "$host" "python3 - $runs $remote_out" < "$here/summarize.py"
mkdir -p "$here/runs"
scp -q "$host:$remote_out/*.json" "$here/runs/"
echo "harvested $(ls "$here/runs" | wc -l) run summaries into $here/runs"
