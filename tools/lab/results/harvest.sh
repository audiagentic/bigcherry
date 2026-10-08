#!/bin/bash
# Pull compact summaries of finished Brutus lab runs into docs/evidence.
# Run from the controller checkout after runs land; re-running refreshes summaries.
set -eu
host=${BC_BENCH_HOST:-audumla@10.10.100.10}
runs=${BC_RUNS_DIR:-/mnt/data/bigcherry-work/runs}
remote_out=/mnt/data/bigcherry-work/results-summary
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
dest="$repo/docs/evidence/lab-run-summaries"
ssh "$host" "python3 - $runs $remote_out" < "$here/summarize.py"
mkdir -p "$dest"
scp -q "$host:$remote_out/*.json" "$dest/"
echo "harvested $(find "$dest" -maxdepth 1 -name '*.json' | wc -l) run summaries into $dest"
