# Lab run summary tooling

`summarize.py` and `harvest.sh` produce compact summaries of A/B, KLD and
patch-campaign runs executed on the bench host. Raw logs and per-round server
output stay on the host under `/mnt/data/bigcherry-work/runs/<run>/`.

Tracked summaries are evidence, not tooling, and live under
`docs/evidence/lab-run-summaries/`.

- `summarize.py` runs on the host and emits one JSON summary per run.
- `harvest.sh` summarizes on the host and copies those JSON files into the
  tracked evidence directory.
- A/B summaries include per-arm means, paired effect/CI95 and pooled MTP
  acceptance; KLD summaries include mean/p99 KLD, same-top rate and PPL ratio.

Run names follow the queue row names (`ab-27b-*`, `kld-*`, `kldd-*`,
`t-<patch>-*`, `qs-*`).
