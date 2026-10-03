# Lab run results

Compact, committed summaries of every A/B, KLD and patch-campaign run executed on the bench host
(Brutus). Raw logs and per-round server output stay on the host under
`/mnt/data/bigcherry-work/runs/<run>/`; these summaries are what later analysis should read.

- `summarize.py` — runs on the host; one JSON per run:
  - A/B: per-arm metric means, paired comparisons (effect + CI95), pooled MTP draft acceptance
    (same calculation as `tools/lab/ar-accuracy/gates.py`), and each arm's binary, server args and
    environment.
  - KLD: mean and p99 KLD, same-top-token rate, PPL ratio, run environment.
  - Patch campaigns: contract lane effects and correctness verdicts.
- `harvest.sh` — run from the controller checkout: summarises on the host and copies into `runs/`.
  Commit `runs/` after each harvest.

Run names follow the queue row names (`ab-27b-*`, `kld-*`, `kldd-*` decode-mode KLD, `t-<patch>-*`
contract sessions, `qs-*` quant sweep).
