# PA36/PA37/PA39 Brutus hardware launch scripts

Plan item: PA36 (patching-patch-system)
Status: archived (one-shot, spent)
Owner: patching-patch-system
Question state: historical — the campaigns these launched are complete

## Question

How to background PA36's RD04/RD13/RD26 migrated-producer hardware legs on
Brutus over SSH without the backgrounded process dying when the SSH/Windows
quoting layer or the parent agent session disconnects.

## Inputs

Real Brutus hardware (gfx1030/gfx1100 devices), the `bigcherry-tuning`
5-build scaffold, and the specific patch/producer combinations named in each
script (RD04/1202, RD13/1206, RD26/1210).

## Outputs

Run logs under `/mnt/vault/tmp/bc-runs/` on Brutus (not in this repo) —
already consumed and folded into PA36's plan-item record and this project's
`docs/evidence/` / `patches/*/evidence/` entries.

## Runtime

GPU required: yes (real Brutus hardware). Real compilation required: yes.
Mutates canonical BigCherry state: no — these only invoke the standard
`bigcherry build`/campaign entry points via SSH+nohup backgrounding.

## Safety

Host-specific (hardcoded `/home/audumla/...`, `/mnt/vault/...` paths) —
not portable, not intended to be re-run as-is on a different host or for a
different patch.

## Disposition

One-shot launch scripts for PA36's now-complete hardware acceptance runs.
Retained here (moved from the repo root, see AGENTS.md's ad-hoc output
placement rule) as a historical record of the exact invocation used, not as
reusable/maintained tooling. Safe to delete once no longer useful as a
reference; do not extend or generalize in place — a new hardware campaign
should use `bigcherry build`/`bigcherry runtime-matrix` directly, or a fresh
`tools/lab/<topic>/` script with its own README.
