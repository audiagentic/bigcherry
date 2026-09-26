# Plan-item patch qualification campaigns

Plan item: RDNA/nasone plan implementation (PRBE*/PNRO*)
Status: active
Owner: plan implementation loop
Question state: reusable launcher

## Question

Does a plan item's patch pass its validation.toml checks (activation,
correctness, performance, controls) on real hardware at the current pin?

## Inputs

`run_campaign.sh <patch-id> <producer|-> <arch> <device> <run-name> [args]`
with `BC_HIP_PATH` and `BC_MODEL` set in the environment.

## Outputs

`$BIGCHERRY_WORK_ROOT/runs/<run-name>/` campaign workdir (resolved by `work-root.sh`: the environment variable, else `BIGCHERRY_WORK_ROOT` in the untracked `config/environment.local.toml` `[env]` table, else `work/`; on the build server point it at a large scratch volume); evidence appended by the campaign
to `patches/<id>/evidence/validation.json`.

## Runtime

GPU required: yes (one device). Real compilation: yes. Mutates canonical
BigCherry state: appends patch evidence only.

## Safety

Run under nohup on the build server; one campaign per GPU; no other GPU jobs
on that device while it runs.

## Disposition

Keep while plan items are being qualified.
