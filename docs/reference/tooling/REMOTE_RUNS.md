# Remote lab runs

`python -m bigcherry remote` runs a bigcherry command detached on the GPU host and tracks it. It replaces the hand-written
"ssh, cd to the tree, checkout, export, nohup, poll the log" scripts. It is platform scope: it names no engine, the inner
arguments carry their own `--engine`.

```
python -m bigcherry remote run --host dev --ref feat/some-branch [--allow-busy] -- --engine radiance <command> ...
python -m bigcherry remote status <run-id> --host dev
python -m bigcherry remote wait <run-id> --host dev [--timeout 7200] [--tail 40]
python -m bigcherry remote list --host dev [--limit 10]
```

## What the host config supplies

Everything comes from the `[host.<name>]` table of the environment file (`config/environment.local.toml` or
`$BIGCHERRY_ENVIRONMENT`); `--host` defaults to `default-host`.

| Key | Use |
|---|---|
| `hostname` | ssh alias the commands go to |
| `repo` | the campaign tree: a checkout (or worktree) that shares the primary work root. A fresh clone does not. |
| `work-root` | the work root; runs live in `<work-root>/remote-runs/<run-id>/` |
| `rocm` | exported as `ROCM_PATH`, with `$ROCM_PATH/bin` first on `PATH` |
| `environment-file` | exported as `BIGCHERRY_ENVIRONMENT` (default `<repo>/config/environment.local.toml`) |
| `busy-processes` | exact process names that mean the GPUs are in use (for example the server, a helper, `ninja`). Required: with none listed, `run` refuses. |

The command runs as `python -m bigcherry ...` (the host interpreter is `python3`) with `PYTHONPATH=tools` in the tree. `BIGCHERRY_REMOTE_PYTHON` on the host
overrides the interpreter.

## Preflight (`run`)

In order, on the host, in one ssh call; the first failure refuses (exit 2) and nothing is changed:

1. the tree is a git checkout and the work root exists;
2. the tree-activity lease is free: no live lease and no maintenance lock for the tree (`core/tree_activity.py`, HI151);
3. none of `busy-processes` is running (`--allow-busy` skips only this check);
4. no tracked file is modified (untracked files are ignored);
5. `git fetch origin`, then `--ref` resolves (`origin/<ref>` first, then `<ref>`) and is checked out detached.

Then `run.sh` starts under `nohup` (and `setsid` when present). Metadata `meta.json` records id, ref, sha, argv, start time,
log path and pid. The wrapper writes `rc` last, atomically; its presence is the done marker.

## Status and wait

| State | Meaning |
|---|---|
| `running` | process alive, log grew within `--stall-minutes` (default 20) |
| `stalled` | process alive, no log growth for longer than that |
| `done rc=0` | exited 0 |
| `failed rc=N` | exited N; `rc` is absent when the process died without writing one (killed, host restart) |

`status` exits 0 for running/done, 1 for failed, 3 for stalled. `wait` polls until the run is not running, prints the log tail
and exits 0 (done), the command's rc or 1 (failed), 3 (stalled), or 124 (`--timeout` passed while still running).

A stalled run is still running; `wait` returns so you can look, it does not kill anything.
