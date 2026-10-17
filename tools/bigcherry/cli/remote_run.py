"""Detached lab runs on a configured GPU host: ``remote run | status | wait | list`` (MEN18).

One controller call does what every hand-written remote script did: preflight the campaign tree (tree-activity lease
free, host GPUs idle, tracked files clean), fetch and check out a ref detached, then launch ``python -m bigcherry <args>``
under ``nohup`` with a run id, a log and an exit-code file. The host, tree, work root and environment file all come from
the environment config (``[host.*]``); nothing here names an engine, the inner arguments carry their own ``--engine``.

Host-side layout, under ``<cache-root>/remote-runs/<run-id>/``: ``meta.json`` (id, ref, sha, argv, start time, log path,
pid), ``run.sh`` (the launched wrapper), ``run.log``, ``pid`` and ``rc`` (written last, atomically, when the command
exits -- its presence is the done marker).
"""

from __future__ import annotations

import json
import re
import secrets
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Callable, Protocol

from ..core import environment, paths

RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+-]*$")
_PROCESS = re.compile(r"^[A-Za-z0-9._+-]{1,15}$")

#: Exit codes of ``remote wait`` / ``remote status`` besides the command's own nonzero rc.
EXIT_STALLED = 3
EXIT_TIMEOUT = 124

DEFAULT_STALL_MINUTES = 20.0


class RemoteRunError(RuntimeError):
    pass


class PreflightRefused(RemoteRunError):
    """The host or tree was not in a state to start a run; nothing was launched."""


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str


class Transport(Protocol):
    def run(self, script: str) -> Completed: ...


class SshTransport:
    """Runs a bash script on the host over ssh (script on stdin, bytes so Windows does not turn LF into CRLF)."""

    def __init__(self, hostname: str) -> None:
        self.hostname = hostname

    def run(self, script: str) -> Completed:
        done = subprocess.run(
            ("ssh", "-o", "BatchMode=yes", self.hostname, "bash", "-se"),
            input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return Completed(done.returncode, done.stdout.decode("utf-8", "replace"))


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    state: str  # running | done | failed | stalled
    rc: int | None
    idle_seconds: int
    meta: dict

    def line(self) -> str:
        rc = "" if self.rc is None else f" rc={self.rc}"
        idle = f" (no log growth for {self.idle_seconds // 60} min)" if self.state == "stalled" else ""
        return f"{self.run_id} {self.state}{rc}{idle} ref={self.meta.get('ref', '?')}"


def _check_host(host: environment.Host) -> None:
    if not host.hostname or not host.repo or not host.cache_root:
        raise RemoteRunError("remote host needs hostname, repo and cache-root")
    for name, value in (("repo", host.repo), ("cache-root", host.cache_root), ("environment-file", host.environment_file)):
        if not PurePosixPath(value).is_absolute() or any(c.isspace() for c in value):
            raise RemoteRunError(f"host {name} must be an absolute POSIX path without whitespace, got {value!r}")
    if not host.busy_processes:
        raise RemoteRunError(
            f"host {host.name!r} has no busy-processes configured; list the process names that mean its GPUs are in use "
            "(config/environment.example.toml)"
        )
    for name in host.busy_processes:
        if not _PROCESS.fullmatch(name):
            raise RemoteRunError(f"busy-processes entry {name!r} is not an exact process name (max 15 chars)")


def runs_dir(host: environment.Host) -> str:
    return f"{host.cache_root.rstrip('/')}/remote-runs"


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


_LEASE_PY = """
import sys
from pathlib import Path
from bigcherry.core import tree_activity as t
work, tree = Path(sys.argv[1]), Path(sys.argv[2])
if t.maintenance_is_held(work, tree):
    print("maintenance lock held"); sys.exit(1)
live = t.list_live_leases(work, tree)
if live:
    print("live lease(s): " + ", ".join(f"{l.command}({l.run_id}) pid={l.pid}" for l in live)); sys.exit(1)
"""


def _wrapper(host: environment.Host, run_dir: str, argv: tuple[str, ...]) -> str:
    q = shlex.quote
    lines = [
        "#!/usr/bin/env bash",
        f'echo $$ > {q(run_dir + "/pid")}',
        f"cd {q(host.repo)}",
    ]
    if host.rocm:
        lines += [f"export ROCM_PATH={q(host.rocm)}", 'export PATH="$ROCM_PATH/bin:$PATH"']
    lines += [
        f"export BIGCHERRY_ENVIRONMENT={q(host.environment_file)}",
        "export PYTHONPATH=tools",
        'echo "[remote-run] start $(date -u +%FT%TZ) in $PWD at $(git rev-parse HEAD)"',
        f'"${{BIGCHERRY_REMOTE_PYTHON:-python3}}" -m bigcherry {shlex.join(argv)}',
        "rc=$?",
        'echo "[remote-run] done rc=$rc $(date -u +%FT%TZ)"',
        f'echo "$rc" > {q(run_dir + "/rc.tmp")} && mv {q(run_dir + "/rc.tmp")} {q(run_dir + "/rc")}',
        'exit "$rc"',
    ]
    return "\n".join(lines) + "\n"


def build_launch_script(
    host: environment.Host, *, run_id: str, ref: str, argv: tuple[str, ...], allow_busy: bool, started: float,
) -> str:
    _check_host(host)
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid run id {run_id!r}")
    if not _REF.fullmatch(ref):
        raise ValueError(f"invalid ref {ref!r}")
    if not argv:
        raise ValueError("expected -- <bigcherry args...>")
    if any("\n" in a or "\r" in a for a in argv):
        raise ValueError("arguments cannot contain newlines")
    q = shlex.quote
    run_dir = f"{runs_dir(host)}/{run_id}"
    meta = json.dumps({
        "id": run_id, "host": host.name, "ref": ref, "sha": "__SHA__", "argv": ["python", "-m", "bigcherry", *argv],
        "tree": host.repo, "environment-file": host.environment_file, "started-at": started,
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(), "log": f"{run_dir}/run.log",
        "pid": "__PID__",
    }, sort_keys=True)
    names = " ".join(q(n) for n in host.busy_processes)
    return "\n".join([
        "set -u",
        f"tree={q(host.repo)}", f"work={q(host.cache_root)}", f"dir={q(run_dir)}", f"ref={q(ref)}",
        f"allow_busy={1 if allow_busy else 0}",
        'py="${BIGCHERRY_REMOTE_PYTHON:-python3}"',
        'refuse() { echo "REFUSE $*"; exit 3; }',
        'git -C "$tree" rev-parse --git-dir >/dev/null 2>&1 </dev/null || refuse "not a git checkout: $tree"',
        'test -d "$work" || refuse "work root missing: $work"',
        f"lease_py={q(_LEASE_PY)}",
        'out=$(cd "$tree" && PYTHONPATH=tools "$py" -c "$lease_py" "$work" "$tree" </dev/null 2>&1) '
        '|| refuse "tree-activity: $(printf %s "$out" | tr "\\n" " ")"',
        'if [ "$allow_busy" != 1 ]; then',
        '  command -v pgrep >/dev/null || refuse "pgrep not available to check for GPU users"',
        '  busy=""',
        f"  for name in {names}; do",
        '    hit=$(pgrep -x -l "$name" </dev/null || true)',
        '    [ -z "$hit" ] || busy="$busy $(printf %s "$hit" | tr "\\n" ",")"',
        "  done",
        '  [ -z "$busy" ] || refuse "GPU host busy:$busy (use --allow-busy to override)"',
        "fi",
        'dirty=$(git -C "$tree" status --porcelain --untracked-files=no </dev/null)',
        '[ -z "$dirty" ] || refuse "tree has uncommitted tracked changes: $(printf %s "$dirty" | head -n 3 | tr "\\n" ";")"',
        'git -C "$tree" fetch -q origin </dev/null || refuse "git fetch origin failed in $tree"',
        'sha=$(git -C "$tree" rev-parse --verify -q "origin/$ref^{commit}" </dev/null '
        '|| git -C "$tree" rev-parse --verify -q "$ref^{commit}" </dev/null) || refuse "ref not found after fetch: $ref"',
        'git -C "$tree" checkout -q --detach "$sha" </dev/null || refuse "checkout of $sha failed"',
        'mkdir -p "$dir" || refuse "cannot create $dir"',
        "cat > \"$dir/run.sh\" <<'BC_REMOTE_RUN_EOF'",
        _wrapper(host, run_dir, argv).rstrip("\n"),
        "BC_REMOTE_RUN_EOF",
        f"cat > \"$dir/meta.tmpl\" <<'BC_REMOTE_META_EOF'\n{meta}\nBC_REMOTE_META_EOF",
        'launcher="nohup"; command -v setsid >/dev/null && launcher="setsid nohup"',
        '$launcher bash "$dir/run.sh" > "$dir/run.log" 2>&1 < /dev/null &',
        "for _ in $(seq 1 100); do [ -s \"$dir/pid\" ] && break; sleep 0.1; done",
        '[ -s "$dir/pid" ] || refuse "run did not start (no pid file in $dir)"',
        'pid=$(cat "$dir/pid")',
        'sed -e "s/__SHA__/$sha/" -e "s/\\"__PID__\\"/$pid/" "$dir/meta.tmpl" > "$dir/meta.json" && rm -f "$dir/meta.tmpl"',
        'echo "SHA=$sha"', 'echo "PID=$pid"', 'echo "LOG=$dir/run.log"',
    ]) + "\n"


def _markers(output: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in output.splitlines():
        for key in ("SHA", "PID", "LOG"):
            if line.startswith(key + "="):
                found[key.lower()] = line[len(key) + 1:].strip()
    return found


def launch(
    host: environment.Host, transport: Transport, *, ref: str, argv: tuple[str, ...], allow_busy: bool = False,
    run_id: str | None = None, clock: Callable[[], float] = time.time,
) -> dict:
    """Preflight, check out ``ref`` detached in the host's campaign tree and start the run. Returns its metadata."""
    run_id = run_id or new_run_id()
    started = clock()
    script = build_launch_script(host, run_id=run_id, ref=ref, argv=argv, allow_busy=allow_busy, started=started)
    done = transport.run(script)
    for line in done.stdout.splitlines():
        if line.startswith("REFUSE "):
            raise PreflightRefused(line[len("REFUSE "):])
    if done.returncode != 0:
        raise RemoteRunError(f"launch failed (rc={done.returncode}): {done.stdout.strip()[-600:]}")
    found = _markers(done.stdout)
    if not {"sha", "pid", "log"} <= found.keys():
        raise RemoteRunError(f"launch gave no run markers: {done.stdout.strip()[-600:]}")
    return {"id": run_id, "ref": ref, "sha": found["sha"], "pid": int(found["pid"]), "log": found["log"],
            "argv": ["python", "-m", "bigcherry", *argv], "started-at": started, "host": host.name}


def _check_run_id(run_id: str) -> None:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid run id {run_id!r}")


def status(host: environment.Host, transport: Transport, run_id: str, *,
           stall_minutes: float = DEFAULT_STALL_MINUTES) -> RunStatus:
    _check_run_id(run_id)
    q = shlex.quote
    script = "\n".join([
        f"dir={q(runs_dir(host) + '/' + run_id)}",
        '[ -f "$dir/meta.json" ] || { echo "ERR unknown run"; exit 4; }',
        'echo "META $(tr -d "\\n" < "$dir/meta.json")"',
        'echo "RC $(cat "$dir/rc" 2>/dev/null)"',
        'pid=$(cat "$dir/pid" 2>/dev/null); alive=0',
        '[ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null || alive=1',
        'echo "ALIVE $alive"', 'echo "NOW $(date +%s)"',
        'echo "LOGMTIME $(stat -c %Y "$dir/run.log" 2>/dev/null || echo 0)"',
    ]) + "\n"
    done = transport.run(script)
    if "ERR unknown run" in done.stdout:
        raise RemoteRunError(f"unknown run {run_id} on host {host.name!r}")
    if done.returncode != 0:
        raise RemoteRunError(f"status failed (rc={done.returncode}): {done.stdout.strip()[-400:]}")
    fields = {}
    for line in done.stdout.splitlines():
        key, _, value = line.partition(" ")
        fields[key] = value.strip()
    try:
        meta = json.loads(fields["META"])
        now, mtime = int(fields["NOW"]), int(fields["LOGMTIME"])
    except (KeyError, ValueError) as exc:
        raise RemoteRunError(f"unreadable status for {run_id}: {done.stdout.strip()[-400:]}") from exc
    idle = max(0, now - mtime)
    rc_text = fields.get("RC", "")
    if rc_text:
        rc = int(rc_text)
        return RunStatus(run_id, "done" if rc == 0 else "failed", rc, idle, meta)
    if fields.get("ALIVE") == "1":
        state = "stalled" if idle > stall_minutes * 60 else "running"
        return RunStatus(run_id, state, None, idle, meta)
    # The process is gone and never wrote its exit code: killed, or the host restarted.
    return RunStatus(run_id, "failed", None, idle, meta)


def tail_log(host: environment.Host, transport: Transport, run_id: str, lines: int) -> str:
    _check_run_id(run_id)
    log = f"{runs_dir(host)}/{run_id}/run.log"
    return transport.run(f"tail -n {int(lines)} {shlex.quote(log)} 2>&1 </dev/null\n").stdout


def wait(
    host: environment.Host, transport: Transport, run_id: str, *, timeout: float | None = None, poll: float = 15.0,
    stall_minutes: float = DEFAULT_STALL_MINUTES, sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunStatus | None:
    """Poll until the run is done, failed or stalled. ``None`` when ``timeout`` seconds pass first."""
    deadline = None if timeout is None else monotonic() + timeout
    while True:
        current = status(host, transport, run_id, stall_minutes=stall_minutes)
        if current.state != "running":
            return current
        if deadline is not None and monotonic() >= deadline:
            return None
        sleep(poll if deadline is None else max(0.0, min(poll, deadline - monotonic())))


def recent(host: environment.Host, transport: Transport, limit: int) -> list[str]:
    script = f"ls -1t {shlex.quote(runs_dir(host))} 2>/dev/null </dev/null | head -n {int(limit)}\n"
    return [i for i in transport.run(script).stdout.split() if RUN_ID.fullmatch(i)]


def _exit_code(result: RunStatus | None) -> int:
    if result is None:
        return EXIT_TIMEOUT
    if result.state == "stalled":
        return EXIT_STALLED
    if result.state == "done":
        return 0
    return result.rc if result.rc else 1


def _resolve_host(name: str | None) -> environment.Host:
    return environment.load_default(repo_root=paths.primary_root()).host(name)


def cmd_remote(args) -> int:
    try:
        host = _resolve_host(args.host)
        transport = SshTransport(host.hostname)
        action = args.remote_action
        if action == "run":
            argv = tuple(args.argv)
            if argv[:1] == ("--",):
                argv = argv[1:]
            meta = launch(host, transport, ref=args.ref, argv=argv, allow_busy=args.allow_busy)
            print(f"run-id: {meta['id']}\nhost: {host.name}\nref: {meta['ref']} ({meta['sha']})\npid: {meta['pid']}\n"
                  f"log: {meta['log']}\nnext: python -m bigcherry remote wait {meta['id']} --host {host.name}")
            return 0
        if action == "status":
            result = status(host, transport, args.run_id, stall_minutes=args.stall_minutes)
            print(result.line())
            return 3 if result.state == "stalled" else (0 if result.state in ("running", "done") else 1)
        if action == "wait":
            result = wait(host, transport, args.run_id, timeout=args.timeout, poll=args.poll,
                          stall_minutes=args.stall_minutes)
            print(result.line() if result else f"{args.run_id} still running after {args.timeout}s")
            print(f"--- last {args.tail} log lines ---")
            print(tail_log(host, transport, args.run_id, args.tail), end="")
            return _exit_code(result)
        for run_id in recent(host, transport, args.limit):
            print(status(host, transport, run_id, stall_minutes=args.stall_minutes).line())
        return 0
    except PreflightRefused as exc:
        print(f"remote: refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RemoteRunError, environment.EnvironmentError_) as exc:
        print(f"remote: {exc}", file=sys.stderr)
        return 1
