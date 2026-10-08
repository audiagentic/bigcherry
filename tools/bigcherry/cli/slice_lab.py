"""Execute slice scripts on a configured lab through the GPU queue."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..core import environment, paths

BRANCH = re.compile(r"^(?:feat|fix|perf|chore|docs|test|refactor|ci|build)/[a-z0-9]+(?:-[a-z0-9]+)+$|^bump/b[0-9]+$")


@dataclass(frozen=True)
class LabCommand:
    ssh: tuple[str, ...]
    script: str


def build_lab_command(
    *, branch: str, host: environment.Host, script: str,
    arguments: tuple[str, ...] = (), run_id: str | None = None,
) -> LabCommand:
    if not BRANCH.fullmatch(branch):
        raise ValueError(f"invalid branch: {branch}")
    path = PurePosixPath(script)
    if (not script or path.is_absolute() or ".." in path.parts
            or not script.startswith("tools/lab/") or script.startswith("-")):
        raise ValueError("script must be a repository-relative tools/lab/ script")
    if any("\n" in arg or "\r" in arg for arg in (script, *arguments)):
        raise ValueError("script and arguments cannot contain newlines")
    if not host.hostname or not host.repo or not host.cache_root:
        raise ValueError("lab host needs hostname, repo, cache-root")
    for value in (host.repo, host.cache_root):
        if not PurePosixPath(value).is_absolute() or any(char.isspace() for char in value):
            raise ValueError("lab repo and cache-root must be absolute POSIX paths")

    token = run_id if run_id is not None else uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[a-f0-9]{12}", token):
        raise ValueError("invalid lab run identifier")
    label = f"pa47-{branch.replace('/', '-')}-{token}"
    q = shlex.quote
    stage = f"{host.cache_root}/worktrees/{label}"
    private_ref = f"refs/bigcherry-lab/{token}"
    source_ref = q("refs/heads/" + branch)
    wrapper = f"{host.cache_root}/runs/{label}.sh"
    jobs = f"{host.cache_root}/runs/{label}.jobs"
    exec_line = shlex.join(["bash", script, *arguments])
    # These two scripts submit their own queue jobs. An outer SCRIPT lock
    # would deadlock when their child queue acquires the same GPU lock.
    kind = "QUEUE" if path.name in ("queue.sh", "queue-meta-mem.sh", "queue-env-ab.sh") else "SCRIPT"
    lines = [
        "set -Eeuo pipefail",
        f"primary={q(host.repo)}",
        f"work={q(host.cache_root)}",
        f"stage={q(stage)}",
        f"private_ref={q(private_ref)}",
        f"wrapper={q(wrapper)}",
        f"jobs={q(jobs)}",
        'test "$(git -C "$primary" branch --show-current)" = main || { echo "lab primary must stay on main" >&2; exit 1; }',
        'test -z "$(git -C "$primary" status --porcelain)" || { echo "lab primary is dirty" >&2; exit 1; }',
        'mkdir -p "$work/worktrees" "$work/runs"',
        'test ! -e "$stage" || { echo "lab worktree already exists" >&2; exit 1; }',
        'if git -C "$primary" show-ref --verify --quiet "$private_ref"; then',
        '  echo "lab private ref already exists" >&2; exit 1',
        'fi',
        'active=0',
        "cleanup() {",
        '  rc=$?',
        '  trap - EXIT',
        '  if [ "$active" -eq 1 ]; then',
        '    if ! git -C "$primary" worktree remove "$stage"; then',
        '      echo "slice lab: cleanup refused dirty or locked worktree: $stage" >&2',
        '      rc=1',
        '    fi',
        '  fi',
        '  if git -C "$primary" show-ref --verify --quiet "$private_ref"; then',
        '    git -C "$primary" update-ref -d "$private_ref" || rc=1',
        '  fi',
        '  rm -f "$wrapper" "$jobs"',
        '  exit "$rc"',
        "}",
        "trap cleanup EXIT",
        f'git -C "$primary" fetch --no-tags origin {source_ref}:"$private_ref"',
        'head=$(git -C "$primary" rev-parse --verify "$private_ref^{commit}")',
        'git -C "$primary" worktree add --detach "$stage" "$head"',
        'active=1',
        "cat > \"$wrapper\" <<'PA47_WRAPPER'",
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        f"cd {q(stage)}",
        f"exec {exec_line}",
        "PA47_WRAPPER",
        f"printf '%s\\n' {q(kind + ' ' + label + ' ' + wrapper)} > \"$jobs\"",
        'export BC_PRIMARY_ROOT="$primary" BIGCHERRY_WORK_ROOT="$work"',
        'export BIGCHERRY_PROJECT_ROOT="$stage"',
        f'export BIGCHERRY_ARTIFACT_ROOT="$work/runs/{label}/artifacts"',
        'export PYTHONDONTWRITEBYTECODE=1 BC_QUEUE_STREAM=1',
        'bash "$stage/tools/lab/plan-qualification/queue.sh" "$jobs"',
    ]
    return LabCommand(("ssh", "-o", "BatchMode=yes", host.hostname, "bash", "-se"), "\n".join(lines) + "\n")


def cmd_slice_lab(args) -> int:
    try:
        host = environment.load_default(repo_root=paths.primary_root()).host(args.host)
        argv = tuple(args.script)
        if argv[:1] == ("--",):
            argv = argv[1:]
        if not argv:
            raise ValueError("expected -- <tools/lab/script.sh> [args...]")
        command = build_lab_command(
            branch=args.branch, host=host,
            script=argv[0], arguments=argv[1:],
        )
        if args.dry_run:
            print(shlex.join(command.ssh))
            print(command.script, end="")
            return 0
        process = subprocess.Popen(
            command.ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdin is not None
        process.stdin.write(command.script)
        process.stdin.close()
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        return process.wait()
    except (OSError, ValueError, environment.EnvironmentError_) as exc:
        print(f"slice lab: {exc}", file=sys.stderr)
        return 1
