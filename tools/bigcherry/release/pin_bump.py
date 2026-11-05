"""Single-tree pin-bump orchestrator.

Each phase either records its proof and resume cursor or stops with a
structured failure envelope. Multi-tree completion remains the authority of
`pin-status --complete --all-remotes`; this runner never coordinates trees.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, TypeVar

from ..build import source_compile_check
from ..core import components as components_module, engines, paths
from ..core import tree_activity
from ..core.context import ProjectContext
from ..engine_adapters import BumpCheckError, adapter
from ..patch import apply as patch_apply
from ..patch import disposition as patch_disposition
from ..patch import overlay as patch_overlay
from ..patch import patchset as patch_patchset
from ..patch import build_evidence
from ..patch import rebase as patch_rebase
from ..patch import registry as patch_registry
from ..source import identity as source_identity
from .. import pin_transition
from .. import recipes as recipes_module
from . import records as releases
from . import notes as release_notes
from .pin import recipes_path, resolve_pin_sha

PHASES = (
    "preflight",
    "declare",
    "pull",
    "drift",
    "audit",
    "patch-lint",
    "coverage",
    "apply",
    "compile",
    "evidence",
    "lane-build",
    "cpu-smoke",
    "reaudit",
    "complete",
)


class PinBumpStop(Exception):
    """Raised by any phase to STOP the orchestrator. Carries everything
    needed to build the structured failure envelope -- callers should
    never need to re-derive context from a bare message string."""

    def __init__(
        self,
        phase: str,
        code: str,
        summary: str,
        *,
        human_required: bool = True,
        retryable: bool = True,
        evidence: dict[str, Any] | None = None,
        recommended_actions: Sequence[str] = (),
    ):
        super().__init__(summary)
        self.phase = phase
        self.code = code
        self.summary = summary
        self.human_required = human_required
        self.retryable = retryable
        self.evidence = evidence or {}
        # Canonical immutable shape: callers pass lists; the envelope and
        # any persistence see one tuple form (never a mutable list).
        self.recommended_actions = tuple(recommended_actions)
        # Set by run() before re-raising, once a run_id has been assigned --
        # never set here, since most call sites raise before a run exists.
        self.run_id: str | None = None
        self.target: dict[str, str] | None = None
        self.transition_commit: str = ""
        self.tree: dict[str, str] | None = None


@dataclass
class PinBumpState:
    schema_version: int
    run_id: str
    from_ref: str
    from_sha: str
    to_ref: str
    to_sha: str
    transition_commit: str
    tree_name: str
    tree_path: str
    completed_phases: list[str] = field(default_factory=list)
    next_phase: str = PHASES[0]
    #: schema 2+ only. Binds WHICH selector (always "source" kind now) this
    #: run was started with, so --resume reconstructs the identical
    #: selector rather than silently
    #: reinterpreting whatever the resuming invocation happens to pass.
    #: Deliberately NOT the source's patch_set_id: patch IMPLEMENTATION
    #: hashes legitimately change during reconciliation, so binding on
    #: content identity would make ordinary --resume impossible.
    #: selector_patch_ids freezes MEMBERSHIP (which patch IDs) while still
    #: allowing their contents to be repaired between phases.
    selector_kind: str = ""
    selector_name: str = ""
    selector_patch_ids: tuple[str, ...] = ()
    #: sha256 of the exact rebase-recipe.json the coverage phase wrote --
    #: re-checked before apply so a report modified/replaced on disk
    #: between phases (by hand, or a bug) is caught, on top of
    #: apply_known_good()'s own live source/ref/patch-set staleness checks.
    coverage_report_sha256: str = ""
    #: PA44-A: exact effective vendor-tree identity that passed (or was
    #: explicitly skipped at) the compile phase. A resumed in-flight bump
    #: re-runs compile when this identity no longer matches the live tree.
    compile_tree_oid: str = ""
    compile_status: str = ""
    #: MEN15: "passed" (the RAM-only tiny-model smoke ran with no GPU visible) or "skipped" (--skip-cpu-smoke), recorded
    #: here and so in every resume of the run
    cpu_smoke_status: str = ""
    #: MEN17: "passed" when `--accept` ran the engine's hardware acceptance against its stored baselines; "" when not asked
    accept_status: str = ""
    #: the recipe ``[platform.*]`` the compile gate configures with (``--platform``); "" when the run named none. Saved so
    #: a resumed run builds the same way.
    compile_platform: str = ""
    #: MEN16: the recipe ``[build.*]`` profile the lane-build phase builds (``--build``). Saved so a resumed run builds the
    #: same lane; "" when the run named none.
    lane_build_name: str = ""
    #: MEN16: "passed" (the bumped engine was built through its lane), "skipped-by-override" (--cpu-smoke-build named a
    #: build, so no lane was built) or "skipped" (--skip-cpu-smoke, nothing needs a build).
    lane_build_status: str = ""
    #: MEN16: the lane (``<source>:<build>:<platform>``), campaign build plan, build directory and launcher sha256 of the
    #: lane build; for "skipped-by-override" only ``lane_build_dir`` (the override) is set.
    lane_build_lane: str = ""
    lane_build_plan_id: str = ""
    lane_build_dir: str = ""
    lane_build_binary_sha256: str = ""
    #: MEN16: the explicit --cpu-smoke-build override, saved so a resume reuses it and a different one is refused.
    cpu_smoke_build_override: str = ""
    evidence_gate: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "target": {
                "from_ref": self.from_ref,
                "from_sha": self.from_sha,
                "to_ref": self.to_ref,
                "to_sha": self.to_sha,
            },
            "transition_commit": self.transition_commit,
            "tree": {"name": self.tree_name, "path": self.tree_path},
            "resume": {
                "completed_phases": list(self.completed_phases),
                "next_phase": self.next_phase,
            },
        }
        # PA34 (dev-gpt-agent req_41a3133e657340c7 Q3): the persisted
        # membership freeze is its OWN resume record, not a serialization of
        # the canonical SelectorIdentity -- the identity schema is owned
        # exclusively by campaign.resolution. Renamed "selector" ->
        # "selection_freeze" at schema 3 so no downstream consumer owns an
        # object named "selector" with a competing shape. Never both keys.
        if self.schema_version >= 3:
            d["selection_freeze"] = {
                "kind": self.selector_kind,
                "name": self.selector_name,
                "patch_ids": list(self.selector_patch_ids),
            }
            d["coverage_report_sha256"] = self.coverage_report_sha256
            d["compile"] = {
                "status": self.compile_status,
                "tree_oid": self.compile_tree_oid,
                "platform": self.compile_platform,
            }
            d["lane_build"] = {
                "status": self.lane_build_status,
                "build": self.lane_build_name,
                "lane": self.lane_build_lane,
                "build_plan_id": self.lane_build_plan_id,
                "build_dir": self.lane_build_dir,
                "binary_sha256": self.lane_build_binary_sha256,
            }
            d["cpu_smoke"] = {"status": self.cpu_smoke_status, "build_override": self.cpu_smoke_build_override}
            d["accept"] = {"status": self.accept_status}
            d["evidence_gate"] = self.evidence_gate
        return d

    def save(self, state_dir: Path) -> Path:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def advance(self, state_dir: Path, phase: str, next_phase: str) -> None:
        """Persist the phase completion and resume cursor in one operation."""
        self.completed_phases.append(phase)
        self.next_phase = next_phase
        self.save(state_dir)

    @classmethod
    def load(cls, state_dir: Path) -> PinBumpState:
        data = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
            # Schema 2's selector is promoted in memory to schema 3's
        # selection_freeze; schema 1 stays unbound and cannot be resumed.
        raw_version = int(data["schema_version"])
        # Fail on a structurally incomplete state (missing run_id/target/
        # tree/resume) the same way any other schema does, BEFORE the
        # schema-2 selector-shape check below -- a state missing its base
        # fields is generically invalid, not specifically "legacy/unbound".
        for _required_key in ("run_id", "target", "tree", "resume"):
            if _required_key not in data:
                raise KeyError(_required_key)
        if raw_version >= 3:
            freeze = data.get("selection_freeze") or {}
        elif raw_version == 2:
            # Validate the schema-2 selector before promotion, or a
            # missing/malformed binding could be accepted as schema 3.
            freeze = data.get("selector")
            if (
                not isinstance(freeze, dict)
                # A foreign kind would be incorrectly resolved as a source
                # by _selector_patch_ids on resume.
                or freeze.get("kind") != "source"
                or not isinstance(freeze.get("name"), str)
                or not freeze.get("name")
                or not isinstance(freeze.get("patch_ids"), list)
                or not all(isinstance(pid, str) for pid in freeze.get("patch_ids", []))
            ):
                raise PinBumpStop(
                    "resume",
                    "LEGACY_STATE_SELECTOR_UNBOUND",
                    "this schema-2 state's \"selector\" key is missing or "
                    "malformed -- it cannot prove which selector this run "
                    "was started with, so resuming it could silently apply "
                    "a different composition than the original invocation "
                    "intended",
                    recommended_actions=[
                        "start a fresh run instead of resuming this one",
                        "if this state must be rescued, inspect its coverage "
                        "report (if the coverage phase already ran) to "
                        "recover its real selector by hand before resuming",
                    ],
                )
            raw_version = 3
        else:
            freeze = {}
        compile_state = data.get("compile") or {}
        cpu_smoke_state = data.get("cpu_smoke") or {}
        accept_state = data.get("accept") or {}
        lane_build_state = data.get("lane_build") or {}
        return cls(
            schema_version=raw_version,
            run_id=data["run_id"],
            from_ref=data["target"]["from_ref"],
            from_sha=data["target"]["from_sha"],
            to_ref=data["target"]["to_ref"],
            to_sha=data["target"]["to_sha"],
            transition_commit=data["transition_commit"],
            tree_name=data["tree"]["name"],
            tree_path=data["tree"]["path"],
            completed_phases=list(data["resume"]["completed_phases"]),
            next_phase=data["resume"]["next_phase"],
            selector_kind=freeze.get("kind", ""),
            selector_name=freeze.get("name", ""),
            selector_patch_ids=tuple(freeze.get("patch_ids", ())),
            coverage_report_sha256=data.get("coverage_report_sha256", ""),
            compile_tree_oid=str(compile_state.get("tree_oid", "")),
            compile_status=str(compile_state.get("status", "")),
            compile_platform=str(compile_state.get("platform", "")),
            cpu_smoke_status=str(cpu_smoke_state.get("status", "")),
            accept_status=str(accept_state.get("status", "")),
            cpu_smoke_build_override=str(cpu_smoke_state.get("build_override", "")),
            lane_build_status=str(lane_build_state.get("status", "")),
            lane_build_name=str(lane_build_state.get("build", "")),
            lane_build_lane=str(lane_build_state.get("lane", "")),
            lane_build_plan_id=str(lane_build_state.get("build_plan_id", "")),
            lane_build_dir=str(lane_build_state.get("build_dir", "")),
            lane_build_binary_sha256=str(lane_build_state.get("binary_sha256", "")),
            evidence_gate=data.get("evidence_gate", {}),
        )


def failure_envelope(
    run_id: str, target: dict, transition_commit: str, tree: dict, exc: PinBumpStop
) -> dict:
    return {
        "schema_version": 1,
        "operation": "pin-bump",
        "status": "STOPPED",
        "run_id": run_id,
        "target": target,
        "transition_commit": transition_commit,
        "phase": exc.phase,
        "tree": tree,
        "failure": {
            "code": exc.code,
            "human_required": exc.human_required,
            "retryable": exc.retryable,
            "summary": exc.summary,
            "evidence": exc.evidence,
            "recommended_actions": list(exc.recommended_actions),
        },
    }


def _git_result(root: Path, *args: str, **kwargs) -> subprocess.CompletedProcess[str]:
    """One git invocation; callers retain their own failure-code policy."""
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, **kwargs
    )


def _git(root: Path, *args: str) -> str:
    result = _git_result(root, *args)
    if result.returncode != 0:
        raise PinBumpStop(
            "preflight",
            "GIT_COMMAND_FAILED",
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}",
            evidence={"args": list(args), "stderr": result.stderr.strip()},
            recommended_actions=[
                "inspect the repo state directly",
                "rerun with --resume",
            ],
        )
    return result.stdout.strip()


def require_clean_controller_checkout(repo_root: Path) -> None:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise PinBumpStop(
            "preflight",
            "CONTROLLER_DIRTY",
            "the bigcherry controller checkout has uncommitted changes",
            evidence={"git_status_porcelain": status},
            recommended_actions=[
                "commit or revert the uncommitted changes, then retry"
            ],
        )


def check_overlay_self_heal(audit_report: dict) -> tuple[bool, list[str]]:
    """The one narrow, provably-safe auto-repair carve-out (HI150 round 2):
    ONLY when audit's failed checks are exactly {overlay.vendor_sync}, and
    every drifted file is overlay-owned/not-upstream-tracked/UTF-8/differs
    from the current overlay ONLY by newline normalization -- exactly what
    _write_overlay_snapshot/_copy_overlay already fix on the next real
    apply. Returns (safe_to_self_heal, drifted_paths). Never inspects
    anything beyond the audit report already computed -- no new mechanism,
    no invented "previous materialization" history (gpt retracted that
    clause; see HI150 notes)."""
    failed_checks = {
        check["id"]
        for check in audit_report.get("checks", ())
        if not check.get("ok", True)
    }
    if failed_checks != {"overlay.vendor_sync"}:
        return False, []
    overlay_check = next(
        check
        for check in audit_report["checks"]
        if check["id"] == "overlay.vendor_sync"
    )
    drifted: list[str] = [str(item) for item in (overlay_check.get("actual", ()) or ())]
    return True, drifted


def acquire_maintenance_lock(project_root: Path, engine: str) -> tree_activity.MaintenanceLock:
    """Returns an UNACQUIRED lock -- the caller uses it as `with lock:`, whose
    __enter__ does the single real acquire() call. (Found live on pin-bump's
    first real invocation: this used to call .acquire() itself AND get used
    as a context manager, double-acquiring the same lock in one process and
    tripping its own "already held" check.)"""
    from ..core.context import ProjectContext

    context = ProjectContext.resolve(engine=engine, project_root=project_root)
    return tree_activity.MaintenanceLock(context.work_root, context.project_root)


def _sync_campaign_mirror_best_effort(*, engine: str, target_ref: str, revision: str) -> None:
    """Bring the new pin into the campaign build mirror (ProjectContext.
    upstream_repo), which does not learn tags from the vendor checkout. Found
    live on three bumps (b10687, b10692, b11126). Best-effort: never raises
    PinBumpStop, but always warns when the mirror still lacks the tag."""
    from ..core.context import ProjectContext
    from ..source import upstream

    warning = upstream.sync_mirror_ref(
        ProjectContext.resolve(engine=engine).upstream_repo, target_ref, revision or None
    )
    if warning:
        print(f"pin-bump: WARNING {warning}", file=sys.stderr)

def run_phase_preflight(
    *, repo_root: Path, engine: str, vendor_root: Path, target_ref: str
) -> tuple[str, str]:
    """Returns (resolved_from_ref, resolved_to_sha). Requires a clean
    controller checkout and refuses an incompatible existing transition
    marker (RE48's invariant, reused rather than re-derived)."""
    require_clean_controller_checkout(repo_root)
    marker_path = pin_transition.marker_path(engine)
    if (
        marker_path.is_file()
        and pin_transition.committed_state(marker_path) != "committed-clean"
    ):
        raise PinBumpStop(
            "preflight",
            "TRANSITION_MARKER_UNCOMMITTED",
            "an existing pin-transition marker is uncommitted -- a prior bump "
            "was declared but not committed",
            evidence={"marker_path": str(marker_path)},
            recommended_actions=[
                "commit the marker together with the recipes file, or delete "
                "it if that bump was abandoned",
            ],
        )
    try:
        to_sha = resolve_pin_sha(engine, target_ref, vendor_root)
    except Exception as exc:  # upstream.UpstreamError et al.
        raise PinBumpStop(
            "preflight",
            "UNRESOLVABLE_PIN",
            f"could not resolve {target_ref!r}: {exc}",
            evidence={"target_ref": target_ref},
            recommended_actions=[
                "verify the ref exists upstream",
                "rerun with --resume",
            ],
        ) from exc
    current_pinned = recipes_module.pinned(recipes_path(engine))
    return current_pinned, to_sha


def run_phase_drift(*, engine: str, vendor_root: Path, target_ref: str) -> None:
    """Run the engine's source drift checks (its adapter's ``source_drift_checks``) on the freshly pulled upstream tree.

    Every check runs, and the first that finds drift STOPs the bump under its own code with the failures by file, so one run
    shows the whole port. llama.cpp declares none."""

    resume = f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume"
    try:
        checks = adapter(engine).source_drift_checks(ProjectContext.resolve(engine=engine), vendor_root)
    except (ValueError, OSError) as exc:
        raise PinBumpStop(
            "drift",
            "SOURCE_DRIFT_UNCHECKED",
            f"the source drift checks of {engine} could not be set up: {type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=[f"fix the engine declaration, then {resume}"],
        ) from exc
    for check in checks:
        try:
            failures = check.run()
        except BumpCheckError as exc:
            raise PinBumpStop(
                "drift",
                "SOURCE_DRIFT_UNCHECKED",
                f"source drift check {check.id} could not run: {exc}",
                evidence={"check": check.id, "error": str(exc)},
                recommended_actions=[f"fix the cause, then {resume}"],
            ) from exc
        if failures:
            count = sum(len(problems) for problems in failures.values())
            raise PinBumpStop(
                "drift",
                check.failure_code,
                f"{check.id}: {count} failing block(s)/rule(s) in {len(failures)} file(s) against {target_ref}",
                evidence={"check": check.id, "failures": failures},
                recommended_actions=[check.fix_hint, f"then {resume}"],
            )


def run_phase_declare(*, repo_root: Path, engine: str, vendor_root: Path, target_ref: str) -> str:
    """Calls the existing, already-real repin() + pin_transition.write(),
    then commits the recipes file + the marker together. Returns the
    declaring commit SHA."""
    recipes_file = recipes_path(engine)
    old = recipes_module.repin(target_ref, recipes_file)
    # What moves with the pin: the release version beside it (engines whose release line follows upstream) and every
    # component's built-against, so the declarations the tests compare never lag the pin.
    if engines.load(repo_root, engine).version_source == "pinned-version":
        recipes_module.set_pinned_version(recipes_module.version_of_tag(target_ref), recipes_file)
    declarations = components_module.repin_components(repo_root, engine, target_ref)
    from_sha = resolve_pin_sha(engine, old, vendor_root)
    to_sha = resolve_pin_sha(engine, target_ref, vendor_root)
    declaring_before = _git(repo_root, "rev-parse", "HEAD")
    marker = pin_transition.marker_path(engine)
    pin_transition.write(from_sha, to_sha, target_ref, declaring_before, marker)
    _git(
        repo_root,
        "add",
        recipes_file.relative_to(repo_root).as_posix(),
        marker.relative_to(repo_root).as_posix(),
        *(path.relative_to(repo_root).as_posix() for path in declarations),
    )
    _git(
        repo_root,
        "commit",
        "-m",
        f"pin: {old} -> {target_ref} ({to_sha[:8]}) -- rebase in flight (pin-bump orchestrator)",
    )
    return _git(repo_root, "rev-parse", "HEAD")


def failed_edits(entry: dict) -> list[dict]:
    """The edits of one rebase-report patch entry that did not apply, reduced
    to what a reconciler needs: which file, which edit, and why."""
    return [
        {
            "path": file_entry["path"],
            "edit_id": edit["edit_id"],
            "reason_code": edit.get("reason_code"),
            "actual_matches": edit.get("actual_matches"),
            "expect_matches": edit.get("expect_matches"),
        }
        for file_entry in entry.get("files", ())
        for edit in file_entry.get("edits", ())
        if edit.get("status") not in ("applied-clean", "already-applied")
    ]


def stop_on_recipe_failures(*, phase: str, recipe_report: dict) -> None:
    """A recipe-selected patch that does not apply blocks the bump (nothing
    can be built), and no disposition can excuse it. Every failing recipe
    patch is reported in one stop, each with its failing edits, so the whole
    reconciliation list is known from a single run."""
    failures = [
        {
            "patch_id": entry["patch_id"],
            "status": entry["status"],
            "requires": list(entry.get("requires", ())),
            "failed_edits": failed_edits(entry),
        }
        for entry in recipe_report.get("patches", ())
        if entry["status"] not in patch_disposition.CLEAN_STATUSES
    ]
    if not failures:
        return
    raise PinBumpStop(
        phase,
        "RECIPE_PATCHES_FAILED",
        f"{len(failures)} recipe-selected patch(es) do not apply to the new revision",
        evidence={"patch_failures": failures},
        recommended_actions=[
            "reconcile each listed patch in its own package (failed_edits names the anchors)",
            "if UPSTREAM_ABSORBED: retire the patch (state superseded) via the patch-lifecycle procedure",
            "rerun `bigcherry --engine <engine> pin-bump --resume`",
        ],
    )


def classify_patch_coverage(
    *,
    all_report: dict,
    recipe_report: dict,
    catalog_states: dict[str, str],
    dispositions_dir: Path,
    target_revision: str,
    focal_probe: Callable[[str], dict | None],
) -> dict:
    """Coverage of every non-rejected patch against the new revision, without
    stopping the bump for a patch that is not in the build.

    The recipe report is the authority for recipe-selected patches. For any
    other patch that the all-patches probe reports as bad, `focal_probe`
    re-probes it over the real source selection: a patch that is clean there
    only collided with patches it is never built with (`composition_only`),
    and is not a failure against this upstream. What remains is `outstanding`:
    the list of patches to reconcile after the bump."""
    recipe_entries = {entry["patch_id"]: entry for entry in recipe_report.get("patches", ())}
    merged = {entry["patch_id"]: entry for entry in all_report.get("patches", ())}
    merged.update(recipe_entries)
    result = patch_disposition.compute_coverage(
        catalog_states=catalog_states,
        all_report={"patches": list(merged.values())},
        recipe_patch_ids=frozenset(recipe_entries),
        dispositions=patch_disposition.list_dispositions(dispositions_dir),
        target_revision=target_revision,
    )
    composition_only: list[str] = []
    outstanding: list[dict] = []
    for patch_id in result.uncovered_patch_ids:
        entry = merged.get(patch_id)
        if entry is None:
            outstanding.append({"patch_id": patch_id, "status": "UNDISCOVERED", "probe": "none",
                                "failed_edits": []})
            continue
        probe = "all-patches"
        if entry["status"] != "UPSTREAM_ABSORBED":
            focal = focal_probe(patch_id)
            if focal is not None:
                if focal["status"] in patch_disposition.CLEAN_STATUSES:
                    composition_only.append(patch_id)
                    continue
                entry, probe = focal, "focal-overlay"
        outstanding.append({"patch_id": patch_id, "status": entry["status"], "probe": probe,
                            "failed_edits": failed_edits(entry)})
    coverage = result.as_dict()
    coverage["composition_only_patch_ids"] = composition_only
    coverage["outstanding"] = outstanding
    coverage["complete"] = not outstanding
    return coverage


PATCH_TESTS_DIR = Path("tools") / "tests" / "patch"


def select_patch_test_modules(repo_root: Path, engine: str) -> list[str]:
    """The offline patch test modules a bump of ``engine`` can affect (MEN24).

    An engine owns the test files named ``test_<engine>_*.py``. A bump of an engine that owns files runs exactly those;
    a bump of an engine that owns none (llama.cpp, whose tests predate the convention) runs every file no other engine
    owns. An engine cannot affect another engine's tests, so neither selection crosses over."""
    from ..core import engines

    files = sorted(path.stem for path in (repo_root / PATCH_TESTS_DIR).glob("test_*.py"))
    owned = [stem for stem in files if stem.startswith(f"test_{engine}_")]
    if owned:
        selected = owned
    else:
        others = tuple(f"test_{other}_" for other in engines.names(repo_root) if other != engine)
        selected = [stem for stem in files if not stem.startswith(others)]
    if not selected:
        raise ValueError(f"no patch test modules select for engine {engine!r} under {PATCH_TESTS_DIR}")
    return selected


def exec_capable_tempdir(fallback_parent: Path) -> Path | None:
    """The directory test subprocesses should use as TMPDIR, or None when the system one already allows execution.
    Some tests compile a small program into a temporary directory and run it; where that directory is mounted
    ``noexec`` (a hardened container's /tmp) they fail with "permission denied", which a bump would report as a
    failure of the new revision. The fallback lives under the work root, which is exec-capable by construction."""
    probe_dir = Path(tempfile.gettempdir())
    try:
        with tempfile.NamedTemporaryFile("w", dir=probe_dir, suffix=".sh", delete=False) as handle:
            handle.write("#!/bin/sh\nexit 0\n")
            probe = Path(handle.name)
        try:
            probe.chmod(0o755)
            if subprocess.run([str(probe)], capture_output=True).returncode == 0:
                return None
        finally:
            probe.unlink(missing_ok=True)
    except OSError:
        pass
    fallback_parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="pin-bump-tests-", dir=fallback_parent))


def run_patch_tests(repo_root: Path, engine: str) -> list[str]:
    """Run the bumped engine's offline patch mechanics tests and return the failing test ids.
    Called before apply, while the vendor tree is the pristine new revision:
    the fixtures copy vendor files, so an already-patched tree would turn every
    fail-closed test red for reasons unrelated to upstream."""
    env = {**os.environ, "PYTHONPATH": str(repo_root / "tools")}
    scratch = exec_capable_tempdir(ProjectContext.resolve(engine=engine).work_root / "tmp")
    if scratch is not None:
        env["TMPDIR"] = str(scratch)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", *select_patch_test_modules(repo_root, engine)],
            cwd=repo_root / PATCH_TESTS_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    failing = [
        line.split(": ", 1)[1].strip()
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.startswith(("FAIL: ", "ERROR: "))
    ]
    if completed.returncode != 0 and not failing:
        failing = [f"patch test suite exited {completed.returncode} without a parsable failure"]
    return failing


@dataclass
class PinBumpResult:
    ok: bool
    state: PinBumpState
    coverage: dict


def default_report_dir(engine: str, target_ref: str) -> Path:
    """Where a run state and reports live unless --report-dir says otherwise: one directory per engine and target."""
    return paths.ARTIFACTS / "pin-bump" / engine / f"resume-{target_ref}"


def run(
    *,
    engine: str,
    target_ref: str,
    source_name: str | None = None,
    root: Path | None = None,
    dispositions_dir: Path | None = None,
    resume: bool = False,
    report_dir: Path | None = None,
    skip_compile: bool = False,
    cpu_smoke_build: Path | None,
    skip_cpu_smoke: bool,
    accept: bool,
    platform_name: str | None,
    build_name: str | None,
) -> PinBumpResult:
    """Run a fresh single-tree bump or resume the recorded run.

    Fresh runs default to source `bigcherry` where declared. Resumes reuse
    the persisted selector, compile platform and lane build unless explicitly
    given matching inputs. Stop conditions carry their original failure codes.
    """
    repo_root = paths.REPO_ROOT
    vendor_root = root if root is not None else ProjectContext.resolve(engine=engine).vendor_root
    dispositions_dir = dispositions_dir or paths.DISPOSITIONS
    report_dir = report_dir or default_report_dir(engine, target_ref)

    # Resume state loading can itself fail; the envelope still needs context.
    state: PinBumpState | None = None
    try:
        # Never treat a missing resume state as a fresh run. Keep validation
        # inside this try so every stop gets a structured envelope.
        if resume:
            if not (report_dir / "state.json").is_file():
                raise PinBumpStop(
                    "resume",
                    "RESUME_STATE_MISSING",
                    f"--resume was given but no state.json exists under {report_dir}",
                    evidence={"report_dir": str(report_dir)},
                    recommended_actions=[
                        "omit --resume to start a fresh run, or point "
                        "--report-dir at the run you meant to resume",
                    ],
                )
            state = _load_state_or_stop(report_dir)
            _validate_resume(state, target_ref=target_ref, vendor_root=vendor_root)
            ensure_vendor_at_target(vendor_root, to_ref=state.to_ref, to_sha=state.to_sha)
            resolved_kind, resolved_name = _resume_selector(
                state,
                source_name=source_name,
            )
            platform_name = _resume_platform(state, platform_name=platform_name)
            build_name, cpu_smoke_build = _resume_lane_build_inputs(
                state, build_name=build_name, cpu_smoke_build=cpu_smoke_build)
        else:
            state = None
            resolved_kind, resolved_name = _resolve_fresh_selector(
                engine, source_name=source_name,
            )
        _require_lane_build_inputs(
            state, build_name=build_name, platform_name=platform_name,
            cpu_smoke_build=cpu_smoke_build, skip_cpu_smoke=skip_cpu_smoke,
        )

        return _run_phases(
            engine=engine,
            state=state,
            target_ref=target_ref,
            selector_kind=resolved_kind,
            selector_name=resolved_name,
            repo_root=repo_root,
            vendor_root=vendor_root,
            dispositions_dir=dispositions_dir,
            report_dir=report_dir,
            skip_compile=skip_compile,
            cpu_smoke_build=cpu_smoke_build,
            skip_cpu_smoke=skip_cpu_smoke,
            accept=accept,
            platform_name=platform_name,
            build_name=build_name,
        )
    except PinBumpStop as exc:
        # Prefer the loaded state: re-reading a corrupt state file here could
        # mask the original error. Disk recovery below is best effort only.
        if state is not None:
            exc.run_id = state.run_id
            exc.target = {"from_ref": state.from_ref, "to_ref": state.to_ref}
            exc.transition_commit = state.transition_commit
            exc.tree = {"name": state.tree_name, "path": state.tree_path}
        else:
            saved = None
            try:
                if (report_dir / "state.json").is_file():
                    saved = PinBumpState.load(report_dir)
            except Exception:  # noqa: BLE001 -- context recovery must never mask exc
                saved = None
            if saved is not None:
                exc.run_id = saved.run_id
                exc.target = {"from_ref": saved.from_ref, "to_ref": saved.to_ref}
                exc.transition_commit = saved.transition_commit
                exc.tree = {"name": saved.tree_name, "path": saved.tree_path}
            else:
                exc.run_id = "unresolved"
                exc.target = {"from_ref": "?", "to_ref": target_ref}
                exc.tree = {"name": "local", "path": str(vendor_root)}
        raise


def _vendor_head(vendor_root: Path) -> str:
    done = _git_result(vendor_root, "rev-parse", "HEAD")
    return done.stdout.strip() if done.returncode == 0 else ""


def ensure_vendor_at_target(vendor_root: Path, *, to_ref: str, to_sha: str) -> bool:
    """A resumed run reads the vendored tree, so it must be on the target the run pulled. Whoever put it elsewhere (an
    operator restoring the old pin, another tool) would otherwise get drift failures that name the wrong thing, so the
    tree is put back: checked out to the recorded target when it has no tracked change to lose, refused when it has.
    Returns True when it had to move the tree."""
    head = _vendor_head(vendor_root)
    if not head:
        return False   # not a git checkout: there is no revision to put it on
    if head and (head == to_sha or (to_sha and head.startswith(to_sha)) or (to_sha and to_sha.startswith(head))):
        return False
    known = subprocess.run(["git", "-C", str(vendor_root), "cat-file", "-e", f"{to_sha}^{{commit}}"], capture_output=True)
    if not to_sha or known.returncode != 0:
        return False   # the tree never fetched this target (the pull phase has not run): nothing to put it on
    dirty = _git_result(vendor_root, "status", "--porcelain", "--untracked-files=no")
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise PinBumpStop(
            "resume",
            "VENDOR_TREE_NOT_AT_TARGET",
            f"the vendored tree {vendor_root} is at {head[:12]}, not at this run's target {to_ref} "
            f"({to_sha[:12]}), and it has tracked changes that a checkout would lose",
            evidence={"vendor_head": head, "target": to_ref, "target_sha": to_sha, "status": dirty.stdout.strip()[:400]},
            recommended_actions=[f"inspect `git -C {vendor_root} status`, then `git -C {vendor_root} checkout {to_ref}`, then --resume"],
        )
    moved = _git_result(vendor_root, "checkout", "--quiet", to_sha)
    if moved.returncode != 0:
        raise PinBumpStop(
            "resume",
            "VENDOR_TREE_NOT_AT_TARGET",
            f"the vendored tree {vendor_root} could not be put on this run's target {to_ref}: {moved.stderr.strip()[-300:]}",
            evidence={"vendor_head": head, "target": to_ref, "target_sha": to_sha},
            recommended_actions=[f"`git -C {vendor_root} fetch` the target, then --resume"],
        )
    print(f"pin-bump: the vendored tree was at {head[:12]}; checked out this run's target {to_ref} ({to_sha[:12]})",
          file=sys.stderr)
    return True


def fresh_state(
    *,
    from_ref: str,
    to_ref: str,
    to_sha: str,
    vendor_root: Path,
    selector_kind: str,
    selector_name: str,
    selector_patch_ids: Sequence[str],
    build_name: str | None,
    platform_name: str | None,
    cpu_smoke_build: Path | None,
) -> PinBumpState:
    """The state of a run that has just passed preflight. Every input the later phases need to be re-run without
    arguments is recorded here, the compile platform with the lane-build ones: a run that stops before its compile
    phase (a drift stop is the common one) must resume with a bare ``--resume``, not by repeating them."""
    return PinBumpState(
        schema_version=3,
        run_id=uuid.uuid4().hex,
        from_ref=from_ref,
        from_sha="",
        to_ref=to_ref,
        to_sha=to_sha,
        transition_commit="",
        tree_name="local",
        tree_path=str(vendor_root),
        completed_phases=["preflight"],
        next_phase="declare",
        selector_kind=selector_kind,
        selector_name=selector_name,
        selector_patch_ids=tuple(selector_patch_ids),
        lane_build_name=build_name or "",
        compile_platform=platform_name or "",
        cpu_smoke_build_override=str(cpu_smoke_build) if cpu_smoke_build is not None else "",
    )


def _load_state_or_stop(report_dir: Path) -> PinBumpState:
    """Wraps PinBumpState.load() so a corrupt/unreadable state.json becomes
    a structured PinBumpStop (gpt-dev-agent review, session
    ses_5307d9c58ec645cb, second pass) -- pin_bump's own contract is "raise
    PinBumpStop, never a bare exception", which PinBumpState.load() itself
    does not uphold (JSONDecodeError/KeyError/OSError on bad input)."""
    try:
        return PinBumpState.load(report_dir)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PinBumpStop(
            "resume",
            "RESUME_STATE_INVALID",
            f"state.json under {report_dir} could not be loaded: "
            f"{type(exc).__name__}: {exc}",
            evidence={"report_dir": str(report_dir), "error": str(exc)},
            recommended_actions=[
                "inspect state.json by hand",
                "if it is unrecoverable, start a fresh run instead",
            ],
        ) from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



@dataclass(frozen=True)
class SourceLineOwner:
    """The production patch edit that last produced one composed source line."""

    patch_id: str
    edit_id: str


def _selected_file_patches(
    engine: str,
    patch_ids: Sequence[str],
) -> list[tuple[str, patch_apply.FilePatch]]:
    """Load the exact selected patch sequence without discarding patch IDs."""

    selection = patch_patchset.resolve_exact(engine, tuple(patch_ids), allow_rejected=False)
    registries: dict[Path, patch_registry.PatchRegistry] = {}
    loaded: list[tuple[str, patch_apply.FilePatch]] = []
    for module in selection.modules:
        root = module.catalog_root or paths.patches_root(engine)
        registry = registries.get(root)
        if registry is None:
            registry = patch_registry.load_registry(engine, root)
            registries[root] = registry
        descriptor = registry.get(module.patch_id)
        for file_patch in patch_registry.load_implementation(engine, descriptor, root=root):
            loaded.append((module.patch_id, file_patch))
    return loaded


def _component_tree_ids(engine: str) -> list[str]:
    """The committed tree of every component the engine builds (its [components] directories in this repository).
    The compile gate and the lane build compile them into the same runtime home as the vendor tree, so a component
    fix must invalidate a passed compile exactly as a vendor change does. Preflight refuses a dirty controller
    checkout, so the committed tree is what gets built."""
    ids = []
    for name, directory in engines.load(paths.REPO_ROOT, engine).components:
        result = _git_result(paths.REPO_ROOT, "rev-parse", f"HEAD:{directory}")
        if result.returncode != 0:
            raise PinBumpStop(
                "compile",
                "SOURCE_COMPILE_FAILED",
                f"could not identify component {name!r} ({directory}) before compile: {result.stderr.strip()}",
                evidence={"component": name, "directory": directory},
                recommended_actions=["commit the component directory, then rerun bigcherry --engine <engine> pin-bump --resume"],
            )
        ids.append(f"{name}={result.stdout.strip()}")
    return ids


def _compile_tree_oid(engine: str, vendor_root: Path, patch_ids: Sequence[str]) -> str:
    """Exact effective source identity used to invalidate compile resume: the patched vendor tree and every built
    component's committed tree."""

    allowed_untracked = set(patch_rebase._overlay_texts(engine))
    for _, file_patch in _selected_file_patches(engine, patch_ids):
        if file_patch.create:
            allowed_untracked.add(file_patch.path)
    try:
        vendor = source_identity.git_tree_oid(
            vendor_root, allowed_untracked=allowed_untracked
        )
    except source_identity.SourceIdentityError as exc:
        raise PinBumpStop(
            "compile",
            "SOURCE_COMPILE_FAILED",
            f"could not identify the patched source tree before compile: {exc}",
            evidence={"source_identity_error": str(exc)},
            recommended_actions=[
                "inspect unexpected files in the vendor checkout and restore the applied production tree",
                "rerun bigcherry --engine <engine> pin-bump --resume",
            ],
        ) from exc
    components = _component_tree_ids(engine)
    if not components:
        return vendor
    return hashlib.sha1("\n".join([vendor, *components]).encode()).hexdigest()


def _git_head_text(root: Path, relative: str) -> str | None:
    result = _git_result(root, "show", f"HEAD:{relative}", encoding="utf-8", errors="replace")
    return result.stdout if result.returncode == 0 else None


def _line_owners_after_change(
    before: str,
    before_owners: list[SourceLineOwner | None],
    after: str,
    changed_owner: SourceLineOwner | None,
) -> list[SourceLineOwner | None]:
    """Carry provenance through one exact edit using the edit's real output."""

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    if len(before_lines) != len(before_owners):
        raise ValueError("source provenance line count drifted")
    owners: list[SourceLineOwner | None] = []
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            owners.extend(before_owners[i1:i2])
        elif tag in ("replace", "insert"):
            owners.extend([changed_owner] * (j2 - j1))
        elif tag != "delete":  # pragma: no cover - SequenceMatcher closed vocabulary
            raise AssertionError(tag)
    return owners


def _build_patch_line_ownership(
    engine: str, vendor_root: Path, patch_ids: Sequence[str]
) -> dict[str, list[SourceLineOwner | None]]:
    """Replay selected edits and map patched lines to production patch/edit IDs.

    The replay uses the same FilePatch/Edit objects and apply_patch semantics
    as the real apply phase. It never consults source comments for ownership.
    """

    overlay_texts = patch_rebase._overlay_texts(engine)
    texts: dict[str, str] = {}
    owners: dict[str, list[SourceLineOwner | None]] = {}

    for patch_id, file_patch in _selected_file_patches(engine, patch_ids):
        if file_patch.path not in texts:
            initial = overlay_texts.get(file_patch.path)
            if initial is None:
                initial = _git_head_text(vendor_root, file_patch.path)
            if initial is None:
                initial = ""
            texts[file_patch.path] = initial
            owners[file_patch.path] = [None] * len(initial.splitlines(keepends=True))

        for edit in file_patch.edits:
            before = texts[file_patch.path]
            one_text = {file_patch.path: before}
            one_patch = patch_apply.FilePatch(
                path=file_patch.path,
                edits=(edit,),
                description=file_patch.description,
                language=file_patch.language,
                create=file_patch.create,
            )
            result = patch_apply.apply_patch(
                one_patch, vendor_root, dry_run=True, texts=one_text
            )
            if not result.ok:
                raise PinBumpStop(
                    "compile",
                    "SOURCE_COMPILE_FAILED",
                    "could not reconstruct patch ownership for compiler diagnostics",
                    evidence={
                        "path": file_patch.path,
                        "patch_id": patch_id,
                        "edit_id": edit.id,
                        "detail": result.failed[0].detail if result.failed else "",
                    },
                    recommended_actions=[
                        "inspect the named production patch edit",
                        "rerun bigcherry --engine <engine> pin-bump --resume",
                    ],
                )
            after = one_text[file_patch.path]
            if after != before:
                owners[file_patch.path] = _line_owners_after_change(
                    before,
                    owners[file_patch.path],
                    after,
                    SourceLineOwner(patch_id, edit.id),
                )
                texts[file_patch.path] = after

    # Align the replay with the exact live patched bytes. Any live line not
    # explained by selected edits remains unowned and is reported as upstream.
    for relative, replayed in list(texts.items()):
        target = vendor_root / relative
        if not target.is_file():
            continue
        live = target.read_text(encoding="utf-8", errors="replace")
        if live != replayed:
            owners[relative] = _line_owners_after_change(
                replayed, owners[relative], live, None
            )

    # Compiler diagnostics can point at an upstream line in a CUDA source file
    # that no selected patch touched. Keep those real file/line locations too;
    # their line owners are intentionally all None -> owner "upstream".
    for source in source_compile_check._source_files(vendor_root):
        relative = source.relative_to(vendor_root).as_posix()
        if relative not in owners:
            text = source.read_text(encoding="utf-8", errors="replace")
            owners[relative] = [None] * len(text.splitlines(keepends=True))
    return owners


def _location_owner(
    relative: str,
    line: int,
    ownership: dict[str, list[SourceLineOwner | None]],
) -> dict[str, Any]:
    line_owners = ownership.get(relative, [])
    index = line - 1
    if 0 <= index < len(line_owners) and line_owners[index] is not None:
        owner = line_owners[index]
        assert owner is not None
        return {
            "owner": owner.patch_id,
            "edit_id": owner.edit_id,
            "related_patches": [],
        }

    preceding = next(
        (
            item
            for item in reversed(line_owners[: max(0, index)])
            if item is not None
        ),
        None,
    )
    if preceding is not None:
        related = [preceding.patch_id]
    else:
        related = list(
            dict.fromkeys(item.patch_id for item in line_owners if item is not None)
        )
    return {"owner": "upstream", "edit_id": None, "related_patches": related}


def _compile_failure_evidence(
    failures: Sequence[str],
    ownership: dict[str, list[SourceLineOwner | None]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    known_paths = sorted(ownership, key=len, reverse=True)
    for failure in failures:
        message = [line for line in failure.splitlines() if line.strip()][:6]
        normalized_failure = failure.replace("\\", "/")
        found: set[tuple[str, int]] = set()
        for relative in known_paths:
            for match in re.finditer(
                re.escape(relative) + r":(\d+)", normalized_failure
            ):
                location = (relative, int(match.group(1)))
                if location in found:
                    continue
                found.add(location)
                error = {
                    "path": relative,
                    "line": location[1],
                    "message": message,
                }
                error.update(_location_owner(relative, location[1], ownership))
                errors.append(error)
        if not found:
            errors.append(
                {
                    "path": "<compile-probe>",
                    "line": 0,
                    "owner": "upstream",
                    "edit_id": None,
                    "related_patches": [],
                    "message": message,
                }
            )
    return errors


def run_phase_compile(
    *,
    engine: str,
    vendor_root: Path,
    workdir: Path,
    recipe_report_path: Path,
    target_ref: str,
    platform_name: str | None,
) -> str:
    """Run the engine's compile checks (its adapter's ``compile_checks``) and return the checked tree OID.

    The first check that cannot run or finds failures STOPs the bump under the check's own failure code."""

    resume = f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume"
    try:
        checks = adapter(engine).compile_checks(
            ProjectContext.resolve(engine=engine), vendor_root, workdir, platform_name=platform_name)
    except (ValueError, OSError) as exc:
        raise PinBumpStop(
            "compile",
            "SOURCE_COMPILE_FAILED",
            f"the compile checks of {engine} could not be set up: {type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=[f"fix the engine declaration, then {resume}"],
        ) from exc
    for check in checks:
        absent = [tool for tool in check.tools if shutil.which(tool) is None]
        if absent:
            raise PinBumpStop(
                "compile",
                "SOURCE_COMPILE_UNAVAILABLE",
                f"required source compile tool is not available: {', '.join(absent)}",
                evidence={"check": check.id, "missing": absent, "compiler": absent[0]},
                recommended_actions=[
                    check.install_hint or f"install {', '.join(absent)}",
                    f"rerun {resume}",
                    "only if intentionally accepting the missing gate, rerun with --resume --skip-compile",
                ],
            )

    try:
        report = json.loads(recipe_report_path.read_text(encoding="utf-8"))
        patch_ids = tuple(report["selector"]["patch_ids"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PinBumpStop(
            "compile",
            "SOURCE_COMPILE_FAILED",
            f"source compile check could not run: {type(exc).__name__}: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=[f"rerun {resume}"],
        ) from exc

    for check in checks:
        try:
            failures = check.run()
        except (OSError, BumpCheckError) as exc:
            raise PinBumpStop(
                "compile",
                "SOURCE_COMPILE_FAILED",
                f"source compile check {check.id} could not run: {type(exc).__name__}: {exc}",
                evidence={"check": check.id, "error": str(exc)},
                recommended_actions=[
                    "fix the source or compile-check failure in the owning production patch",
                    f"rerun {resume}",
                ],
            ) from exc
        if not failures:
            continue
        if check.patch_attributed:
            evidence: dict[str, Any] = {
                "check": check.id,
                "errors": _compile_failure_evidence(failures, _build_patch_line_ownership(engine, vendor_root, patch_ids)),
            }
            actions = ["fix each error inside the named owning patch; for an upstream line, start with related_patches"]
        else:
            evidence = {"check": check.id, "failures": failures}
            actions = [f"fix the component or the patch that broke {check.id}, then rerun the bump"]
        raise PinBumpStop(
            "compile",
            check.failure_code,
            f"{len(failures)} failure(s) in compile check {check.id} on the patched tree",
            evidence=evidence,
            recommended_actions=[*actions, f"rerun {resume}"],
        )

    return _compile_tree_oid(engine, vendor_root, patch_ids)


@dataclass(frozen=True)
class LaneBuildOutcome:
    """What the lane-build phase learns about the build it made."""

    build_plan_id: str
    build_dir: Path
    binary_sha256: str


class LaneBuildError(Exception):
    """``kind`` is "request" (the lane cannot be planned) or "build" (the lane ran and failed)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def execute_lane_build(*, engine: str, lane: str, run_id: str) -> LaneBuildOutcome:
    """Build ``lane`` (``<source>:<build>:<platform>``) exactly as `bigcherry build --lane` does: the campaign planner
    plans it and ``run_campaign`` runs it. The launcher is the engine's serve binary, which is what the cpu smoke starts."""
    from ..build.builds import build_directory
    from ..campaign import planner
    from ..core import config as campaign_config
    from ..core import engines
    from ..core.artifacts import ArtifactStore

    parts = lane.split(":")
    if len(parts) != 3 or not all(parts):
        raise LaneBuildError("request", f"lane {lane!r} must be SOURCE:BUILD:PLATFORM")
    context = ProjectContext.resolve(engine=engine)
    try:
        cfg = campaign_config.load(context.config_path)
        request = planner.CampaignRequest(
            selectors=(campaign_config.CampaignLaneSelector(*parts),),
            binary_relative_path=engines.load(paths.REPO_ROOT, engine).serve.binary,
        )
        lanes = planner.plan(request, cfg)
    except (campaign_config.ConfigError, planner.CampaignPlannerError) as exc:
        raise LaneBuildError("request", str(exc)) from exc
    results = planner.run_campaign(
        lanes, cfg=cfg, context=context, store=ArtifactStore(context.work_root / "artifacts-store"), run_id=run_id)
    (result,) = results.values()
    if isinstance(result, Exception):
        raise LaneBuildError("build", f"{type(result).__name__}: {result}")
    return LaneBuildOutcome(
        build_plan_id=result.build_plan_id,
        build_dir=build_directory(context, result.source_slice_id, result.build_plan),
        binary_sha256=result.binary_ref.content_hash,
    )


def run_phase_lane_build(*, engine: str, lane: str, run_id: str, target_ref: str) -> LaneBuildOutcome:
    """MEN16: build the bumped engine through its canonical lane."""
    resume = f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume"
    try:
        return execute_lane_build(engine=engine, lane=lane, run_id=run_id)
    except LaneBuildError as exc:
        invalid = exc.kind == "request"
        raise PinBumpStop(
            "lane-build",
            "LANE_BUILD_REQUEST_INVALID" if invalid else "LANE_BUILD_FAILED",
            f"lane {lane} {'cannot be planned' if invalid else 'failed to build'}: {exc}",
            evidence={"lane": lane, "error": str(exc)},
            recommended_actions=[
                f"fix the cause, then {resume} (with a corrected --build for an invalid lane)",
                f"to smoke a build made by hand, {resume} --cpu-smoke-build <build dir>",
            ],
        ) from exc


ResumeInput = TypeVar("ResumeInput")


def _reconcile_resume_input(
    requested: ResumeInput | None, saved: ResumeInput | None, *,
    incompatible: bool, failure: PinBumpStop,
) -> ResumeInput | None:
    """Use the saved input unless an explicit resume option changes its identity."""
    if requested is not None and (incompatible or (saved is not None and requested != saved)):
        raise failure
    return requested or saved


def _resume_lane_build_inputs(
    state: PinBumpState, *, build_name: str | None, cpu_smoke_build: Path | None,
) -> tuple[str | None, Path | None]:
    """Reconcile the saved lane profile and smoke override; a passed build cannot be replaced."""
    saved_build = state.lane_build_name or None
    effective_build = _reconcile_resume_input(
        build_name, saved_build, incompatible=False,
        failure=PinBumpStop(
            "resume",
            "RESUME_BUILD_MISMATCH",
            f"--resume was given --build {build_name!r}, but this run's recorded lane build profile is {saved_build!r}",
            evidence={"resumed_build": build_name, "state_build": saved_build},
            recommended_actions=[
                "omit --build on --resume to use the run's original profile, or start a fresh run with the new one",
            ],
        ),
    )
    saved_override = Path(state.cpu_smoke_build_override) if state.cpu_smoke_build_override else None
    effective_override = _reconcile_resume_input(
        cpu_smoke_build, saved_override, incompatible=state.lane_build_status == "passed",
        failure=PinBumpStop(
            "resume",
            "RESUME_CPU_SMOKE_BUILD_MISMATCH",
            f"--resume was given --cpu-smoke-build {str(cpu_smoke_build)!r}, but this run's recorded build is "
            f"{state.cpu_smoke_build_override or state.lane_build_dir!r}",
            evidence={"resumed_cpu_smoke_build": str(cpu_smoke_build), "state_build": state.lane_build_dir},
            recommended_actions=[
                "omit --cpu-smoke-build on --resume to use the run's original build, or start a fresh run",
            ],
        ),
    )
    return effective_build, effective_override


def _require_lane_build_inputs(
    state: PinBumpState | None, *, build_name: str | None, platform_name: str | None,
    cpu_smoke_build: Path | None, skip_cpu_smoke: bool,
) -> None:
    """The lane build has no default profile: without --build (and --platform) the run is refused before any phase,
    unless an override or --skip-cpu-smoke means no build is needed or the saved run already has its build."""
    if skip_cpu_smoke or cpu_smoke_build is not None:
        return
    if state is not None and state.lane_build_status in {"passed", "skipped-by-override", "skipped"}:
        return
    missing = [flag for flag, value in (("--build", build_name), ("--platform", platform_name)) if not value]
    if missing:
        raise PinBumpStop(
            "preflight",
            "LANE_BUILD_INPUT_MISSING",
            f"pin-bump builds the bumped engine through its lane for the cpu smoke and needs {' and '.join(missing)}",
            evidence={"missing": missing},
            recommended_actions=[
                "pass --build <recipe [build.*] profile> and --platform <recipe [platform.*]>",
                "or pass --cpu-smoke-build <build dir> to smoke a build made by hand",
                "or, only if intentionally omitting the gate, pass --skip-cpu-smoke",
            ],
        )


def run_phase_cpu_smoke(*, engine: str, build_dir: Path | None, out_dir: Path, target_ref: str) -> None:
    """MEN15: run the engine's RAM-only tiny-model smoke against a build of the bumped tree."""
    from ..campaign import cpu_smoke

    resume = f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume"
    if build_dir is None:
        raise PinBumpStop(
            "cpu-smoke",
            "CPU_SMOKE_BUILD_MISSING",
            "the cpu-smoke phase needs a build of the bumped tree and none was given",
            recommended_actions=[
                f"{resume} --build <build profile> --platform <platform> (builds the bumped tree through its lane), or "
                f"{resume} --cpu-smoke-build <build dir>",
                f"only if intentionally omitting the gate, {resume} --skip-cpu-smoke",
            ],
        )
    try:
        # pin-bump smokes the bump's own build, which carries the engine's shutdown route.
        cpu_smoke.run_for_engine(engine, build_dir, None, out_dir, None)
    except cpu_smoke.CpuSmokeError as exc:
        raise PinBumpStop(
            "cpu-smoke",
            "CPU_SMOKE_FAILED",
            f"CPU smoke failed: {exc.failures[0]}",
            evidence={"failures": exc.failures, "result": str(out_dir / cpu_smoke.RESULT_NAME), "build": str(build_dir)},
            recommended_actions=[
                f"fix the cause (see {out_dir / 'server.log'}), then {resume} (the saved build is reused)",
                f"only if intentionally omitting the gate, {resume} --skip-cpu-smoke",
            ],
        ) from exc


def run_phase_accept(*, engine: str, out_dir: Path, target_ref: str, vendor_root: Path) -> None:
    """MEN17 (`--accept`): the engine's hardware acceptance against its stored baselines, after the CPU smoke.

    Needs the GPUs; with none visible the bump STOPs rather than recording an acceptance that did not happen. The
    result is written to the release record's notes.
    """
    from ..campaign import acceptance
    from ..core import engines as _engines
    from ..core import gpu as _gpu
    from ..core import paths as _paths

    resume = f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume"
    try:
        _gpu.free_vram_bytes((0,))
    except _gpu.GpuError as exc:
        raise PinBumpStop(
            "cpu-smoke",
            "ACCEPT_NO_GPU",
            f"--accept needs the GPUs and none could be queried: {exc}",
            recommended_actions=[
                f"run on the GPU host (python -m bigcherry remote run), or leave out --accept: {resume}",
            ],
        ) from exc
    try:
        layout = _engines.load(_paths.REPO_ROOT, engine)
        passed, results, evidence = acceptance.accept(_paths.REPO_ROOT, layout, None, out_dir)
    except (acceptance.AcceptanceError, _engines.EngineError, FileExistsError) as exc:
        raise PinBumpStop(
            "cpu-smoke", "ACCEPT_UNRUNNABLE", f"hardware acceptance could not run: {exc}",
            recommended_actions=[f"fix the acceptance declaration or output directory, then {resume} --accept"],
        ) from exc
    if not passed:
        failed = [f"{r.name}: {row.arm} {row.metric} {row.measured} (baseline {row.baseline})"
                  for r in results for row in r.rows if not row.passed]
        raise PinBumpStop(
            "cpu-smoke",
            "ACCEPT_FAILED",
            "hardware acceptance regressed against the stored baselines",
            evidence={"failed": failed or [f"{r.name}: {r.error}" for r in results if not r.passed], "evidence": str(evidence)},
            recommended_actions=[
                f"read {evidence}; fix the regression, then {resume} --accept",
                "if the baseline itself is outdated, re-measure and update its baselines file deliberately",
            ],
        )
    _record_release_note(
        vendor_root, f"pin-bump hardware acceptance: PASSED against stored baselines (evidence {evidence})", phase="cpu-smoke")


def _append_release_note(vendor_root: Path, note: str) -> None:
    record = releases.record_for_checkout(vendor_root)
    existing = record.notes or ""
    if note not in existing:
        record.notes = (existing.rstrip() + "\n" + note).lstrip()
        record.save()


def _record_release_note(vendor_root: Path, note: str, *, phase: str) -> None:
    """Append ``note`` to the bump's release record (once)."""
    try:
        _append_release_note(vendor_root, note)
    except (OSError, ValueError) as exc:
        raise PinBumpStop(
            phase,
            "RELEASE_NOTE_RECORD_FAILED",
            f"could not record {note!r} in the release record: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=["fix the release record write failure", "rerun with --resume"],
        ) from exc


def run_phase_evidence(
    *, engine: str, source_name: str, target_ref: str, to_sha: str,
    repo_root: Path, vendor_root: Path, report_dir: Path,
) -> dict[str, Any]:
    """MEN28: re-audit production validated sets at the NEW pin.

    Pending engines retain their failure count and proceed. Enforced engines
    retain the report and stop, so --resume repeats this phase after repairs.
    """
    from ..core import config as campaign_config
    from ..core import engines as engine_layouts

    phase = "evidence"
    try:
        release_config = release_notes.load_config(repo_root, engine)
        layout = engine_layouts.load(repo_root, engine)
        cfg = campaign_config.load(layout.recipes_path(repo_root))
        # Audit the engine's declared production source, not an optional
        # selector used to run this bump (which might omit shipped patches).
        production_source = layout.promotion_source
        if not production_source or production_source not in cfg.sources:
            raise ValueError("engine does not declare a valid production promotion source")
        audit = build_evidence.evaluate_source(
            engine=engine, cfg=cfg, source_name=production_source,
            patches_dir=layout.patches_root(repo_root),
            pinned_ref=target_ref, resolved_base_revision=to_sha,
            mode=release_config.evidence_gate,
        )
    except (OSError, KeyError, TypeError, ValueError, release_notes.ReleaseNotesError) as exc:
        raise PinBumpStop(
            phase, "EVIDENCE_GATE_UNRUNNABLE",
            f"cannot evaluate production patch evidence at {target_ref}: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=["repair evidence inputs, then rerun with --resume"],
        ) from exc

    result = audit.record()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evidence-gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        record = releases.record_for_checkout(vendor_root)
        record.evidence_gate = result
        record.save()
    except (OSError, TypeError, ValueError) as exc:
        raise PinBumpStop(
            phase, "EVIDENCE_RECORD_FAILED",
            f"cannot persist evidence gate result at {target_ref}: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=["repair the release record, then rerun with --resume"],
        ) from exc

    if audit.mode == "pending-requalification":
        print(audit.pending_line)
        print("first five patches: " +
              (", ".join(p for p, _ in audit.failed[:5]) or "(none)"))
        return result
    if audit.failed:
        raise PinBumpStop(
            phase, "EVIDENCE_GATE_STALE", build_evidence.failure_text(audit),
            evidence=result,
            recommended_actions=[
                *(
                    f"python -m bigcherry --engine {engine} patch-gates "
                    f"--intent build --source {production_source} {patch}"
                    for patch, _ in audit.failed
                ),
                "re-record evidence using skill bigcherry-patch-qualification "
                "(patch-qualification flow) at the new pin",
                f"python -m bigcherry --engine {engine} pin-bump {target_ref} --resume",
            ],
        )
    for line in build_evidence.report_lines(audit):
        print(line)
    return result


def _record_compile_skip(vendor_root: Path) -> None:
    note = "pin-bump source compile check: SKIPPED explicitly with --skip-compile"
    try:
        _append_release_note(vendor_root, note)
    except (OSError, ValueError) as exc:
        raise PinBumpStop(
            "compile",
            "SOURCE_COMPILE_SKIP_RECORD_FAILED",
            f"could not record the explicit compile skip in the release record: {exc}",
            evidence={"error": str(exc)},
            recommended_actions=[
                "fix the release record write failure",
                "rerun bigcherry --engine <engine> pin-bump --resume --skip-compile",
            ],
        ) from exc


def _rewind_compile_if_tree_changed(
    state: PinBumpState, *, engine: str, vendor_root: Path
) -> bool:
    """Invalidate compile and later phases when a checked source tree changed."""

    if "compile" not in state.completed_phases or not state.compile_tree_oid:
        return False
    live_oid = _compile_tree_oid(engine, vendor_root, state.selector_patch_ids)
    if live_oid == state.compile_tree_oid:
        return False
    state.completed_phases = [
        phase
        for phase in state.completed_phases
        if phase not in {"compile", "evidence", "lane-build", "cpu-smoke", "reaudit", "complete"}
    ]
    state.next_phase = "compile"
    state.compile_tree_oid = ""
    state.compile_status = ""
    state.evidence_gate = {}
    state.lane_build_status = ""
    state.lane_build_lane = ""
    state.lane_build_plan_id = ""
    state.lane_build_dir = ""
    state.lane_build_binary_sha256 = ""
    state.cpu_smoke_status = ""
    state.accept_status = ""
    return True



def _remove_bigcherry_untracked_source(
    engine: str, vendor_root: Path, patch_ids: Sequence[str]
) -> None:
    """Remove only untracked files that overlay/selected patches own.

    A forced checkout restores tracked upstream files but deliberately leaves
    untracked files alone. FilePatch(create=True) output must therefore be
    removed explicitly before a compile-failure resume recomposes the tree.
    """

    owned = set(patch_rebase._overlay_texts(engine))
    for _, file_patch in _selected_file_patches(engine, patch_ids):
        if file_patch.create:
            owned.add(file_patch.path)
    for relative in sorted(owned):
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(vendor_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode == 0:
            continue
        target = patch_apply.resolve_contained_target(vendor_root, relative)
        if target.is_file() or target.is_symlink():
            target.unlink()


def _rewind_compile_resume_if_inputs_changed(
    state: PinBumpState,
    *,
    engine: str,
    vendor_root: Path,
    recipe_report_path: Path,
) -> bool:
    """Recompose from pristine upstream when patch inputs changed after apply."""

    if state.next_phase != "compile":
        return False
    try:
        report = patch_rebase.load_report(recipe_report_path)
        patch_rebase._require_fresh(engine, report, vendor_root)
        return False
    except patch_rebase.StaleRebaseReportError:
        try:
            _remove_bigcherry_untracked_source(
                engine, vendor_root, state.selector_patch_ids
            )
        except (OSError, patch_apply.PatchError) as exc:
            raise PinBumpStop(
                "compile",
                "SOURCE_COMPILE_FAILED",
                f"compile inputs changed but the applied tree could not be reset safely: {exc}",
                evidence={"reset_error": str(exc)},
                recommended_actions=[
                    "restore the vendor checkout to the pinned upstream tree without deleting unrelated files",
                    "rerun bigcherry --engine <engine> pin-bump --resume",
                ],
            ) from exc

        # cmd_pull's forced checkout restores every tracked patched/overlay
        # file. From there re-run every controller-side proof that can have
        # changed with a patch fix before composing again.
        state.completed_phases = [
            phase for phase in state.completed_phases
            if phase in {"preflight", "declare"}
        ]
        state.next_phase = "pull"
        state.coverage_report_sha256 = ""
        state.compile_tree_oid = ""
        state.compile_status = ""
        state.evidence_gate = {}
        state.cpu_smoke_status = ""
        state.accept_status = ""
        return True


def _require_coverage_report(state: PinBumpState, report_path: Path) -> None:
    """Unconditional pre-apply gate (gpt-dev-agent review of c236acc, P1,
    session ses_5307d9c58ec645cb): the coverage phase's rebase-recipe.json
    must exist, and its live bytes must match what coverage actually wrote,
    every time apply is about to run -- fresh same-process coverage->apply
    included, not just a --resume. A missing or empty-bound digest is
    itself a failure, not something to silently skip past."""
    if not state.coverage_report_sha256:
        raise PinBumpStop(
            "apply",
            "COVERAGE_REPORT_UNBOUND",
            "no coverage_report_sha256 was ever recorded for this run -- "
            "refusing to apply an unproven report",
            recommended_actions=["re-run coverage from a fresh run"],
        )
    if not report_path.is_file():
        raise PinBumpStop(
            "apply",
            "COVERAGE_REPORT_MISSING",
            f"{report_path} does not exist -- refusing to apply",
            evidence={"report_path": str(report_path)},
            recommended_actions=[
                "restore the original rebase-recipe.json, or re-run "
                "coverage from a fresh run",
            ],
        )
    try:
        live_digest = _sha256_file(report_path)
    except FileNotFoundError as exc:
        # A TOCTOU window against is_file() above -- the report vanished
        # between the check and the read (a concurrent process on this
        # shared, multi-agent repo, or a race with cleanup).
        raise PinBumpStop(
            "apply",
            "COVERAGE_REPORT_MISSING",
            f"{report_path} disappeared before it could be read -- refusing to apply",
            evidence={"report_path": str(report_path), "error": str(exc)},
            recommended_actions=[
                "restore the original rebase-recipe.json, or re-run "
                "coverage from a fresh run",
            ],
        ) from exc
    except OSError as exc:
        raise PinBumpStop(
            "apply",
            "COVERAGE_REPORT_UNREADABLE",
            f"{report_path} could not be read: {exc}",
            evidence={"report_path": str(report_path), "error": str(exc)},
            recommended_actions=[
                "inspect the report file's permissions/state by hand",
                "re-run coverage from a fresh run",
            ],
        ) from exc
    if live_digest != state.coverage_report_sha256:
        raise PinBumpStop(
            "apply",
            "COVERAGE_REPORT_MODIFIED",
            "rebase-recipe.json's contents no longer match what the "
            "coverage phase wrote (modified or replaced on disk) -- "
            "refusing to apply an unproven report",
            evidence={
                "recorded_sha256": state.coverage_report_sha256,
                "live_sha256": live_digest,
            },
            recommended_actions=[
                "restore the original rebase-recipe.json, or re-run "
                "coverage from a fresh run",
            ],
        )


def _validate_resume(
    state: PinBumpState,
    *,
    target_ref: str,
    vendor_root: Path,
) -> None:
    """Resume identity check: state stores target/tree, but the phases
    never actually checked the RESUMING invocation's target_ref/vendor_root
    against them. A --resume with a different target or tree must fail
    closed, not silently continue against the wrong state."""
    if state.schema_version < 2:
        raise PinBumpStop(
            "resume",
            "LEGACY_STATE_SELECTOR_UNBOUND",
            "this in-flight run's state predates selector binding (schema "
            f"{state.schema_version} < 2) -- it cannot prove which selector "
            "(--source NAME) it was started with, so resuming it "
            "could silently apply a different composition than the "
            "original invocation intended",
            recommended_actions=[
                "start a fresh run instead of resuming this one",
                "if this state must be rescued, inspect its coverage report "
                "(if the coverage phase already ran) to recover its real "
                "selector by hand before resuming",
            ],
        )
    if target_ref != state.to_ref:
        raise PinBumpStop(
            "resume",
            "RESUME_TARGET_MISMATCH",
            f"--resume target {target_ref!r} does not match this run's "
            f"recorded target {state.to_ref!r}",
            evidence={"resumed_target": target_ref, "state_target": state.to_ref},
            recommended_actions=[
                f"pass --resume with target {state.to_ref!r}, or start a fresh run",
            ],
        )
    resolved_tree = str(vendor_root)
    if resolved_tree != state.tree_path:
        raise PinBumpStop(
            "resume",
            "RESUME_TREE_MISMATCH",
            f"--resume is running against {resolved_tree!r}, but this run's "
            f"recorded tree is {state.tree_path!r}",
            evidence={"resumed_tree": resolved_tree, "state_tree": state.tree_path},
            recommended_actions=[
                "resume from the same vendor checkout the run was started against",
            ],
        )


DEFAULT_RELEASE_SOURCE = "bigcherry"


def _resolve_fresh_selector(
    engine: str,
    *,
    source_name: str | None,
) -> tuple[str, str]:
    """Fresh-run selector default: no ``--source`` supplied -> "bigcherry"
    -- the real v2 release-source composition -- when the recipes of the
    engine declare a source of that name; otherwise the run must name one."""
    if source_name is not None:
        return "source", source_name
    from ..core import config as campaign_config

    declared = sorted(campaign_config.load(recipes_path(engine)).sources)
    if DEFAULT_RELEASE_SOURCE not in declared:
        raise PinBumpStop(
            "preflight",
            "SOURCE_REQUIRED",
            f"the {engine} recipes declare no {DEFAULT_RELEASE_SOURCE!r} source to default to; "
            "pass --source NAME",
            evidence={"declared_sources": declared},
            recommended_actions=[f"rerun with --source (one of: {', '.join(declared)})"],
        )
    return "source", DEFAULT_RELEASE_SOURCE


def _selector_patch_ids(engine: str, *, selector_kind: str, selector_name: str) -> tuple[str, ...]:
    return patch_rebase._selection_patch_ids(
        engine, source_name=selector_name, all_patches=False
    )


def _resume_selector(
    state: PinBumpState,
    *,
    source_name: str | None = None,
) -> tuple[str, str]:
    """Recover the persisted selector, rejecting an explicitly changed source."""
    selected = _reconcile_resume_input(
        source_name, state.selector_name, incompatible=state.selector_kind != "source",
        failure=PinBumpStop(
            "resume",
            "RESUME_SELECTOR_MISMATCH",
            f"--resume was given source:{source_name!r}, but "
            f"this run's recorded selector is "
            f"{state.selector_kind}:{state.selector_name!r}",
            evidence={
                "resumed_selector": f"source:{source_name}",
                "state_selector": f"{state.selector_kind}:{state.selector_name}",
            },
            recommended_actions=[
                "omit --source on --resume to use the run's "
                "original selector, or start a fresh run with the new one",
            ],
        ),
    )
    return state.selector_kind, selected


def _resume_platform(state: PinBumpState, *, platform_name: str | None) -> str | None:
    """A resumed compile must use the persisted platform unless none was saved."""
    saved = state.compile_platform or None
    return _reconcile_resume_input(
        platform_name, saved, incompatible=False,
        failure=PinBumpStop(
            "resume",
            "RESUME_PLATFORM_MISMATCH",
            f"--resume was given --platform {platform_name!r}, but this run's recorded compile platform is {saved!r}",
            evidence={"resumed_platform": platform_name, "state_platform": saved},
            recommended_actions=[
                "omit --platform on --resume to use the run's original platform, or start a fresh run with the new one",
            ],
        ),
    )


def _require_selector_membership_unchanged(
    state: PinBumpState,
    *,
    engine: str,
    selector_kind: str,
    selector_name: str,
) -> None:
    """Before coverage runs (fresh or resumed), re-derive the selector's
    real patch-id membership and require it still matches what the state
    recorded -- catches a concurrent config/recipes.toml or patch-registry
    edit on this shared, multi-agent repo, whether or not this is a
    --resume. Membership drift requires a deliberate restart, not a
    silent scope change to a production release run."""
    current_ids = tuple(
        sorted(
            _selector_patch_ids(
                engine, selector_kind=selector_kind, selector_name=selector_name
            )
        )
    )
    recorded_ids = tuple(sorted(state.selector_patch_ids))
    if current_ids != recorded_ids:
        raise PinBumpStop(
            "coverage",
            "RESUME_SELECTION_CHANGED",
            f"selector {state.selector_kind}:{state.selector_name!r}'s real "
            "patch membership changed since this run's state recorded it",
            evidence={
                "only_in_state": sorted(set(recorded_ids) - set(current_ids)),
                "only_live": sorted(set(current_ids) - set(recorded_ids)),
            },
            recommended_actions=[
                "start a fresh run to pick up the current composition deliberately",
            ],
        )


def _run_phases(
    *,
    engine: str,
    state: PinBumpState | None,
    target_ref: str,
    selector_kind: str,
    selector_name: str,
    repo_root: Path,
    vendor_root: Path,
    dispositions_dir: Path,
    report_dir: Path,
    skip_compile: bool = False,
    cpu_smoke_build: Path | None,
    skip_cpu_smoke: bool,
    accept: bool,
    platform_name: str | None,
    build_name: str | None,
) -> PinBumpResult:
    from ..patch import catalog as patch_catalog
    from ..source import audit as source_audit

    context = ProjectContext.resolve(engine=engine)
    with acquire_maintenance_lock(repo_root, engine):
        if state is None:
            from_ref, to_sha = run_phase_preflight(
                repo_root=repo_root, engine=engine, vendor_root=vendor_root, target_ref=target_ref
            )
            selector_patch_ids = tuple(
                sorted(
                    _selector_patch_ids(
                        engine, selector_kind=selector_kind, selector_name=selector_name
                    )
                )
            )
            state = fresh_state(
                from_ref=from_ref, to_ref=target_ref, to_sha=to_sha, vendor_root=vendor_root,
                selector_kind=selector_kind, selector_name=selector_name, selector_patch_ids=selector_patch_ids,
                build_name=build_name, platform_name=platform_name, cpu_smoke_build=cpu_smoke_build,
            )
            state.save(report_dir)

        if _rewind_compile_if_tree_changed(state, engine=engine, vendor_root=vendor_root):
            state.save(report_dir)
        if _rewind_compile_resume_if_inputs_changed(
            state,
            engine=engine,
            vendor_root=vendor_root,
            recipe_report_path=report_dir / "rebase-recipe.json",
        ):
            state.save(report_dir)

        if state.next_phase == "declare":
            declaring = run_phase_declare(
                repo_root=repo_root, engine=engine, vendor_root=vendor_root, target_ref=target_ref
            )
            state.transition_commit = declaring
            state.advance(report_dir, "declare", "pull")

        if state.next_phase == "pull":
            from argparse import Namespace

            from ..cli import source as cli_source

            # The orchestrator already owns the exact target ref -- pull
            # must not be made composition-dependent (source=None, not
            # selector_name), matching its own real ref-resolution priority
            # (--ref, when given, is used as-is; --source is only ever a
            # fallback for resolving a ref from a name).
            rc = cli_source.cmd_pull(
                Namespace(
                    engine=engine,
                    llama_root=None,
                    ref=target_ref,
                    source=None,
                    full=False,
                )
            )
            if rc != 0:
                raise PinBumpStop(
                    "pull",
                    "PULL_FAILED",
                    "bigcherry --engine <engine> pull did not exit cleanly",
                    evidence={"exit_code": rc},
                    recommended_actions=[
                        "inspect the vendor checkout directly",
                        "rerun with --resume",
                    ],
                )
            pulled = releases.record_for_checkout(vendor_root)
            if pulled.slug() != target_ref and not (
                pulled.slug() == pulled.revision[:12] and pulled.revision.startswith(target_ref)
            ):
                raise PinBumpStop(
                    "pull",
                    "RELEASE_RECORD_KEY_MISMATCH",
                    f"release record key {pulled.slug()!r} does not match the pinned ref {target_ref!r}; "
                    "pin-release and release-notes would not find the record",
                    evidence={"record_key": pulled.slug(), "target_ref": target_ref},
                    recommended_actions=[
                        f"make sure the vendor clone has the tag locally (git fetch origin refs/tags/{target_ref}:refs/tags/{target_ref})",
                        "rerun with --resume",
                    ],
                )
            _sync_campaign_mirror_best_effort(
                engine=engine, target_ref=target_ref, revision=state.to_sha
            )
            state.advance(report_dir, "pull", "drift")

        if state.next_phase == "drift":
            # Right after the pull: the tree is the pristine bumped upstream, so what these checks report is the engine's
            # own change and cannot be a patch, and it is found before the audit, patch rebase and builds spend time on a
            # tree BigCherry's own derived source (libr3's native rewrites) already no longer fits.
            run_phase_drift(engine=engine, vendor_root=vendor_root, target_ref=target_ref)
            state.advance(report_dir, "drift", "audit")

        if state.next_phase == "audit":
            report = source_audit.audit(vendor_root, engine=engine)
            if not source_audit.passed(report, strict=False):
                safe, drifted = check_overlay_self_heal(report)
                if not safe:
                    raise PinBumpStop(
                        "audit",
                        "AUDIT_FAILED",
                        "source audit failed on more than overlay staleness",
                        evidence={
                            "failed_checks": [
                                c["id"] for c in report["checks"] if not c["ok"]
                            ]
                        },
                        recommended_actions=[
                            f"run `bigcherry --engine {engine} audit --verbose` and reconcile",
                            "rerun with --resume",
                        ],
                    )
                patch_overlay.copy_overlay(vendor_root, context.overlay_root, dry_run=False)
                report = source_audit.audit(vendor_root, engine=engine)
                if not source_audit.passed(report, strict=False):
                    raise PinBumpStop(
                        "audit",
                        "OVERLAY_STALENESS_UNPROVEN",
                        "overlay drift persisted after the narrow newline-only self-heal",
                        evidence={"drifted": drifted},
                        recommended_actions=[
                            "inspect the drifted file(s) by hand",
                            "rerun with --resume",
                        ],
                    )
            # Apply checks the persisted record.audit, not this in-memory
            # report, so save the audit proof before running coverage/apply.
            record = releases.record_for_checkout(vendor_root)
            record.audit = releases.summarise_audit(report, strict=True)
            if record.stage == "pulled":
                record.advance_to("audited")
            record.save()
            state.advance(report_dir, "audit", "patch-lint")

        if state.next_phase == "patch-lint":
            problems = patch_catalog.cross_check(engine, allow_legacy_grandfather=True)
            if problems:
                raise PinBumpStop(
                    "patch-lint",
                    "PATCH_LINT_FAILED",
                    f"{len(problems)} patch-lint problem(s)",
                    evidence={"problems": problems},
                    recommended_actions=[
                        "fix the catalog/package mismatch",
                        "rerun with --resume",
                    ],
                )
            state.advance(report_dir, "patch-lint", "coverage")

        if state.next_phase == "coverage":
            _require_selector_membership_unchanged(
                state,
                engine=engine,
                selector_kind=selector_kind,
                selector_name=selector_name,
            )
            all_report = patch_rebase.run_rebase_check(engine, vendor_root, all_patches=True)
            recipe_report = patch_rebase.run_rebase_check(
                engine,
                vendor_root,
                source_name=selector_name,
            )
            stop_on_recipe_failures(phase="coverage", recipe_report=recipe_report)
            catalog_states = {
                patch_id: entry.state
                for patch_id, entry in patch_catalog.load_catalog(engine).items()
            }

            def focal_probe(patch_id: str) -> dict | None:
                # Only a source selection can carry a focal overlay; a patch that
                # cannot be layered over it (declared conflict, missing requires)
                # keeps its all-patches verdict.
                if selector_kind != "source":
                    return None
                try:
                    focal_report = patch_rebase.run_rebase_check(
                        engine, vendor_root, source_name=selector_name, focal_overlay_patch_id=patch_id
                    )
                except (RuntimeError, ValueError, KeyError):
                    return None
                for focal_entry in focal_report.get("patches", ()):
                    if focal_entry["patch_id"] == patch_id:
                        return focal_entry
                return None

            coverage = classify_patch_coverage(
                all_report=all_report,
                recipe_report=recipe_report,
                catalog_states=catalog_states,
                dispositions_dir=dispositions_dir,
                target_revision=state.to_sha,
                focal_probe=focal_probe,
            )
            coverage["test_failures"] = run_patch_tests(repo_root, engine)
            (report_dir / "patch-failures.json").write_text(
                json.dumps(
                    {"patches": coverage["outstanding"], "tests": coverage["test_failures"]},
                    indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            recipe_report_path = report_dir / "rebase-recipe.json"
            patch_rebase.write_report(recipe_report_path, recipe_report)
            (report_dir / "coverage.json").write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            state.coverage_report_sha256 = _sha256_file(recipe_report_path)
            state.advance(report_dir, "coverage", "apply")
        else:
            recipe_report_path = report_dir / "rebase-recipe.json"
            coverage = (
                json.loads((report_dir / "coverage.json").read_text(encoding="utf-8"))
                if (report_dir / "coverage.json").is_file()
                else {}
            )

        # Check report integrity on both a fresh coverage->apply and resume.
        if state.next_phase == "apply":
            _require_coverage_report(state, recipe_report_path)
            try:
                result = patch_rebase.apply_known_good(
                    engine, vendor_root, recipe_report_path, force=False, dry_run=False
                )
            except patch_rebase.StaleRebaseReportError as exc:
                # The controller checkout or the registry moved after coverage ran
                # (e.g. a commit landed mid-bump): coverage must be recomputed.
                state.completed_phases.remove("coverage")
                state.next_phase = "coverage"
                state.coverage_report_sha256 = ""
                state.save(report_dir)
                raise PinBumpStop(
                    "apply",
                    "COVERAGE_REPORT_STALE",
                    str(exc),
                    recommended_actions=[
                        "do not commit in this checkout while a bump is running",
                        "rerun `bigcherry --engine <engine> pin-bump --resume` (coverage is recomputed)",
                    ],
                ) from exc
            if not result.ok:
                raise PinBumpStop(
                    "apply",
                    "APPLY_FAILED",
                    "known-good apply did not succeed",
                    evidence={"known_good": list(result.known_good_patch_ids)},
                    recommended_actions=[
                        "inspect the apply failure directly",
                        "rerun with --resume",
                    ],
                )
            state.advance(report_dir, "apply", "compile")

        if state.next_phase == "compile":
            if skip_compile:
                compile_tree_oid = _compile_tree_oid(
                    engine, vendor_root, state.selector_patch_ids
                )
                _record_compile_skip(vendor_root)
                state.compile_status = "skipped"
            else:
                compile_tree_oid = run_phase_compile(
                    engine=engine,
                    vendor_root=vendor_root,
                    workdir=repo_root / "work" / "source-compile-check",
                    recipe_report_path=recipe_report_path,
                    target_ref=target_ref,
                    platform_name=platform_name,
                )
                state.compile_status = "passed"
            state.compile_platform = platform_name or ""
            state.compile_tree_oid = compile_tree_oid
            state.advance(report_dir, "compile", "evidence")

        if state.next_phase == "evidence":
            state.evidence_gate = run_phase_evidence(
                engine=engine, source_name=selector_name, target_ref=target_ref,
                to_sha=state.to_sha, repo_root=repo_root,
                vendor_root=vendor_root, report_dir=report_dir,
            )
            state.advance(report_dir, "evidence", "lane-build")

        if state.next_phase == "lane-build":
            # MEN16: build the bumped engine through its canonical lane so the cpu-smoke phase (and the release record)
            # speak of one build the bump itself made. An explicit --cpu-smoke-build replaces it and is recorded as such.
            if skip_cpu_smoke:
                state.lane_build_status = "skipped"
            elif cpu_smoke_build is not None:
                state.cpu_smoke_build_override = str(cpu_smoke_build)
                state.lane_build_status = "skipped-by-override"
                state.lane_build_dir = str(cpu_smoke_build)
                _record_release_note(
                    vendor_root, f"pin-bump cpu-smoke build: override {cpu_smoke_build} (no lane build)", phase="lane-build")
            else:
                state.lane_build_name = build_name or ""
                lane = f"{selector_name}:{build_name}:{platform_name}"
                # The lane build refuses a dirty controller tree and the records written so far are this run's own:
                # commit exactly those (idempotent) before it, not only at the end.
                _commit_release_records(repo_root=repo_root, target_ref=target_ref)
                outcome = run_phase_lane_build(
                    engine=engine, lane=lane, run_id=f"pin-bump-{state.run_id[:12]}", target_ref=target_ref)
                state.lane_build_status = "passed"
                state.lane_build_lane = lane
                state.lane_build_plan_id = outcome.build_plan_id
                state.lane_build_dir = str(outcome.build_dir)
                state.lane_build_binary_sha256 = outcome.binary_sha256
                _record_release_note(
                    vendor_root,
                    f"pin-bump lane build: {lane} build_plan_id={outcome.build_plan_id} "
                    f"binary_sha256={outcome.binary_sha256}",  # the plan id names the build; the release record never carries a local path
                    phase="lane-build",
                )
            state.advance(report_dir, "lane-build", "cpu-smoke")

        if state.next_phase == "cpu-smoke":
            # MEN15: the first smoke of the bumped engine, before any GPU smoke. It needs no GPU, so a busy host
            # cannot hold it up; only an explicit --skip-cpu-smoke omits it, and the skip is saved with the run.
            if skip_cpu_smoke:
                state.cpu_smoke_status = "skipped"
            else:
                run_phase_cpu_smoke(engine=engine, build_dir=Path(state.lane_build_dir) if state.lane_build_dir else None, out_dir=report_dir / "cpu-smoke", target_ref=target_ref)
                state.cpu_smoke_status = "passed"
            if accept and state.accept_status != "passed":
                run_phase_accept(engine=engine, out_dir=report_dir / "accept", target_ref=target_ref, vendor_root=vendor_root)
                state.accept_status = "passed"
            state.advance(report_dir, "cpu-smoke", "reaudit")

        if state.next_phase == "reaudit":
            report = source_audit.audit(vendor_root, engine=engine)
            if not source_audit.passed(report, strict=True):
                raise PinBumpStop(
                    "reaudit",
                    "POST_APPLY_AUDIT_FAILED",
                    "source audit failed after a full known-good apply",
                    evidence={
                        "failed_checks": [
                            c["id"] for c in report["checks"] if not c["ok"]
                        ]
                    },
                    recommended_actions=[
                        f"run `bigcherry --engine {engine} audit --verbose` and reconcile",
                        "rerun with --resume",
                    ],
                )
            state.advance(report_dir, "reaudit", "complete")

        if state.next_phase == "complete" and "complete" not in state.completed_phases:
            _write_release_doc_best_effort(
                engine=engine,
                repo_root=repo_root,
                vendor_root=vendor_root,
                selector_kind=selector_kind,
                selector_name=selector_name,
                target_ref=target_ref,
                report_dir=report_dir,
            )
            _commit_release_records(repo_root=repo_root, target_ref=target_ref)
            state.completed_phases.append("complete")
            state.save(report_dir)

    return PinBumpResult(ok=True, state=state, coverage=coverage)


def _commit_record_git(repo_root: Path, owned: list[str], label: str, *args: str) -> None:
    result = _git_result(repo_root, *args)
    if result.returncode != 0:
        raise PinBumpStop(
            "complete",
            "RELEASE_RECORD_COMMIT_FAILED",
            f"{label} failed for release records: {result.stderr.strip()}",
            evidence={"paths": owned, "stderr": result.stderr.strip()},
            recommended_actions=[
                "inspect the repo state directly",
                "rerun with --resume",
            ],
        )


def _commit_release_records(*, repo_root: Path, target_ref: str) -> None:
    """Commit only this run's release records, before the lane build and at completion.

    Exact pathspecs and `commit --only` exclude unrelated staged work on a
    shared checkout. The transition marker is NOT owned here; it requires
    a separately proven completion. Repeated resume commits are no-ops.
    """
    owned = [f"releases/{target_ref}.json", "releases/index.json"]
    if (repo_root / "releases" / f"{target_ref}-patches.md").is_file():
        owned.append(f"releases/{target_ref}-patches.md")

    existing = [p for p in owned if (repo_root / p).exists()]
    if not existing:
        return  # nothing this orchestrator owns exists yet -- idempotent no-op

    status = _git_result(repo_root, "status", "--porcelain", "--", *existing)
    if status.returncode == 0 and not status.stdout.strip():
        return  # already committed -- idempotent, matches a --resume re-run

    _commit_record_git(repo_root, existing, "git add", "add", "--", *existing)
    _commit_record_git(
        repo_root, existing, "git commit --only", "commit", "--only",
        "-m", f"pin-bump: record release {target_ref}", "--", *existing,
    )


def _write_release_doc_best_effort(
    *,
    engine: str,
    repo_root: Path,
    vendor_root: Path,
    selector_kind: str,
    selector_name: str,
    target_ref: str,
    report_dir: Path | None = None,
) -> None:
    """Merge the applied recipe's per-patch SUMMARY.md into a release doc.

    Best-effort like _sync_campaign_mirror_best_effort: a release doc is a
    documentation convenience, not a bump-correctness requirement, so a
    failure here must never turn a successful bump into a PinBumpStop. The
    broad catch is deliberate (any renderer bug must not become a bump
    correctness dependency); it prints WHAT failed to stderr rather than
    silently swallowing it (gpt-dev-agent review, 2026-08-31 -- the
    original bare `except Exception: pass` here and in
    _sync_campaign_mirror_best_effort discarded the only evidence of a
    real failure).

    Reads patch_ids from the SAME rebase-recipe.json report that
    authorized `apply`, rather than
    re-resolving the selector fresh at completion time -- a real TOCTOU
    otherwise, since config/recipes.toml or the patch registry could have
    changed between the coverage/apply phases and this one on this shared,
    multi-agent repo. When ``report_dir`` is supplied, this is now the
    ONLY source: a missing/invalid report skips the doc (still
    best-effort, never a PinBumpStop) rather than silently falling back to
    a fresh, TOCTOU-prone resolution. Fresh resolution only remains for a
    direct call site with no ``report_dir`` at all (e.g. a legacy/manual
    invocation that never ran coverage) -- existing callers/tests without
    report_dir keep working exactly as before.
    """
    try:
        from ..patch import docs as patch_docs
        from ..patch import rebase as patch_rebase
        from . import records as release_records

        if report_dir is not None:
            report_path = report_dir / "rebase-recipe.json"
            if not report_path.is_file():
                print(
                    f"pin-bump: release doc for {target_ref!r} skipped: "
                    f"{report_path} does not exist",
                    file=sys.stderr,
                )
                return
            patch_ids = tuple(
                json.loads(report_path.read_text(encoding="utf-8"))["selector"][
                    "patch_ids"
                ]  # PA34 schema-2 selector identity
            )
        else:
            patch_ids = _selector_patch_ids(
                engine,
                selector_kind=selector_kind,
                selector_name=selector_name,
            )
        pin_info = {
            f"{engine} upstream revision": patch_rebase._git(vendor_root, "rev-parse", "HEAD"),
            "bigcherry revision": patch_rebase._git(repo_root, "rev-parse", "HEAD"),
            "source": selector_name,
            "target": target_ref,
        }
        doc = patch_docs.render_patch_selection_doc(
            engine,
            patch_ids=patch_ids,
            pin_info=pin_info,
            selection_label=f"--source {selector_name}",
        )
        release_records.RELEASES_DIR.mkdir(parents=True, exist_ok=True)
        (release_records.RELEASES_DIR / f"{target_ref}-patches.md").write_text(
            doc, encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort convenience, never fatal
        print(
            f"pin-bump: release doc for {target_ref!r} "
            f"({selector_kind}:{selector_name!r}) was not written: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
