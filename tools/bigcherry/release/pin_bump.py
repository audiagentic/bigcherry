"""HI153: ``bigcherry pin-bump`` -- single-tree pin-bump orchestrator.

Automates the common case of docs/reference/build/PIN_BUMP.md's manual
procedure (repin -> commit -> pull -> audit -> patch-rebase-check ->
apply -> pin-status) while stopping cleanly and precisely -- via a
structured JSON failure envelope -- on anything not provably safe to
auto-proceed past. Converged design: see HI150 (round 1 + round 2 notes)
and its implementation sub-items HI151 (tree-activity liveness) and
HI152 (coverage/disposition schema), both landed before this.

Deliberately single-tree only (Phase 1). Multi-tree SSH sequencing is
explicitly deferred to a later Phase 2, per the HI150 design debate: this
bump-process test itself surfaced multiple never-before-seen edge cases
in the single-tree case alone, and gpt's cadence argument (once corrected
against the real release-record timestamps) still supports not building
distributed orchestration before the single-tree semantics have survived
several real bumps. ``bigcherry pin-status --complete --all-remotes``
remains the real cross-tree completion authority; this module does not
touch it beyond reporting its result.
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
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..build import source_compile_check
from ..core import paths
from ..core import tree_activity
from ..core.context import ProjectContext
from ..engine_adapters import BumpCheckError, adapter
from ..patch import apply as patch_apply
from ..patch import disposition as patch_disposition
from ..patch import overlay as patch_overlay
from ..patch import patchset as patch_patchset
from ..patch import rebase as patch_rebase
from ..patch import registry as patch_registry
from ..source import identity as source_identity
from .. import pin_transition
from .. import recipes as recipes_module
from . import records as releases
from .pin import recipes_path, resolve_pin_sha

PHASES = (
    "preflight",
    "declare",
    "pull",
    "audit",
    "patch-lint",
    "coverage",
    "apply",
    "compile",
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
            d["cpu_smoke"] = {"status": self.cpu_smoke_status}
            d["accept"] = {"status": self.accept_status}
        return d

    def save(self, state_dir: Path) -> Path:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, state_dir: Path) -> PinBumpState:
        data = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        # PA34 Q3 migration: schema 3 renames "selector" -> "selection_freeze".
        # A loaded schema-2 state migrates to schema 3 IN MEMORY (it persists
        # under the new key at the next successful save; never both keys).
        # Schema 1 states carry no selector binding at all and keep their
        # version so the resume guard still rejects them.
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
            # PA34 Q3 (dev-gpt-agent req_1d02cb052310446c Q3): a schema-2
            # state's "selector" key is the ONLY thing that makes it
            # resumable (see _validate_resume's schema<2 rejection above).
            # Promoting raw_version to 3 before checking the legacy shape
            # would let a missing/malformed "selector" key slip past that
            # guard as an unbound-but-schema-3 state instead of being
            # rejected as legacy/unbound. Validate first, fail closed.
            freeze = data.get("selector")
            if (
                not isinstance(freeze, dict)
                # Schema-2 pin-bump state only ever represented SOURCE
                # selectors (dev-gpt-agent round-3 req_ab94edd31aa04419 Q3):
                # accepting any other non-empty string here would let a
                # malformed/foreign kind (e.g. "experiment") be promoted
                # and later resumed, since _selector_patch_ids() ignores
                # selector_kind entirely and always resolves selector_name
                # as a source -- a malformed kind + valid source name would
                # silently resume.
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


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
    )
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


def run_phase_declare(*, repo_root: Path, engine: str, vendor_root: Path, target_ref: str) -> str:
    """Calls the existing, already-real repin() + pin_transition.write(),
    then commits the recipes file + the marker together. Returns the
    declaring commit SHA."""
    recipes_file = recipes_path(engine)
    old = recipes_module.repin(target_ref, recipes_file)
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


def run_patch_tests(repo_root: Path) -> list[str]:
    """Run the offline patch mechanics suite and return the failing test ids.
    Called before apply, while the vendor tree is the pristine new revision:
    the fixtures copy vendor files, so an already-patched tree would turn every
    fail-closed test red for reasons unrelated to upstream."""
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tools/tests/patch"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "tools")},
        capture_output=True,
        text=True,
    )
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
) -> PinBumpResult:
    """The Phase 1 single-tree orchestrator. Raises PinBumpStop (never a
    bare exception) on anything not provably safe to auto-proceed past;
    the caller is expected to catch it, write ``failure_envelope(...)``
    under ``report_dir``, and exit non-zero.

    ``source_name`` unsupplied on a fresh run defaults to ``bigcherry`` (the
    real v2 release-source composition -- serving-core + upstream-fixes +
    validated-enhancements; see ``_resolve_fresh_selector``). A ``--resume``
    with no selector reuses
    whatever the original invocation was started with (``_resume_selector``),
    rather than re-defaulting.

    ``platform_name`` is the recipe ``[platform.*]`` the compile gate configures with (``--platform``), for the engines whose
    gate builds. It is saved in the resume state; a ``--resume`` with none reuses the saved one and a different one is
    refused.
    """
    repo_root = paths.REPO_ROOT
    vendor_root = root if root is not None else ProjectContext.resolve(engine=engine).vendor_root
    dispositions_dir = dispositions_dir or paths.DISPOSITIONS
    report_dir = report_dir or default_report_dir(engine, target_ref)

    # Bound before the try, unconditionally: RESUME_STATE_MISSING and
    # _load_state_or_stop() can both raise before any assignment below
    # would otherwise run, and the except block needs `state` defined
    # either way (None means "fall back to a best-effort disk re-read").
    state: PinBumpState | None = None
    try:
        # gpt-dev-agent review of c236acc (P1, session ses_5307d9c58ec645cb):
        # --resume must never silently reinterpret a missing/wrong state as
        # "start fresh" -- that would let preflight/declare re-execute as a
        # NEW operation under a resume invocation. State load, resume
        # validation, and selector reconciliation all now happen INSIDE this
        # try (P2 of the same review): so a PinBumpStop raised by any of
        # them still gets real run_id/target/tree context attached below,
        # instead of falling through to the "unresolved" placeholder.
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
            resolved_kind, resolved_name = _resume_selector(
                state,
                source_name=source_name,
            )
            platform_name = _resume_platform(state, platform_name=platform_name)
        else:
            state = None
            resolved_kind, resolved_name = _resolve_fresh_selector(
                engine, source_name=source_name,
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
        )
    except PinBumpStop as exc:
        # gpt-dev-agent review (session ses_5307d9c58ec645cb, second pass):
        # prefer the ALREADY-LOADED `state` local for context when we have
        # one -- re-reading disk here could itself hit a corrupt/missing
        # state.json and mask the ORIGINAL structured failure with a
        # different, unrelated one. Only best-effort re-read from disk when
        # we don't already have a state (e.g. the exception came from
        # inside _run_phases, which may have advanced state past what this
        # local variable holds) -- and that best-effort attempt must never
        # itself raise and replace `exc`.
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


def _compile_tree_oid(engine: str, vendor_root: Path, patch_ids: Sequence[str]) -> str:
    """Exact effective source-tree identity used to invalidate compile resume."""

    allowed_untracked = set(patch_rebase._overlay_texts(engine))
    for _, file_patch in _selected_file_patches(engine, patch_ids):
        if file_patch.create:
            allowed_untracked.add(file_patch.path)
    try:
        return source_identity.git_tree_oid(
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


def _git_head_text(root: Path, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
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
                f"build the bumped tree, then {resume} --cpu-smoke-build <build dir>",
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
                f"fix the cause (see {out_dir / 'server.log'}), then {resume} --cpu-smoke-build {build_dir}",
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
    _record_note(vendor_root, f"pin-bump hardware acceptance: PASSED against stored baselines (evidence {evidence})")


def _record_note(vendor_root: Path, note: str) -> None:
    record = releases.record_for_checkout(vendor_root)
    existing = record.notes or ""
    if note not in existing:
        record.notes = (existing.rstrip() + "\n" + note).lstrip()
        record.save()


def _record_compile_skip(vendor_root: Path) -> None:
    note = "pin-bump source compile check: SKIPPED explicitly with --skip-compile"
    try:
        record = releases.record_for_checkout(vendor_root)
        existing = record.notes or ""
        if note not in existing:
            record.notes = (existing.rstrip() + "\n" + note).lstrip()
            record.save()
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
        if phase not in {"compile", "cpu-smoke", "reaudit", "complete"}
    ]
    state.next_phase = "compile"
    state.compile_tree_oid = ""
    state.compile_status = ""
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
    """Reconciles the RESUMING invocation's selector against the state's
    recorded one. Caller supplies none -> use the persisted selector;
    caller supplies one -> it must exactly equal the persisted name.
    Never silently reinterprets -- a real, deliberate change requires a
    fresh run, not a --resume."""
    if source_name is None:
        return state.selector_kind, state.selector_name
    if (state.selector_kind, state.selector_name) != ("source", source_name):
        raise PinBumpStop(
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
        )
    return state.selector_kind, state.selector_name


def _resume_platform(state: PinBumpState, *, platform_name: str | None) -> str | None:
    """The compile platform of a resumed run: the saved one unless the resuming invocation names it; a different name
    is refused (the saved compile result was produced with the saved platform)."""
    saved = state.compile_platform or None
    if platform_name is None or saved is None:
        return platform_name or saved
    if platform_name != saved:
        raise PinBumpStop(
            "resume",
            "RESUME_PLATFORM_MISMATCH",
            f"--resume was given --platform {platform_name!r}, but this run's recorded compile platform is {saved!r}",
            evidence={"resumed_platform": platform_name, "state_platform": saved},
            recommended_actions=[
                "omit --platform on --resume to use the run's original platform, or start a fresh run with the new one",
            ],
        )
    return saved


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
            state = PinBumpState(
                schema_version=3,
                run_id=uuid.uuid4().hex,
                from_ref=from_ref,
                from_sha="",
                to_ref=target_ref,
                to_sha=to_sha,
                transition_commit="",
                tree_name="local",
                tree_path=str(vendor_root),
                completed_phases=["preflight"],
                next_phase="declare",
                selector_kind=selector_kind,
                selector_name=selector_name,
                selector_patch_ids=selector_patch_ids,
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
            state.completed_phases.append("declare")
            state.next_phase = "pull"
            state.save(report_dir)

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
            state.completed_phases.append("pull")
            state.next_phase = "audit"
            state.save(report_dir)

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
            # Persist the result into the ReleaseRecord exactly like
            # `bigcherry audit` (cli/source.py's cmd_audit) does -- otherwise
            # apply_known_good's own `record.audit.get("passed")` gate (a
            # DIFFERENT, persisted-record check, not this in-memory report)
            # never sees a pass and refuses every real run. Found live: this
            # bump's first successful run through the coverage phase still
            # failed at apply for exactly this reason.
            record = releases.record_for_checkout(vendor_root)
            record.audit = releases.summarise_audit(report, strict=True)
            if record.stage == "pulled":
                record.advance_to("audited")
            record.save()
            state.completed_phases.append("audit")
            state.next_phase = "patch-lint"
            state.save(report_dir)

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
            state.completed_phases.append("patch-lint")
            state.next_phase = "coverage"
            state.save(report_dir)

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
            coverage["test_failures"] = run_patch_tests(repo_root)
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
            state.completed_phases.append("coverage")
            state.next_phase = "apply"
            state.save(report_dir)
        else:
            recipe_report_path = report_dir / "rebase-recipe.json"
            coverage = (
                json.loads((report_dir / "coverage.json").read_text(encoding="utf-8"))
                if (report_dir / "coverage.json").is_file()
                else {}
            )

        # gpt-dev-agent review of c236acc (P1, session ses_5307d9c58ec645cb):
        # the digest check used to run ONLY in the resumed (`else`) branch,
        # so a fresh same-process coverage->apply never re-checked, and a
        # deleted report skipped the check entirely instead of failing
        # closed. Unconditional, immediately before apply, covers both.
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
            state.completed_phases.append("apply")
            state.next_phase = "compile"
            state.save(report_dir)

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
            state.completed_phases.append("compile")
            state.next_phase = "cpu-smoke"
            state.save(report_dir)

        if state.next_phase == "cpu-smoke":
            # MEN15: the first smoke of the bumped engine, before any GPU smoke. It needs no GPU, so a busy host
            # cannot hold it up; only an explicit --skip-cpu-smoke omits it, and the skip is saved with the run.
            if skip_cpu_smoke:
                state.cpu_smoke_status = "skipped"
            else:
                run_phase_cpu_smoke(engine=engine, build_dir=cpu_smoke_build, out_dir=report_dir / "cpu-smoke", target_ref=target_ref)
                state.cpu_smoke_status = "passed"
            if accept and state.accept_status != "passed":
                run_phase_accept(engine=engine, out_dir=report_dir / "accept", target_ref=target_ref, vendor_root=vendor_root)
                state.accept_status = "passed"
            state.completed_phases.append("cpu-smoke")
            state.next_phase = "reaudit"
            state.save(report_dir)

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
            state.completed_phases.append("reaudit")
            state.next_phase = "complete"
            state.save(report_dir)

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


def _commit_release_records(*, repo_root: Path, target_ref: str) -> None:
    """Commit exactly pin-bump's own release-record output (releases/<tag>.json,
    releases/index.json, releases/<tag>-patches.md if generated) -- the
    symmetric close to run_phase_declare()'s controlled commit at the start
    of the bump.

    gpt-dev-agent review, 2026-08-31 (converged design, see the b10705 bump's
    real incident): without this, a successful bump leaves
    require_clean_bigcherry() refusing the very next MANDATORY step (the real
    build+smoke test, bump skill section 4b) until a human notices the dirty
    tree and commits these files by hand. require_clean_bigcherry() itself
    stays strict -- gpt's recommendation was explicitly against adding a
    dirty-tree allowlist there, since that would weaken the production
    provenance boundary and can't distinguish this orchestrator's own output
    from an unrelated concurrent edit to the same paths.

    Deliberately narrow: exact pathspecs only (never `git add -A`), and
    `git commit --only --` so even a concurrently staged unrelated change on
    this shared, multi-agent working tree (CLAUDE.md: never assume exclusive
    access) cannot be swept into this commit. NEVER touches
    releases/pin-transition.json -- that marker stays uncommitted-until-
    cleared until build+smoke has actually passed (bump skill step 5); this
    function has no visibility into whether that happened and must not
    guess.

    Idempotent: a --resume re-run (or a race where another process already
    committed these exact bytes) is a silent no-op, not an error.
    """
    owned = [f"releases/{target_ref}.json", "releases/index.json"]
    if (repo_root / "releases" / f"{target_ref}-patches.md").is_file():
        owned.append(f"releases/{target_ref}-patches.md")

    existing = [p for p in owned if (repo_root / p).exists()]
    if not existing:
        return  # nothing this orchestrator owns exists yet -- idempotent no-op

    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--", *existing],
        capture_output=True,
        text=True,
    )
    if status.returncode == 0 and not status.stdout.strip():
        return  # already committed -- idempotent, matches a --resume re-run

    add_result = subprocess.run(
        ["git", "-C", str(repo_root), "add", "--", *existing],
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        raise PinBumpStop(
            "complete",
            "RELEASE_RECORD_COMMIT_FAILED",
            f"git add failed for release records: {add_result.stderr.strip()}",
            evidence={"paths": existing, "stderr": add_result.stderr.strip()},
            recommended_actions=[
                "inspect the repo state directly",
                "rerun with --resume",
            ],
        )
    commit_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "commit",
            "--only",
            "-m",
            f"pin-bump: record release {target_ref}",
            "--",
            *existing,
        ],
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise PinBumpStop(
            "complete",
            "RELEASE_RECORD_COMMIT_FAILED",
            f"git commit --only failed for release records: {commit_result.stderr.strip()}",
            evidence={"paths": existing, "stderr": commit_result.stderr.strip()},
            recommended_actions=[
                "inspect the repo state directly",
                "rerun with --resume",
            ],
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
