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

import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import paths
from ..core import tree_activity
from ..patch import disposition as patch_disposition
from ..patch import rebase as patch_rebase
from .. import pin_transition
from .. import recipes as recipes_module

PHASES = (
    "preflight",
    "declare",
    "pull",
    "audit",
    "patch-lint",
    "coverage",
    "apply",
    "reaudit",
    "complete",
)


class PinBumpStop(Exception):
    """Raised by any phase to STOP the orchestrator. Carries everything
    needed to build the structured failure envelope -- callers should
    never need to re-derive context from a bare message string."""

    def __init__(
        self, phase: str, code: str, summary: str, *,
        human_required: bool = True, retryable: bool = True,
        evidence: dict[str, Any] | None = None,
        recommended_actions: tuple[str, ...] = (),
    ):
        super().__init__(summary)
        self.phase = phase
        self.code = code
        self.summary = summary
        self.human_required = human_required
        self.retryable = retryable
        self.evidence = evidence or {}
        self.recommended_actions = recommended_actions
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

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id,
            "target": {
                "from_ref": self.from_ref, "from_sha": self.from_sha,
                "to_ref": self.to_ref, "to_sha": self.to_sha,
            },
            "transition_commit": self.transition_commit,
            "tree": {"name": self.tree_name, "path": self.tree_path},
            "resume": {
                "completed_phases": list(self.completed_phases),
                "next_phase": self.next_phase,
            },
        }

    def save(self, state_dir: Path) -> Path:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, state_dir: Path) -> "PinBumpState":
        data = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        return cls(
            schema_version=data["schema_version"], run_id=data["run_id"],
            from_ref=data["target"]["from_ref"], from_sha=data["target"]["from_sha"],
            to_ref=data["target"]["to_ref"], to_sha=data["target"]["to_sha"],
            transition_commit=data["transition_commit"],
            tree_name=data["tree"]["name"], tree_path=data["tree"]["path"],
            completed_phases=list(data["resume"]["completed_phases"]),
            next_phase=data["resume"]["next_phase"],
        )


def failure_envelope(run_id: str, target: dict, transition_commit: str, tree: dict, exc: PinBumpStop) -> dict:
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
        ["git", "-C", str(root), *args], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise PinBumpStop(
            "preflight", "GIT_COMMAND_FAILED",
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}",
            evidence={"args": list(args), "stderr": result.stderr.strip()},
            recommended_actions=["inspect the repo state directly", "rerun with --resume"],
        )
    return result.stdout.strip()


def require_clean_controller_checkout(repo_root: Path) -> None:
    status = _git(repo_root, "status", "--porcelain")
    if status:
        raise PinBumpStop(
            "preflight", "CONTROLLER_DIRTY",
            "the bigcherry controller checkout has uncommitted changes",
            evidence={"git_status_porcelain": status},
            recommended_actions=["commit or revert the uncommitted changes, then retry"],
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
    failed_checks = {check["id"] for check in audit_report.get("checks", ()) if not check.get("ok", True)}
    if failed_checks != {"overlay.vendor_sync"}:
        return False, []
    overlay_check = next(
        check for check in audit_report["checks"] if check["id"] == "overlay.vendor_sync"
    )
    drifted = list(overlay_check.get("actual", ()) or ())
    return True, drifted


def acquire_maintenance_lock(project_root: Path) -> tree_activity.MaintenanceLock:
    from ..core.context import ProjectContext

    context = ProjectContext.resolve(project_root=project_root)
    lock = tree_activity.MaintenanceLock(context.work_root, context.project_root)
    lock.acquire()
    return lock


def run_phase_preflight(*, repo_root: Path, target_ref: str) -> tuple[str, str]:
    """Returns (resolved_from_ref, resolved_to_sha). Requires a clean
    controller checkout and refuses an incompatible existing transition
    marker (RE48's invariant, reused rather than re-derived)."""
    require_clean_controller_checkout(repo_root)
    marker_path = paths.REPO_ROOT / "releases" / "pin-transition.json"
    if marker_path.is_file() and pin_transition.committed_state(marker_path) != "committed-clean":
        raise PinBumpStop(
            "preflight", "TRANSITION_MARKER_UNCOMMITTED",
            "an existing pin-transition marker is uncommitted -- a prior bump "
            "was declared but not committed",
            evidence={"marker_path": str(marker_path)},
            recommended_actions=[
                "commit the marker together with config/recipes.toml, or delete "
                "it if that bump was abandoned",
            ],
        )
    from .pin import resolve_pin_sha

    try:
        to_sha = resolve_pin_sha(target_ref)
    except Exception as exc:  # upstream.UpstreamError et al.
        raise PinBumpStop(
            "preflight", "UNRESOLVABLE_PIN", f"could not resolve {target_ref!r}: {exc}",
            evidence={"target_ref": target_ref},
            recommended_actions=["verify the ref exists upstream", "rerun with --resume"],
        ) from exc
    _recipes, current_pinned = recipes_module.load()
    return current_pinned, to_sha


def run_phase_declare(*, repo_root: Path, target_ref: str) -> str:
    """Calls the existing, already-real repin() + pin_transition.write(),
    then commits config/recipes.toml + the marker together. Returns the
    declaring commit SHA."""
    from .pin import resolve_pin_sha

    old = recipes_module.repin(target_ref)
    from_sha = resolve_pin_sha(old)
    to_sha = resolve_pin_sha(target_ref)
    declaring_before = _git(repo_root, "rev-parse", "HEAD")
    pin_transition.write(from_sha, to_sha, target_ref, declaring_before)
    _git(repo_root, "add", "config/recipes.toml", "releases/pin-transition.json")
    _git(
        repo_root, "commit", "-m",
        f"pin: {old} -> {target_ref} ({to_sha[:8]}) -- rebase in flight (pin-bump orchestrator)",
    )
    return _git(repo_root, "rev-parse", "HEAD")


def stop_on_bad_rebase_status(
    *, phase: str, report: dict, patch_id: str, entry: dict,
) -> None:
    status = entry["status"]
    code = {
        "FAILED": "PATCH_FAILED_NEEDS_RECONCILIATION",
        "BLOCKED_BY_DEPENDENCY": "PATCH_BLOCKED_BY_DEPENDENCY",
        "QUARANTINED": "PATCH_QUARANTINED",
    }.get(status, "PATCH_REBASE_BAD_STATUS")
    raise PinBumpStop(
        phase, code, f"{patch_id} is {status} against the new revision",
        evidence={
            "patch_id": patch_id, "status": status,
            "requires": entry.get("requires", ()),
        },
        recommended_actions=[
            "reconcile the patch (see the rebase report's per-edit reason_code)",
            "or record a known_broken disposition via `bigcherry patch-disposition set` "
            "if this patch is not in the build recipe",
            "rerun `bigcherry pin-bump --resume`",
        ],
    )


def enforce_all_patches_clean_or_dispositioned(
    *, all_report: dict, recipe_report: dict, catalog_states: dict[str, str],
    dispositions_dir: Path, target_revision: str, phase: str = "coverage",
) -> dict:
    """HI152's coverage gate. Raises PinBumpStop with the coverage block
    as evidence if anything is uncovered."""
    dispositions = patch_disposition.list_dispositions(dispositions_dir)
    recipe_ids = frozenset(entry["patch_id"] for entry in recipe_report.get("patches", ()))
    result = patch_disposition.compute_coverage(
        catalog_states=catalog_states, all_report=all_report,
        recipe_patch_ids=recipe_ids, dispositions=dispositions,
        target_revision=target_revision,
    )
    if not result.complete:
        raise PinBumpStop(
            phase, "COVERAGE_INCOMPLETE",
            f"{len(result.uncovered_patch_ids)} patch(es) uncovered against {target_revision[:12]}",
            evidence=result.as_dict(),
            recommended_actions=[
                "reconcile each uncovered patch, or record a known_broken "
                "disposition via `bigcherry patch-disposition set` if it is "
                "not in the build recipe",
                "rerun `bigcherry pin-bump --resume`",
            ],
        )
    return result.as_dict()


@dataclass
class PinBumpResult:
    ok: bool
    state: PinBumpState
    coverage: dict


def run(
    *, target_ref: str, recipe_name: str = "bigcherry", root: Path | None = None,
    dispositions_dir: Path | None = None, resume: bool = False,
    report_dir: Path | None = None,
) -> PinBumpResult:
    """The Phase 1 single-tree orchestrator. Raises PinBumpStop (never a
    bare exception) on anything not provably safe to auto-proceed past;
    the caller is expected to catch it, write ``failure_envelope(...)``
    under ``report_dir``, and exit non-zero."""
    from .. import __main__ as legacy
    from ..source import audit as source_audit
    from ..patch import catalog as patch_catalog

    repo_root = paths.REPO_ROOT
    vendor_root = root if root is not None else paths.llama_root()
    dispositions_dir = dispositions_dir or (paths.ARTIFACTS / "pin-bump" / "dispositions")
    report_dir = report_dir or (paths.ARTIFACTS / "pin-bump" / f"resume-{target_ref}")

    if resume and (report_dir / "state.json").is_file():
        state = PinBumpState.load(report_dir)
    else:
        state = None

    try:
        return _run_phases(
            state=state, target_ref=target_ref, recipe_name=recipe_name,
            repo_root=repo_root, vendor_root=vendor_root,
            dispositions_dir=dispositions_dir, report_dir=report_dir,
        )
    except PinBumpStop as exc:
        # `state` may have been created (and mutated) inside _run_phases
        # even though this `state` local is still whatever it was before
        # the call -- PinBumpStop always carries its own `evidence`, but
        # attach run/target/tree context here from what we DO know for
        # certain (the resume file on disk, updated after every phase).
        if (report_dir / "state.json").is_file():
            saved = PinBumpState.load(report_dir)
            exc.run_id = saved.run_id
            exc.target = {"from_ref": saved.from_ref, "to_ref": saved.to_ref}
            exc.transition_commit = saved.transition_commit
            exc.tree = {"name": saved.tree_name, "path": saved.tree_path}
        else:
            exc.run_id = "unresolved"
            exc.target = {"from_ref": "?", "to_ref": target_ref}
            exc.tree = {"name": "local", "path": str(vendor_root)}
        raise


def _run_phases(
    *, state: "PinBumpState | None", target_ref: str, recipe_name: str,
    repo_root: Path, vendor_root: Path, dispositions_dir: Path, report_dir: Path,
) -> "PinBumpResult":
    from .. import __main__ as legacy
    from ..source import audit as source_audit
    from ..patch import catalog as patch_catalog

    with acquire_maintenance_lock(repo_root):
        if state is None:
            from_ref, to_sha = run_phase_preflight(repo_root=repo_root, target_ref=target_ref)
            state = PinBumpState(
                schema_version=1, run_id=uuid.uuid4().hex, from_ref=from_ref, from_sha="",
                to_ref=target_ref, to_sha=to_sha, transition_commit="",
                tree_name="local", tree_path=str(vendor_root),
                completed_phases=["preflight"], next_phase="declare",
            )
            state.save(report_dir)

        if state.next_phase == "declare":
            declaring = run_phase_declare(repo_root=repo_root, target_ref=target_ref)
            state.transition_commit = declaring
            state.completed_phases.append("declare")
            state.next_phase = "pull"
            state.save(report_dir)

        if state.next_phase == "pull":
            from ..cli import source as cli_source
            from argparse import Namespace

            rc = cli_source.cmd_pull(Namespace(
                llama_root=None, ref=target_ref, recipe=recipe_name, full=False,
            ))
            if rc != 0:
                raise PinBumpStop(
                    "pull", "PULL_FAILED", "bigcherry pull did not exit cleanly",
                    evidence={"exit_code": rc},
                    recommended_actions=["inspect the vendor checkout directly", "rerun with --resume"],
                )
            state.completed_phases.append("pull")
            state.next_phase = "audit"
            state.save(report_dir)

        if state.next_phase == "audit":
            report = source_audit.audit(vendor_root)
            if not source_audit.passed(report, strict=False):
                safe, drifted = check_overlay_self_heal(report)
                if not safe:
                    raise PinBumpStop(
                        "audit", "AUDIT_FAILED", "source audit failed on more than overlay staleness",
                        evidence={"failed_checks": [c["id"] for c in report["checks"] if not c["ok"]]},
                        recommended_actions=["run `bigcherry audit --verbose` and reconcile", "rerun with --resume"],
                    )
                legacy._copy_overlay(vendor_root, dry_run=False)
                report = source_audit.audit(vendor_root)
                if not source_audit.passed(report, strict=False):
                    raise PinBumpStop(
                        "audit", "OVERLAY_STALENESS_UNPROVEN",
                        "overlay drift persisted after the narrow newline-only self-heal",
                        evidence={"drifted": drifted},
                        recommended_actions=["inspect the drifted file(s) by hand", "rerun with --resume"],
                    )
            state.completed_phases.append("audit")
            state.next_phase = "patch-lint"
            state.save(report_dir)

        if state.next_phase == "patch-lint":
            problems = patch_catalog.cross_check(allow_legacy_grandfather=True)
            if problems:
                raise PinBumpStop(
                    "patch-lint", "PATCH_LINT_FAILED", f"{len(problems)} patch-lint problem(s)",
                    evidence={"problems": problems},
                    recommended_actions=["fix the catalog/package mismatch", "rerun with --resume"],
                )
            state.completed_phases.append("patch-lint")
            state.next_phase = "coverage"
            state.save(report_dir)

        if state.next_phase == "coverage":
            all_report = patch_rebase.run_rebase_check(vendor_root, all_patches=True)
            recipe_report = patch_rebase.run_rebase_check(vendor_root, recipe_name=recipe_name)
            for entry in recipe_report.get("patches", ()):
                if entry["status"] not in patch_disposition.CLEAN_STATUSES:
                    stop_on_bad_rebase_status(
                        phase="coverage", report=recipe_report,
                        patch_id=entry["patch_id"], entry=entry,
                    )
            catalog_states = {
                patch_id: entry.state for patch_id, entry in patch_catalog.load_catalog().items()
            }
            coverage = enforce_all_patches_clean_or_dispositioned(
                all_report=all_report, recipe_report=recipe_report,
                catalog_states=catalog_states, dispositions_dir=dispositions_dir,
                target_revision=state.to_sha,
            )
            recipe_report_path = report_dir / "rebase-recipe.json"
            patch_rebase.write_report(recipe_report_path, recipe_report)
            (report_dir / "coverage.json").write_text(
                json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            state.completed_phases.append("coverage")
            state.next_phase = "apply"
            state.save(report_dir)
        else:
            recipe_report_path = report_dir / "rebase-recipe.json"
            coverage = json.loads((report_dir / "coverage.json").read_text(encoding="utf-8")) \
                if (report_dir / "coverage.json").is_file() else {}

        if state.next_phase == "apply":
            result = patch_rebase.apply_known_good(vendor_root, recipe_report_path, force=False, dry_run=False)
            if not result.ok:
                raise PinBumpStop(
                    "apply", "APPLY_FAILED", "known-good apply did not succeed",
                    evidence={"known_good": list(result.known_good_patch_ids)},
                    recommended_actions=["inspect the apply failure directly", "rerun with --resume"],
                )
            state.completed_phases.append("apply")
            state.next_phase = "reaudit"
            state.save(report_dir)

        if state.next_phase == "reaudit":
            report = source_audit.audit(vendor_root)
            if not source_audit.passed(report, strict=True):
                raise PinBumpStop(
                    "reaudit", "POST_APPLY_AUDIT_FAILED",
                    "source audit failed after a full known-good apply",
                    evidence={"failed_checks": [c["id"] for c in report["checks"] if not c["ok"]]},
                    recommended_actions=["run `bigcherry audit --verbose` and reconcile", "rerun with --resume"],
                )
            state.completed_phases.append("reaudit")
            state.next_phase = "complete"
            state.save(report_dir)

    return PinBumpResult(ok=True, state=state, coverage=coverage)
