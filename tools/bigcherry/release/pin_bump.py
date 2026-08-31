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

import hashlib
import json
import subprocess
import sys
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
    #: compat.recipe removal plan (gpt-dev-agent-reviewed, session
    #: ses_5307d9c58ec645cb), schema 2+ only. Binds WHICH selector this run
    #: was started with -- "recipe" (legacy) or "source" (v2) -- so
    #: --resume reconstructs the identical selector rather than silently
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

    def as_dict(self) -> dict:
        d = {
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
        if self.schema_version >= 2:
            d["selector"] = {
                "kind": self.selector_kind,
                "name": self.selector_name,
                "patch_ids": list(self.selector_patch_ids),
            }
            d["coverage_report_sha256"] = self.coverage_report_sha256
        return d

    def save(self, state_dir: Path) -> Path:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "state.json"
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, state_dir: Path) -> "PinBumpState":
        data = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        selector = data.get("selector") or {}
        return cls(
            schema_version=data["schema_version"], run_id=data["run_id"],
            from_ref=data["target"]["from_ref"], from_sha=data["target"]["from_sha"],
            to_ref=data["target"]["to_ref"], to_sha=data["target"]["to_sha"],
            transition_commit=data["transition_commit"],
            tree_name=data["tree"]["name"], tree_path=data["tree"]["path"],
            completed_phases=list(data["resume"]["completed_phases"]),
            next_phase=data["resume"]["next_phase"],
            selector_kind=selector.get("kind", ""),
            selector_name=selector.get("name", ""),
            selector_patch_ids=tuple(selector.get("patch_ids", ())),
            coverage_report_sha256=data.get("coverage_report_sha256", ""),
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
    """Returns an UNACQUIRED lock -- the caller uses it as `with lock:`, whose
    __enter__ does the single real acquire() call. (Found live on pin-bump's
    first real invocation: this used to call .acquire() itself AND get used
    as a context manager, double-acquiring the same lock in one process and
    tripping its own "already held" check.)"""
    from ..core.context import ProjectContext

    context = ProjectContext.resolve(project_root=project_root)
    return tree_activity.MaintenanceLock(context.work_root, context.project_root)


def _sync_campaign_mirror_best_effort(*, target_ref: str, revision: str) -> None:
    """`bigcherry build`'s isolated-worktree materialization resolves the
    pinned ref against a SEPARATE bare mirror (ProjectContext.upstream_repo,
    e.g. work/upstream/llama.cpp.git) -- not vendor/llama.cpp, which this
    phase just pulled. That mirror has its own independent fetch history
    and does not learn about a new tag just because vendor/llama.cpp did;
    without this, the very next `bigcherry build` after any real bump fails
    immediately with an ambiguous-ref git error. Found live TWICE (the
    b10680->b10687 and b10687->b10692 bumps) before being made automatic.

    Deliberately best-effort and NEVER raises PinBumpStop: this is a build
    convenience, not a bump-correctness requirement, and a mirror that
    doesn't exist yet (a tree that's never run a campaign build) or a
    network hiccup here must not block a real, already-verified pull."""
    from ..core.context import ProjectContext

    try:
        context = ProjectContext.resolve()
        mirror = context.upstream_repo
        if not (mirror / "HEAD").is_file() and not (mirror / ".git").exists():
            return  # no mirror yet -- nothing to sync
        already = subprocess.run(
            ["git", "-C", str(mirror), "rev-parse", "--verify", f"{target_ref}^{{commit}}"],
            capture_output=True, text=True,
        )
        if already.returncode == 0:
            return  # already resolvable, nothing to do
        subprocess.run(
            ["git", "-C", str(mirror), "fetch", "--depth=1", "origin", revision],
            capture_output=True, text=True, timeout=120,
        )
        tag_result = subprocess.run(
            ["git", "-C", str(mirror), "tag", target_ref, "FETCH_HEAD"],
            capture_output=True, text=True,
        )
        if tag_result.returncode != 0:
            print(
                f"pin-bump: campaign mirror sync for {target_ref!r} did not tag "
                f"cleanly (git tag exit {tag_result.returncode}): "
                f"{tag_result.stderr.strip()}",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 -- best-effort convenience, never fatal
        print(
            f"pin-bump: campaign mirror sync for {target_ref!r} failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        pass


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
    *, target_ref: str, recipe_name: str | None = None, root: Path | None = None,
    dispositions_dir: Path | None = None, resume: bool = False,
    report_dir: Path | None = None,
) -> PinBumpResult:
    """The Phase 1 single-tree orchestrator. Raises PinBumpStop (never a
    bare exception) on anything not provably safe to auto-proceed past;
    the caller is expected to catch it, write ``failure_envelope(...)``
    under ``report_dir``, and exit non-zero.

    ``recipe_name=None`` means "no selector explicitly supplied": a fresh
    run defaults to ``"bigcherry"`` (today's real behavior, unchanged by
    the compat.recipe removal plan's schema-2 binding work -- see
    ``_run_phases``' state-creation branch); a ``--resume`` with no
    selector reuses whatever the original invocation was started with
    (``_resume_selector``), rather than re-defaulting.
    """
    from .. import __main__ as legacy
    from ..source import audit as source_audit
    from ..patch import catalog as patch_catalog

    repo_root = paths.REPO_ROOT
    vendor_root = root if root is not None else paths.llama_root()
    dispositions_dir = dispositions_dir or paths.DISPOSITIONS
    report_dir = report_dir or (paths.ARTIFACTS / "pin-bump" / f"resume-{target_ref}")

    # Bound before the try, unconditionally: RESUME_STATE_MISSING and
    # _load_state_or_stop() can both raise before any assignment below
    # would otherwise run, and the except block needs `state` defined
    # either way (None means "fall back to a best-effort disk re-read").
    state: "PinBumpState | None" = None
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
                    "resume", "RESUME_STATE_MISSING",
                    f"--resume was given but no state.json exists under {report_dir}",
                    evidence={"report_dir": str(report_dir)},
                    recommended_actions=[
                        "omit --resume to start a fresh run, or point "
                        "--report-dir at the run you meant to resume",
                    ],
                )
            state = _load_state_or_stop(report_dir)
            _validate_resume(state, target_ref=target_ref, vendor_root=vendor_root)
            _, resolved_recipe_name = _resume_selector(state, recipe_name=recipe_name)
        else:
            state = None
            resolved_recipe_name = recipe_name if recipe_name is not None else "bigcherry"

        return _run_phases(
            state=state, target_ref=target_ref, recipe_name=resolved_recipe_name,
            repo_root=repo_root, vendor_root=vendor_root,
            dispositions_dir=dispositions_dir, report_dir=report_dir,
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


def _load_state_or_stop(report_dir: Path) -> "PinBumpState":
    """Wraps PinBumpState.load() so a corrupt/unreadable state.json becomes
    a structured PinBumpStop (gpt-dev-agent review, session
    ses_5307d9c58ec645cb, second pass) -- pin_bump's own contract is "raise
    PinBumpStop, never a bare exception", which PinBumpState.load() itself
    does not uphold (JSONDecodeError/KeyError/OSError on bad input)."""
    try:
        return PinBumpState.load(report_dir)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PinBumpStop(
            "resume", "RESUME_STATE_INVALID",
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


def _require_coverage_report(state: "PinBumpState", report_path: Path) -> None:
    """Unconditional pre-apply gate (gpt-dev-agent review of c236acc, P1,
    session ses_5307d9c58ec645cb): the coverage phase's rebase-recipe.json
    must exist, and its live bytes must match what coverage actually wrote,
    every time apply is about to run -- fresh same-process coverage->apply
    included, not just a --resume. A missing or empty-bound digest is
    itself a failure, not something to silently skip past."""
    if not state.coverage_report_sha256:
        raise PinBumpStop(
            "apply", "COVERAGE_REPORT_UNBOUND",
            "no coverage_report_sha256 was ever recorded for this run -- "
            "refusing to apply an unproven report",
            recommended_actions=["re-run coverage from a fresh run"],
        )
    if not report_path.is_file():
        raise PinBumpStop(
            "apply", "COVERAGE_REPORT_MISSING",
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
            "apply", "COVERAGE_REPORT_MISSING",
            f"{report_path} disappeared before it could be read -- "
            "refusing to apply",
            evidence={"report_path": str(report_path), "error": str(exc)},
            recommended_actions=[
                "restore the original rebase-recipe.json, or re-run "
                "coverage from a fresh run",
            ],
        ) from exc
    except OSError as exc:
        raise PinBumpStop(
            "apply", "COVERAGE_REPORT_UNREADABLE",
            f"{report_path} could not be read: {exc}",
            evidence={"report_path": str(report_path), "error": str(exc)},
            recommended_actions=[
                "inspect the report file's permissions/state by hand",
                "re-run coverage from a fresh run",
            ],
        ) from exc
    if live_digest != state.coverage_report_sha256:
        raise PinBumpStop(
            "apply", "COVERAGE_REPORT_MODIFIED",
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
    state: "PinBumpState", *, target_ref: str, vendor_root: Path,
) -> None:
    """compat.recipe removal plan (gpt-dev-agent-reviewed, session
    ses_5307d9c58ec645cb): resume identity gaps that existed even before
    this plan -- state stored target/tree, but the phases never actually
    checked the RESUMING invocation's target_ref/vendor_root against them.
    A --resume with a different target or tree must fail closed, not
    silently continue against the wrong state."""
    if state.schema_version < 2:
        raise PinBumpStop(
            "resume", "LEGACY_STATE_SELECTOR_UNBOUND",
            "this in-flight run's state predates selector binding (schema "
            f"{state.schema_version} < 2) -- it cannot prove which selector "
            "(--recipe/--source NAME) it was started with, so resuming it "
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
            "resume", "RESUME_TARGET_MISMATCH",
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
            "resume", "RESUME_TREE_MISMATCH",
            f"--resume is running against {resolved_tree!r}, but this run's "
            f"recorded tree is {state.tree_path!r}",
            evidence={"resumed_tree": resolved_tree, "state_tree": state.tree_path},
            recommended_actions=[
                "resume from the same vendor checkout the run was started against",
            ],
        )


def _resume_selector(
    state: "PinBumpState", *, recipe_name: str | None,
) -> tuple[str, str]:
    """Reconciles the RESUMING invocation's selector against the state's
    recorded one. Caller supplies none -> use the persisted selector;
    caller supplies one -> it must exactly equal the persisted kind+name.
    Never silently reinterprets -- a real, deliberate change requires a
    fresh run, not a --resume."""
    if recipe_name is None:
        return state.selector_kind, state.selector_name
    if (state.selector_kind, state.selector_name) != ("recipe", recipe_name):
        raise PinBumpStop(
            "resume", "RESUME_SELECTOR_MISMATCH",
            f"--resume was given --recipe {recipe_name!r}, but this run's "
            f"recorded selector is {state.selector_kind}:{state.selector_name!r}",
            evidence={
                "resumed_selector": f"recipe:{recipe_name}",
                "state_selector": f"{state.selector_kind}:{state.selector_name}",
            },
            recommended_actions=[
                "omit --recipe on --resume to use the run's original selector, "
                "or start a fresh run with the new one",
            ],
        )
    return state.selector_kind, state.selector_name


def _require_selector_membership_unchanged(
    state: "PinBumpState", *, recipe_name: str,
) -> None:
    """Before coverage runs (fresh or resumed), re-derive the selector's
    real patch-id membership and require it still matches what the state
    recorded -- catches a concurrent config/recipes.toml or patch-registry
    edit on this shared, multi-agent repo, whether or not this is a
    --resume. Membership drift requires a deliberate restart, not a
    silent scope change to a production release run."""
    current_ids = tuple(sorted(
        patch_rebase._selection_patch_ids(recipe_name=recipe_name, all_patches=False)
    ))
    recorded_ids = tuple(sorted(state.selector_patch_ids))
    if current_ids != recorded_ids:
        raise PinBumpStop(
            "coverage", "RESUME_SELECTION_CHANGED",
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
    *, state: "PinBumpState | None", target_ref: str, recipe_name: str,
    repo_root: Path, vendor_root: Path, dispositions_dir: Path, report_dir: Path,
) -> "PinBumpResult":
    from .. import __main__ as legacy
    from ..source import audit as source_audit
    from ..patch import catalog as patch_catalog

    with acquire_maintenance_lock(repo_root):
        if state is None:
            from_ref, to_sha = run_phase_preflight(repo_root=repo_root, target_ref=target_ref)
            selector_patch_ids = tuple(sorted(
                patch_rebase._selection_patch_ids(recipe_name=recipe_name, all_patches=False)
            ))
            state = PinBumpState(
                schema_version=2, run_id=uuid.uuid4().hex, from_ref=from_ref, from_sha="",
                to_ref=target_ref, to_sha=to_sha, transition_commit="",
                tree_name="local", tree_path=str(vendor_root),
                completed_phases=["preflight"], next_phase="declare",
                selector_kind="recipe", selector_name=recipe_name,
                selector_patch_ids=selector_patch_ids,
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
            _sync_campaign_mirror_best_effort(target_ref=target_ref, revision=state.to_sha)
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
            # Persist the result into the ReleaseRecord exactly like
            # `bigcherry audit` (cli/source.py's cmd_audit) does -- otherwise
            # apply_known_good's own `record.audit.get("passed")` gate (a
            # DIFFERENT, persisted-record check, not this in-memory report)
            # never sees a pass and refuses every real run. Found live: this
            # bump's first successful run through the coverage phase still
            # failed at apply for exactly this reason.
            from ..release import records as releases

            record = legacy._record_for(vendor_root)
            record.audit = releases.summarise_audit(report, strict=True)
            if record.stage == "pulled":
                record.advance_to("audited")
            record.save()
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
            _require_selector_membership_unchanged(state, recipe_name=recipe_name)
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
            state.coverage_report_sha256 = _sha256_file(recipe_report_path)
            state.completed_phases.append("coverage")
            state.next_phase = "apply"
            state.save(report_dir)
        else:
            recipe_report_path = report_dir / "rebase-recipe.json"
            coverage = json.loads((report_dir / "coverage.json").read_text(encoding="utf-8")) \
                if (report_dir / "coverage.json").is_file() else {}

        # gpt-dev-agent review of c236acc (P1, session ses_5307d9c58ec645cb):
        # the digest check used to run ONLY in the resumed (`else`) branch,
        # so a fresh same-process coverage->apply never re-checked, and a
        # deleted report skipped the check entirely instead of failing
        # closed. Unconditional, immediately before apply, covers both.
        if state.next_phase == "apply":
            _require_coverage_report(state, recipe_report_path)
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

        if state.next_phase == "complete" and "complete" not in state.completed_phases:
            _write_release_doc_best_effort(
                repo_root=repo_root, vendor_root=vendor_root,
                recipe_name=recipe_name, target_ref=target_ref,
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
        capture_output=True, text=True,
    )
    if status.returncode == 0 and not status.stdout.strip():
        return  # already committed -- idempotent, matches a --resume re-run

    add_result = subprocess.run(
        ["git", "-C", str(repo_root), "add", "--", *existing],
        capture_output=True, text=True,
    )
    if add_result.returncode != 0:
        raise PinBumpStop(
            "complete", "RELEASE_RECORD_COMMIT_FAILED",
            f"git add failed for release records: {add_result.stderr.strip()}",
            evidence={"paths": existing, "stderr": add_result.stderr.strip()},
            recommended_actions=["inspect the repo state directly", "rerun with --resume"],
        )
    commit_result = subprocess.run(
        [
            "git", "-C", str(repo_root), "commit", "--only", "-m",
            f"pin-bump: record release {target_ref}", "--", *existing,
        ],
        capture_output=True, text=True,
    )
    if commit_result.returncode != 0:
        raise PinBumpStop(
            "complete", "RELEASE_RECORD_COMMIT_FAILED",
            f"git commit --only failed for release records: {commit_result.stderr.strip()}",
            evidence={"paths": existing, "stderr": commit_result.stderr.strip()},
            recommended_actions=["inspect the repo state directly", "rerun with --resume"],
        )


def _write_release_doc_best_effort(
    *, repo_root: Path, vendor_root: Path, recipe_name: str, target_ref: str,
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

    compat.recipe removal plan (gpt-dev-agent review, session
    ses_5307d9c58ec645cb): reads patch_ids from the SAME
    rebase-recipe.json report that authorized `apply`, rather than
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
                json.loads(report_path.read_text(encoding="utf-8"))
                ["selection"]["patch_ids"]
            )
        else:
            patch_ids = patch_rebase._selection_patch_ids(
                recipe_name=recipe_name, all_patches=False,
            )
        pin_info = {
            "llama.cpp revision": patch_rebase._git(vendor_root, "rev-parse", "HEAD"),
            "bigcherry revision": patch_rebase._git(repo_root, "rev-parse", "HEAD"),
            "recipe": recipe_name,
            "target": target_ref,
        }
        doc = patch_docs.render_patch_selection_doc(
            patch_ids=patch_ids, pin_info=pin_info,
            selection_label=f"--recipe {recipe_name}",
        )
        release_records.RELEASES_DIR.mkdir(parents=True, exist_ok=True)
        (release_records.RELEASES_DIR / f"{target_ref}-patches.md").write_text(doc, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 -- best-effort convenience, never fatal
        print(
            f"pin-bump: release doc for {target_ref!r} (recipe {recipe_name!r}) "
            f"was not written: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
