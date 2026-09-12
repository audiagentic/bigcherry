---
id: BRC01
order: 1
plan: build-run-cleanup
state: in_progress
created-at: '2026-09-11T06:44:31.138556+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Remove legacy build/run shims and consolidate bridging layers

## Description

GPT-reviewed cleanup of the remaining production __main__ back-edges. The first implementation slice removes the cli.patch, cli.tuning, patch.rebase, and release.pin_bump dependencies on bigcherry.__main__ while preserving compatibility aliases at the entrypoint. Overlay copy/restore is a shared patch-domain concern, so it belongs in patch/overlay.py; cli.patch owns apply orchestration and release.records remains the record authority. Smoke/report, rocprof, replay/cache, and root-facade retirement remain separate follow-up work.

## Steps

1. Freeze the current call graph and compatibility surface: cli.patch.cmd_apply -> __main__._apply_exact_selection; cli.tuning.cmd_generate -> __main__._record_for; patch.rebase -> __main__._copy_overlay/_restore_overlay/_record_for; release.pin_bump -> __main__._copy_overlay/_record_for. Confirm all affected tests and do not reimplement cli.source, release.records, or release.pin.
2. Add tools/bigcherry/patch/overlay.py as the dependency-light canonical overlay copy/restore owner. Move the existing behavior without semantic changes: sorted traversal, raw-byte/newline-sensitive change detection, UTF-8 writes, simulation text for dry-run, original-content backups including missing files, normalized relative paths, O_NOFOLLOW restore, deletion of newly-created files, and per-file restore-failure isolation.
3. Move _apply_exact_selection from __main__.py into cli/patch.py unchanged in transaction order. Import patch.overlay, patch.apply, patchset, patch.rebase, patch_admission, and release.records at their canonical layers; use releases.record_for_checkout; retain local identity aliases _copy_overlay and _restore_overlay for tested compatibility lookup sites.
4. Retarget patch/rebase.py to patch.overlay.copy_overlay/restore_overlay and release.records.record_for_checkout; retarget release/pin_bump.py to patch.overlay and release.records; retarget cli/tuning.py to release.records. No protected domain may import cli.* or __main__.
5. Reduce __main__.py to entrypoint/compatibility identity aliases and existing supported command/parser exports. Retain _copy_overlay, _restore_overlay, _apply_exact_selection, and _record_for only as direct identity aliases where compatibility tests establish them; remove substantive implementations and dead support helpers.
6. Add AST-backed TR14.CLI_MAIN_BACKEDGE in check.py for static relative/absolute and literal dynamic imports resolving to bigcherry.__main__, excluding __main__.py and ignoring comments/docstrings. Avoid duplicate diagnostics and keep TR14.DOMAIN_CLI_IMPORT.
7. Retarget/add focused tests for cli.patch apply parity, rebase overlay/rollback, pin-bump self-heal, tuning generation, compatibility identity, and CLI layering. Validate before/after streams, exit codes, dry-run, admission-before-mutation, TOCTOU, rollback, release/evidence/tree-state identity, and current facade identity.

## Detailed Solution & Technical Design

GPT implementation design correction: the previous BRC01 statement that cli.patch alone could own _copy_overlay/_restore_overlay was architecturally incomplete. patch.rebase.py and release.pin_bump.py are protected domain modules and cannot import cli.patch. The canonical ownership must therefore be:

- tools/bigcherry/patch/overlay.py: dependency-light copy_overlay(...) and restore_overlay(...), with no CLI, release, or __main__ dependency. Preserve the existing helper's optional backup and sim_texts behavior so dry-run and rollback semantics remain identical.
- tools/bigcherry/cli/patch.py: canonical cmd_apply and _apply_exact_selection orchestration. It imports patch.overlay and release.records. Keep _copy_overlay = patch_overlay.copy_overlay and _restore_overlay = patch_overlay.restore_overlay as direct identity aliases for compatibility and existing lookup-site tests.
- tools/bigcherry/release/records.py: canonical record_for_checkout(root); its implementation is out of scope and must not be duplicated.
- tools/bigcherry/patch/rebase.py: imports patch.overlay and release.records directly. Replace legacy._copy_overlay, legacy._restore_overlay, and legacy._record_for with canonical owners without changing probing, known-good selection, partial reconciliation, report, dry-run, rollback, tree-state, or release-update algorithms.
- tools/bigcherry/release/pin_bump.py: imports patch.overlay and release.records directly. Replace only legacy helper lookups; preserve self-heal output, evidence invalidation, and phase behavior.
- tools/bigcherry/cli/tuning.py: imports release.records directly for cmd_generate; no behavior change.
- tools/bigcherry/__main__.py: entrypoint/compatibility shell. It may import cli.patch and release.records to expose identity aliases, but no production module other than __main__.py may import it. Aliases must be identity aliases, not wrappers.

Preserve _apply_exact_selection's exact order: live HEAD and source-ref resolution; live/ref mismatch refusal; exact source composition and patch-set/ordered patch-ID TOCTOU checks; unresolved-overlay refusal; admission and warnings before overlay mutation; audit gate; overlay simulation/copy; exact patch loading/application; real-apply overlay rollback; tree-state and mutation calculation; existing output; second record lookup and persistence; and evidence invalidation/update semantics. Preserve --force, --dry-run, allow_stale_validation_evidence, stdout/stderr, return codes 0/1/2, release/evidence/tree-state identities, and rebase-report delegation to patch.rebase.apply_known_good. Do not introduce a generic workflow abstraction or merge overlay rollback with patch rollback.

TR14.CLI_MAIN_BACKEDGE must AST-scan production tools/bigcherry/**/*.py except __main__.py. Detect absolute and relative imports that resolve to bigcherry.__main__, plus literal importlib.import_module/import_module/__import__ calls. Ignore comments/docstrings/string mentions and unrelated __main__ modules. Emit one stable diagnostic per offense and leave the existing domain-to-CLI boundary rule intact.

## Code Samples & Guidance

patch/overlay.py:
copy_overlay(root, *, dry_run, backup=None, sim_texts=None) -> list[str]
restore_overlay(root, backup) -> None

cli/patch.py:
_copy_overlay = patch_overlay.copy_overlay
_restore_overlay = patch_overlay.restore_overlay
_record_for = releases.record_for_checkout
# cmd_apply calls the local _apply_exact_selection; no cli.patch -> __main__ import.

patch/rebase.py and release/pin_bump.py:
from ..patch import overlay as patch_overlay
from ..release import records as releases
# use patch_overlay.copy_overlay/restore_overlay and releases.record_for_checkout.

__main__.py:
_copy_overlay = _patch_cli._copy_overlay
_restore_overlay = _patch_cli._restore_overlay
_apply_exact_selection = _patch_cli._apply_exact_selection
_record_for = _release_records.record_for_checkout

## Files

tools/bigcherry/patch/overlay.py (new canonical overlay primitive)
tools/bigcherry/cli/patch.py (apply orchestration, overlay compatibility aliases, _apply_exact_selection)
tools/bigcherry/__main__.py (remove substantive helper implementations; retain identity aliases/entrypoint)
tools/bigcherry/patch/rebase.py (canonical overlay and release-record imports)
tools/bigcherry/release/pin_bump.py (canonical overlay and release-record imports)
tools/bigcherry/cli/tuning.py (canonical release-record import)
tools/bigcherry/check.py (TR14.CLI_MAIN_BACKEDGE)
tools/tests/cli/test_cli_patch_apply.py (move/retarget apply workflow coverage)
tools/tests/patch/test_patch_rebase.py (overlay/rollback lookup sites)
tools/tests/release/test_pin_bump.py (canonical lookup-site patches)
tools/tests/release/test_releases.py (retain record tests; move apply workflow tests as needed)
tools/tests/cli/test_cli_tuning.py or existing generate tests
tools/tests/core/test_cli_layering.py
tools/tests/core/test_compatibility_facades.py
Inspect only: cli/source.py, release/records.py, release/pin.py
Later separate work: smoke/report, rocprof, replay/cache, patcher.py, inventory.py, replay_cache.py

## Validation

Run focused CLI/apply, rebase, pin-bump, tuning, release, and core-layering tests. Add AST unit tests for every supported absolute/relative/literal dynamic import form, comments/docstrings/string false positives, __main__.py exemption, and zero current production back-edges. Run patch and replay regressions, bigcherry check --quick, compileall, and the full offline suite where the local interpreter is available. Compare command help, aliases, stdout/stderr, exit codes, dry-run non-persistence, admission-before-overlay mutation, live HEAD/ref and composition TOCTOU failures, overlay and patch rollback, stale-evidence override, release/evidence/tree-state identity, and rebase-report behavior. Capture repository HEAD, branch, index, worktree list, and unrelated-file state before/after; only disposable fixtures may change.

## Effort & Risk



## Standards



## Acceptance Criteria

No production module other than __main__.py imports bigcherry.__main__, including static and literal dynamic forms. No protected patch or release domain imports cli. patch/overlay is the sole canonical copy/restore implementation, with byte/newline, backup, simulation, O_NOFOLLOW, and rollback behavior unchanged. cli.patch owns _apply_exact_selection and cmd_apply; release.records.record_for_checkout remains authoritative; cli.tuning, patch.rebase, and release.pin_bump use canonical owners. __main__.py contains only entrypoint/compatibility wiring and direct identity aliases where compatibility requires them. Existing admission-before-mutation, HEAD/ref, composition-TOCTOU, stale-evidence override, overlay/patch rollback, output streams, exit codes, dry-run, rebase-report, release/evidence/tree-state, and facade identity contracts remain unchanged. TR14.CLI_MAIN_BACKEDGE is AST-backed, ignores comments/docstrings, exempts __main__.py, avoids duplicate findings, and check --quick prevents reintroduction. No smoke/report, rocprof, replay/cache, or root-facade retirement is included.

## Notes

Initial assessment and design to be supplied by GPT gateway review before implementation.

GPT gateway initial response: audited main at 644b8207d11eaafa14c34f29fe4dc6c95388911d; verdict directionally coherent but still overly bridged. Recommended order: (1) cli/__main__ ownership inversion; (2) reduce __main__ to compatibility shell; (3) enforce no cli -> __main__ back-edge; (4) consolidate smoke/report; (5) reconcile rocprof; (6) retire root facades last. Safe to remove now: none. Root facades are intentional module-identity aliases. The audit also noted BRC01 was not visible on the audited tip, which is why this plan must be committed and pushed before the second review.

GPT second-pass file-level design completed against published main tip 7f61b3b5615cc429efe6d630bffe4a46697d0e10. Key correction: cli/main.py and release/pin.py are already canonical; first tranche is limited to source/patch ownership inversion and __main__ compatibility reduction. No implementation started.

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. This is the first production-code slice after planning reconciliation because current cmd_apply still delegates through __main__. The live remainder is narrowly _copy_overlay/_restore_overlay/_apply_exact_selection ownership inversion plus AST-backed TR14.CLI_MAIN_BACKEDGE and parity tests. cli.source, release.pin, and release.records are already canonical and must not be unnecessarily altered. PA34 and PA33 later touch the same CLI patch ownership, so serialize final integration and use separate worktrees for any parallel exploratory work.

Initial assessment and design to be supplied by GPT gateway review before implementation.

GPT gateway initial response: audited main at 644b8207d11eaafa14c34f29fe4dc6c95388911d; verdict directionally coherent but still overly bridged. Recommended order: (1) cli/__main__ ownership inversion; (2) reduce __main__ to compatibility shell; (3) enforce no cli -> __main__ back-edge; (4) consolidate smoke/report; (5) reconcile rocprof; (6) retire root facades last. Safe to remove now: none. Root facades are intentional module-identity aliases. The audit also noted BRC01 was not visible on the audited tip, which is why this plan must be committed and pushed before the second review.

GPT second-pass file-level design completed against published main tip 7f61b3b5615cc429efe6d630bffe4a46697d0e10. Key correction: cli/main.py and release/pin.py are already canonical; first tranche is limited to source/patch ownership inversion and __main__ compatibility reduction. No implementation started.

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. This is the first production-code slice after planning reconciliation because current cmd_apply still delegates through __main__. The live remainder is narrowly _copy_overlay/_restore_overlay/_apply_exact_selection ownership inversion plus AST-backed TR14.CLI_MAIN_BACKEDGE and parity tests. cli.source, release.pin, and release.records are already canonical and must not be unnecessarily altered. PA34 and PA33 later touch the same CLI patch ownership, so serialize final integration and use separate worktrees for any parallel exploratory work.

GPT implementation design response: request req_eccf031bbe2c42d5, same session ses_76206cac3e6b4be0, completed against exact pushed 66e9de865621dd5daf98f3482f81e9ab1c447544. It corrected the boundary to patch.overlay plus cli.patch orchestration, identified all four production __main__ consumers, specified canonical retargeting for patch.rebase/release.pin_bump/cli.tuning, required identity aliases only in __main__, and defined the AST back-edge checks and focused tests. The response is preserved in review RV176 and must be treated as design guidance, not implementation evidence.

Initial assessment and design to be supplied by GPT gateway review before implementation.

GPT gateway initial response: audited main at 644b8207d11eaafa14c34f29fe4dc6c95388911d; verdict directionally coherent but still overly bridged. Recommended order: (1) cli/__main__ ownership inversion; (2) reduce __main__ to compatibility shell; (3) enforce no cli -> __main__ back-edge; (4) consolidate smoke/report; (5) reconcile rocprof; (6) retire root facades last. Safe to remove now: none. Root facades are intentional module-identity aliases. The audit also noted BRC01 was not visible on the audited tip, which is why this plan must be committed and pushed before the second review.

GPT second-pass file-level design completed against published main tip 7f61b3b5615cc429efe6d630bffe4a46697d0e10. Key correction: cli/main.py and release/pin.py are already canonical; first tranche is limited to source/patch ownership inversion and __main__ compatibility reduction. No implementation started.

GPT roadmap provenance: request req_f7a013040828433c, same session ses_76206cac3e6b4be0, exact pushed bb20f104. This is the first production-code slice after planning reconciliation because current cmd_apply still delegates through __main__. The live remainder is narrowly _copy_overlay/_restore_overlay/_apply_exact_selection ownership inversion plus AST-backed TR14.CLI_MAIN_BACKEDGE and parity tests. cli.source, release.pin, and release.records are already canonical and must not be unnecessarily altered. PA34 and PA33 later touch the same CLI patch ownership, so serialize final integration and use separate worktrees for any parallel exploratory work.

GPT implementation design response: request req_eccf031bbe2c42d5, same session ses_76206cac3e6b4be0, completed against exact pushed 66e9de865621dd5daf98f3482f81e9ab1c447544. It corrected the boundary to patch.overlay plus cli.patch orchestration, identified all four production __main__ consumers, specified canonical retargeting for patch.rebase/release.pin_bump/cli.tuning, required identity aliases only in __main__, and defined the AST back-edge checks and focused tests. The response is preserved in review RV176 and must be treated as design guidance, not implementation evidence.

## Preserved historical GPT design (context only)

The prior design below is retained to preserve the detailed behavior, test, stream, rollback, identity, and compatibility constraints. Where it names source/release migration as future work, the current scope above supersedes that instruction; do not reimplement already-canonical owners.

Edit

1. Exact symbols/files

tools/bigcherry/cli/source.py



Refactor:



cmd_audit(args)

cmd_pull(args)



Remove both function-local:



from .. import __main__ as legacy



Dependencies currently obtained through legacy:



_record_for

_uncommitted_pin_change

paths

pin_transition

upstream

UPSTREAM_URL

_run

sys.stderr



Target ownership:



_record_for                  -> release.records.record_for_checkout

_uncommitted_pin_change      -> release.pin.uncommitted_pin_change

paths                        -> core.paths

pin_transition               -> bigcherry.pin_transition

upstream                     -> source.upstream

UPSTREAM_URL                 -> source.upstream.UPSTREAM_URL

_run                          -> private cli.source._run

sys.stderr                   -> sys.stderr

tools/bigcherry/cli/patch.py



Refactor:



cmd_apply(args)



Remove:



from .. import __main__ as legacy



Move from tools/bigcherry/__main__.py into this module, behavior-preserving:



_copy_overlay(...)

_restore_overlay(...)

_apply_exact_selection(...)



cmd_apply() changes only from:



legacy._apply_exact_selection(...)



to:



_apply_exact_selection(...)



Preserve current arguments, including:



allow_stale_validation_evidence

tools/bigcherry/release/records.py



Move/canonicalize current __main__ helpers:



_git_out(root: Path, *args: str) -> str

_bigcherry_revision() -> str

_record_for(root: Path)



Public canonical API:



def record_for_checkout(root: Path) -> ReleaseRecord:

    ...



Keep _git_out / _bigcherry_revision private.



tools/bigcherry/release/pin.py



Move current __main__:



_PIN_LINE

_uncommitted_pin_change()



Target:



_PIN_LINE

uncommitted_pin_change() -> str | None



Do not duplicate existing canonical:



resolve_pin_sha

cmd_repin

pin_status_paths

render_pin_status

cmd_pin_status

tools/bigcherry/__main__.py



Delete substantive implementations after consumers migrate:



_run

_git_out

_bigcherry_revision

_record_for

_PIN_LINE

_uncommitted_pin_change

_copy_overlay

_restore_overlay

_apply_exact_selection



_legacy_resolve_pin_sha

_legacy_cmd_repin

_legacy_pin_status_paths

_legacy_render_pin_status

_legacy_cmd_pin_status



Retain compatibility exports where currently supported:



cmd_audit

cmd_pull

cmd_apply

cmd_patch_*

cmd_generate

cmd_replay_inspect

cmd_inventory

cmd_build_new

cmd_experiment_*

build_parser

_legacy_main

cmd_repin

cmd_pin_status

_configure_output



Private compatibility aliases may remain if existing tests/consumers require them:



_record_for = release.records.record_for_checkout

_uncommitted_pin_change = release.pin.uncommitted_pin_change

_resolve_pin_sha = release.pin.resolve_pin_sha

_pin_status_paths = release.pin.pin_status_paths

_render_pin_status = release.pin.render_pin_status



No substantive source/patch/release behavior remains in __main__.



tools/bigcherry/check.py



Extend tooling_hygiene() with forbidden production back-edge detection.



New diagnostic:



TR14.CLI_MAIN_BACKEDGE

severity=error



Reject production imports of bigcherry.__main__ from any module except tools/bigcherry/__main__.py, including:



import bigcherry.__main__

from bigcherry import __main__

from .. import __main__

importlib.import_module("bigcherry.__main__")

__import__("bigcherry.__main__")



Use AST resolution; do not grep comments/docstrings.



2. Current/target call graph



Current:



python -m bigcherry

  -> bigcherry.__main__

     -> cli.main

        -> cli.source.cmd_audit

           -> __main__._record_for

        -> cli.source.cmd_pull

           -> __main__._uncommitted_pin_change

           -> __main__.upstream/_run/_record_for/...

        -> cli.patch.cmd_apply

           -> __main__._apply_exact_selection

              -> patch_admission.admit

              -> patch.apply.apply_all



Ownership cycle:



__main__ -> cli.source -> __main__

__main__ -> cli.patch  -> __main__



Target:



python -m bigcherry

  -> __main__

     -> cli.main

        -> cli.source

           -> core.paths

           -> source.upstream

           -> release.pin

           -> release.records

        -> cli.patch

           -> patch.selection

           -> patch_admission

           -> patch.patchset

           -> patch.rebase

           -> patch.apply

           -> release.records



Required invariant:



entrypoint -> CLI -> domain APIs



Never:



CLI/domain -> __main__

3. Target APIs / boundaries

Release-record identity



release.records.record_for_checkout(root) must be a verbatim semantic move:



def record_for_checkout(root: Path) -> ReleaseRecord:

    revision, _ = source_audit.git_revision(root)

    tag = _git_out(root, "describe", "--tags", "--exact-match")

    record = load(revision, tag)

    record.revision = revision

    record.release_tag = tag

    record.bigcherry_revision = _bigcherry_revision()

    return record



Do not change:



revision derivation

exact-tag behavior

fallback behavior

record lookup/path

bigcherry_revision

serialization

Pin guard



release.pin.uncommitted_pin_change() owns RE48 policy.



Move implementation unchanged apart from module-local references.



Source CLI



cli.source owns only command presentation/orchestration.



Private:



def _run(

    args: list[str],

    cwd: Path | None = None,

    *,

    check: bool = True,

) -> subprocess.CompletedProcess[str]:

    return subprocess.run(args, cwd=cwd, text=True, check=check)



Do not create a generic subprocess abstraction in this tranche.



Patch CLI



cli.patch owns direct CLI apply orchestration.



Canonical edit engine remains:



bigcherry.patch.apply.apply_all

bigcherry.patch.apply.format_results



Do not duplicate/reimplement anchored patch application.



_apply_exact_selection() remains CLI workflow logic because it combines:



CLI-selected source composition

live-checkout validation

admission policy

overlay installation

patch engine invocation

CLI rendering

release-record advancement

__main__



Compatibility entrypoint only.



Preferred end-state shape:



from .cli.main import (

    build_parser,

    cmd_pin_status,

    cmd_repin,

    main,

    _configure_output,

    _legacy_main,

)

from .cli.source import cmd_audit, cmd_pull

from .cli.patch import cmd_apply, ...

from .release import pin as _release_pin

from .release import records as _release_records



_record_for = _release_records.record_for_checkout

_uncommitted_pin_change = _release_pin.uncommitted_pin_change

_resolve_pin_sha = _release_pin.resolve_pin_sha

_pin_status_paths = _release_pin.pin_status_paths

_render_pin_status = _release_pin.render_pin_status



if __name__ == "__main__":

    raise SystemExit(main())



Static imports are preferable once cycles are gone; importlib.import_module() should no longer be needed merely to break these back-edges.



4. Ordered patch sequence



release/records.py



add record_for_checkout.



move helper internals.



add direct tests before changing consumers.



release/pin.py



add uncommitted_pin_change.



add direct unit coverage.



cli/source.py



direct imports.



add local _run.



switch audit/pull record creation to record_for_checkout.



remove every __main__ dependency.



Source tests



migrate monkeypatch targets from bigcherry.__main__.* to lookup-site/canonical owners.



preserve command-visible output/return behavior.



cli/patch.py



move _copy_overlay, _restore_overlay, _apply_exact_selection essentially verbatim.



add direct imports:



from .. import patch_admission

from ..release import records as releases

from ..patch import apply as patcher



preserve exact execution order.



Patch tests



relocate _apply_exact_selection behavior tests currently living in tools/tests/release/test_releases.py into tools/tests/cli/test_cli_patch_apply.py.



patch bigcherry.cli.patch, not bigcherry.__main__.



__main__.py



remove migrated bodies.



retain compatibility exports.



replace dynamic import workarounds with ordinary imports where possible.



check.py



add TR14.CLI_MAIN_BACKEDGE.



Core layering/facade tests



prove no production back-edge.



preserve root compatibility facade identity.



Run focused then full validation. No reset/stash/rebase and no mutation of unrelated worktrees.



5. Critical code guidance



Preserve _apply_exact_selection() sequencing exactly:



1. read live vendor HEAD

2. resolve selection.source_ref^{commit} in same checkout

3. reject live/expected revision mismatch

4. re-resolve exact source composition

5. reject patch-set / patch-ID composition drift

6. reject unresolved overlay flag

7. patch_admission.admit(...)

8. emit admission warnings to stderr

9. reject inadmissible patch set

10. load release record / enforce prior strict audit unless --force

11. simulate/write overlay

12. patchset.resolve_exact(...)

13. patchset.load_resolved(...)

14. patch.apply.apply_all(..., initial_texts=overlay_sim)

15. restore overlay on real patch failure

16. compute tree_state/mutation

17. render existing stdout

18. if !dry_run: reload/update/save ReleaseRecord



Keep admission call equivalent to current:



admission = patch_admission.admit(

    selection.patch_ids,

    mode="apply",

    pinned_ref=selection.source_ref,

    resolved_base_revision=live_revision,

    allow_stale_validation_evidence=allow_stale_validation_evidence,

)



Do not lose the current direct-apply stale-evidence override.



Keep canonical patch invocation:



results = patcher.apply_all(

    patches,

    root,

    dry_run=dry_run,

    initial_texts=overlay_sim,

)



Keep current mutation identity:



intended_tree_state = selection.tree_state_key(live_revision)

selection_changed = record.tree_state != intended_tree_state

tree_mutated = bool(written) or any(result.changed for result in results)



Keep release transition:



releases.record_apply_result(

    record,

    ok,

    mutated=selection_changed or tree_mutated,

)



For cmd_apply, preserve exit taxonomy:



argument/selection/config error = 2

apply/reconciliation failure    = 1

success                         = 0



Preserve output ordering:



apply-detail output

selection: ...

  RESULT: PASS|FAIL



For cmd_pull, preserve:



uncommitted-pin guard

uncommitted-marker guard

--ref > --source > current-head precedence

"latest" resolution

clone/fetch semantics

clear_stale_locks immediately before owned fetch

ensure_ref for shallow refs

checkout --force

ReleaseRecord advancement/save

6. Migration hazards

Monkeypatch lookup sites



Current tests patch:



bigcherry.__main__._uncommitted_pin_change

bigcherry.__main__.pin_transition.*

bigcherry.__main__.upstream.*

bigcherry.__main__._run

bigcherry.__main__._record_for



After refactor, patch the symbol where execution looks it up, e.g.:



bigcherry.cli.source._run

bigcherry.cli.source.upstream.*

bigcherry.cli.source.pin_transition.*

bigcherry.cli.source.pin_release.uncommitted_pin_change

bigcherry.cli.source.releases.record_for_checkout



Compatibility aliases in __main__ do not need to remain dependency-injection seams.



stdout/stderr



Do not convert command code to exceptions/result objects in this tranche. Existing text streams are observable CLI behavior.



Particularly preserve stderr for:



unknown source

pin-transition refusal

ref-resolution errors

patch admission warnings/errors

audit prerequisite refusal

apply argument incompatibility

Evidence/receipt identity



Do not modify:



ReleaseRecord schema

ReleaseRecord.save()

summarise_audit()

summarise_patches()

record_apply_result()

tree_state_key()

admission evidence keys

artifact paths

JSON formatting/order where contract-tested

revision/tag derivation



Moving _record_for must not change the resulting release JSON.



Overlay/patch rollback



patch.apply.apply_all() protects files it patches.



Overlay writes are a separate transaction and must retain _restore_overlay() behavior.



Do not merge the two rollback implementations as part of this ownership cleanup.



Shared worktree safety



Keep:



live HEAD validation

same-repository ref resolution

composition re-resolution immediately before mutation

admission before overlay writes

overlay=None fail-closed

patch apply two-pass validation

patch rollback

overlay rollback



There is no cross-process worktree lock in this path today. Do not claim otherwise and do not introduce one casually; locking requires separate ownership/order/deadlock design.



upstream.clear_stale_locks() is intentionally dangerous outside its narrow location. Keep it immediately before cmd_pull's own fetch.



Replay



This tranche must not alter replay/cache code.



Run replay regressions because release/provenance identities are inputs to later workflows, not because replay should be refactored.



Facades



Do not remove:



tools/bigcherry/patcher.py

tools/bigcherry/inventory.py

tools/bigcherry/replay_cache.py



patcher must remain the module-identity alias to bigcherry.patch.apply.



__main__ compatibility



Existing callers/tests may still do:



from bigcherry import __main__ as bigcherry_main

bigcherry_main.cmd_pull(...)

bigcherry_main.cmd_repin(...)



Those command names should continue working.



Private helpers should only remain aliases where current tests/consumers establish a real compatibility need.



7. Tests/files



Modify/add:



tools/tests/cli/test_cli_source_pull.py

tools/tests/cli/test_cli_patch_apply.py        # new/preferred

tools/tests/release/test_releases.py

tools/tests/release/test_pin_status.py

tools/tests/core/test_compatibility_facades.py

tools/tests/core/test_cli_layering.py          # new/preferred

tools/tests/core/test_cli_tooling_surface.py    # if existing ownership fits better



Required source tests:



source "pinned" ref resolves cfg.pinned

explicit source ref remains explicit

unknown source => rc 2/stderr

uncommitted pin refusal unchanged

uncommitted transition refusal unchanged

committed/absent transition proceeds

clone/fetch ordering unchanged

stale-lock clearing remains before fetch

record identity/advance unchanged

audit JSON/record identity unchanged



Required direct-apply tests:



live HEAD mismatch fails before mutation

unresolvable source ref fails

composition drift fails before mutation

overlay=None fails closed

admission reject occurs before overlay mutation

admission warning stream preserved

overlay=false does not call overlay copier

dry-run does not persist checkout/record mutation

patch failure restores overlay

successful apply records exact tree_state

reapply mutation invalidates later evidence as before

allow_stale_validation_evidence reaches admission unchanged

cmd_apply preserves rc/stdout/stderr for all argument modes

rebase-report apply behavior remains unchanged



Back-edge test must fail on static and dynamic forms.



8. Validation commands

PYTHONPATH=tools python -m pytest -q \

  tools/tests/cli/test_cli_source_pull.py \

  tools/tests/cli/test_cli_patch_apply.py \

  tools/tests/release/test_pin_status.py \

  tools/tests/release/test_releases.py \

  tools/tests/core/test_compatibility_facades.py \

  tools/tests/core/test_cli_layering.py



PYTHONPATH=tools python -m pytest -q tools/tests/patch



PYTHONPATH=tools python -m pytest -q \

  tools/tests/tuning/test_correctness_evidence.py \

  tools/tests/tuning/test_replay_cache_wire.py \

  tools/tests/tuning/test_replay_v5.py



PYTHONPATH=tools python -m bigcherry check --quick

PYTHONPATH=tools python -m compileall -q tools/bigcherry



rg -n 'bigcherry\.__main__|from .* import __main__|import_module\(.*__main__' \

  tools/bigcherry --glob '!__main__.py'



PYTHONPATH=tools python -m bigcherry --help

PYTHONPATH=tools python -m bigcherry apply --help

PYTHONPATH=tools python -m bigcherry pull --help

PYTHONPATH=tools python -m bigcherry audit --help



PYTHONPATH=tools python -m pytest -q tools/tests



For mutation-capable fixture tests, capture before/after:



git rev-parse HEAD

git branch --show-current

git status --porcelain=v1

git worktree list --porcelain



They must be identical outside explicitly disposable fixture trees.



9. Acceptance criteria



No production cli.* -> bigcherry.__main__ import, static or dynamic.



No canonical domain module imports __main__.



__main__.py is compatibility/entrypoint wiring, not source/patch implementation ownership.



cli.main remains parser/dispatch authority.



release.pin remains pin-policy authority.



release.records.record_for_checkout() reproduces existing _record_for identity exactly.



cli.source directly owns source CLI orchestration.



cli.patch directly owns apply CLI orchestration.



bigcherry.patch.apply remains the only anchored-edit application engine.



Current patch-admission, HEAD/ref, composition-TOCTOU, overlay and rollback safety is preserved.



CLI stdout/stderr/exit behavior does not regress.



Dry-run remains non-persistent.



Release/evidence/receipt/tree-state identities do not change solely because of the refactor.



Replay/cache regression tests remain unchanged/green.



patcher, inventory, replay_cache facade identity contracts remain green.



bigcherry.__main__.cmd_* compatibility calls continue working.



bigcherry check --quick fails on future __main__ back-edge reintroduction.



Full test suite introduces no new failures relative to current 7f61b3b5 baseline.



No implementation in this tranche touches smoke/report consolidation, either rocprof.py, or facade retirement.



Resolve the helper ownership ambiguity

Specify exact compatibility exports

## Change Log

- 2026-09-11T06:44:31.138556+00:00 (created-by): Created by agent

## Ledger-events






- chg_20260911_064900_added-a-tracked-cleanup-plan-f_3622
- 2026-09-11T06:49:00.213062+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T07:01:03.606121+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260911_070115_recorded-gpts-cleanup-assessm_2226
- 2026-09-11T07:01:15.274980+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T09:27:28.866422+00:00 (updated-by): Updated: section:detailed_solution, section:acceptance_criteria, section:notes
- chg_20260911_233928_recorded-gpts-complete-file-l_6915
- 2026-09-11T23:39:28.415008+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_134118_started-brc01-ownership-invers_2735
- 2026-09-12T13:41:18.149659+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T18:52:27.212820+00:00 (updated-by): Updated: section:notes
- chg_20260912_185547_recorded-the-gpt-guided-non-vu_6004
- 2026-09-12T18:55:48.021901+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T19:10:26.246972+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260912_191132_corrected-the-gpt-guided-imple_8803
- 2026-09-12T19:11:32.051269+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T19:12:31.678488+00:00 (updated-by): Updated: section:detailed_solution
- 2026-09-12T19:13:14.015128+00:00 (updated-by): Updated: section:detailed_solution
- 2026-09-12T19:16:48.726262+00:00 (updated-by): Updated (no visible changes)
- chg_20260912_191736_preserved-all-prior-brc01-gpt_4536
- 2026-09-12T19:17:36.432170+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T20:08:36.807486+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260912_200920_updated-brc01-with-gpts-imple_6565
- 2026-09-12T20:09:20.621985+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T20:09:51.936129+00:00 (updated-by): Updated: section:notes
- chg_20260912_201025_cleaned-the-brc01-provenance-n_3663
- 2026-09-12T20:10:25.428513+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_202702_removed-legacy-__main__-bridgi_9733
- 2026-09-12T20:27:02.015316+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_204955_closed-the-gpt-identified-vali_7517
- 2026-09-12T20:49:55.016283+00:00 (updated-by): Updated: section:ledger-events
- chg_20260912_205819_closed-the-final-gpt-found-ana_1131
- 2026-09-12T20:58:19.235650+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T21:10:36.908253+00:00 (state-transition): State: pending → in_progress
- chg_20260912_211106_final-gpt-review-passes-brc01_3408
- 2026-09-12T21:11:06.771300+00:00 (updated-by): Updated: section:ledger-events
