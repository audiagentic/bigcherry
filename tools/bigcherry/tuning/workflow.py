"""HI130: the full record -> tune -> correctness-evidence -> promote ->
replay pipeline as one repeatable, deterministic call, so it can be kicked
off as a standard bigcherry workflow instead of an operator/agent manually
chaining ~12 commands and working around gotchas live (validated by hand
on real Brutus GPU hardware this session -- dual-XTX + single-GPU R9700,
Qwen3.8-27B-Q8_0 MTP, 20/39 candidates genuinely promoted with real
correctness evidence).

Calls existing campaign/tuning APIs directly -- never shells out to the
``bigcherry`` CLI internally. Every stage that touches a GPU launches a
real server via server_runner.ServerRunner and drives it with a real
request; there is no mock/simulation mode.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import behavioral_corpus as behavioral_corpus_mod
from . import behavioral_gate as behavioral_gate_mod
from . import catalog as catalog_mod
from . import inventory as inv_mod
from . import recovery as recovery_mod
from . import replay as replay_mod
from . import signature_digest_verification as sdv
from . import tune_promotion
from .server_runner import ServerError, ServerRunner
from .. import hi80_generate_correctness_evidence as hi80
from ..campaign import planner as campaign_planner
from ..build import generated_tree
from ..campaign.lane import CampaignLaneResult
from ..core import config as campaign_config
from ..core import gpu as gpu_mod
from ..core import paths
from ..core.artifacts import ArtifactStore
from ..core.context import ProjectContext
from ..source.identity import atomic_write_json


class TuneCampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageIdentity:
    run_id: str
    source_slice_id: str
    build_plan_id: str
    manifest_path: str | None
    source_root: str


@dataclass(frozen=True)
class WorkflowReceipt:
    schema_version: int
    campaign_run_id: str
    model_path: str
    platform_name: str
    devices: str
    runtime_profile_name: str
    effective_tune_context: int
    record: StageIdentity
    tune: StageIdentity
    #: HI125 close-out step 6: the dedicated RECORD=ON + hi105-correctness
    #: test-backend-ops lane that authorized strengthened winner_verification
    #: attestation for this run's ingest -- kept auditable here rather than
    #: discarded, since it is now a trust-bearing lane like the others.
    signature_verifier: StageIdentity
    correctness: StageIdentity | None
    replay: StageIdentity | None
    promoted_before_evidence: int
    promoted_after_evidence: int
    verified_winners: int
    quarantined_unsupported_winners: int
    dispatch_cache_path: str | None
    replay_coverage: dict | None
    started_at: str
    finished_at: str
    replay_validation: StageIdentity | None = None


def _stage_identity(result: CampaignLaneResult) -> StageIdentity:
    return StageIdentity(
        # gpt review (2026-08-27): run_campaign()'s per-lane run_id
        # (_lane_run_id(campaign_run_id, lane)) is what actually got used
        # for this lane's ArtifactStore paths/provenance -- reconstructing
        # a different string here would make the receipt's run_id
        # unusable for re-deriving those paths even though the build
        # itself succeeded.
        run_id=result.run_id,
        source_slice_id=result.source_slice_id,
        build_plan_id=result.build_plan_id,
        manifest_path=str(result.manifest_ref.path) if result.manifest_ref else None,
        source_root=str(result.source_root),
    )


def _devices_tuple(devices: str) -> tuple[int, ...]:
    return tuple(int(d) for d in devices.split(","))


def _plan_and_run_one_lane(
    *, context: ProjectContext, cfg: campaign_config.Config, store: ArtifactStore,
    source_name: str, build_name: str, platform_name: str, run_id: str,
    binary_relative_path: str,
    inputs_by_build: dict[str, tuple[tuple[str, object], ...]] | None = None,
    experiment: str | None = None,
) -> CampaignLaneResult:
    request = campaign_planner.CampaignRequest(
        selectors=(campaign_config.CampaignLaneSelector(
            source=source_name, build=build_name, platform=platform_name,
        ),),
        inputs_by_build=inputs_by_build or {},
        binary_relative_path=binary_relative_path,
        experiment=experiment,
    )
    lanes = campaign_planner.plan(request, cfg)
    results = campaign_planner.run_campaign(
        lanes, cfg=cfg, context=context, store=store, run_id=run_id,
    )
    (result,) = results.values()
    if isinstance(result, Exception):
        raise TuneCampaignError(
            f"{source_name}:{build_name}:{platform_name} build failed: {result}"
        ) from result
    return result


def _stage_record(
    *, context, cfg, store, run_id, platform_name, source_name,
    model_path: Path, devices: str, runtime_profile: campaign_config.RuntimeProfile,
    workdir: Path,
) -> tuple[CampaignLaneResult, Path]:
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="record", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
    )
    binary_path = lane_result.binary_ref.path
    record_db_path = workdir / "record"
    runner = ServerRunner(
        binary=binary_path, model=model_path,
        extra_args=("-ngl", "99", "-c", str(runtime_profile.production_context), *runtime_profile.server_args),
        env_overrides={
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "record",
            "GGML_HIP_DISPATCH_DB": str(record_db_path),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
        },
        log_path=workdir / "record-server.log",
    )
    with runner:
        runner.run_completion("Describe the water cycle in two sentences.", n_predict=96)
    actual_record_path = record_db_path  # the binary writes this exact path, no suffix
    if not actual_record_path.is_file():
        raise TuneCampaignError(f"record stage produced no output at {actual_record_path}")
    return lane_result, actual_record_path


def _stage_inventory_record(*, record_db_path: Path, workdir: Path) -> tuple[Path, Path]:
    record = inv_mod.read_jsonl(record_db_path)
    inventory = inv_mod.build_inventory(record)
    inventory_path = workdir / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="")
    inventory_db_path = workdir / "inventory.sqlite"
    inv_mod.build_database(record, inventory_db_path, paths.SQL / "dispatch-db.sql")
    return inventory_path, inventory_db_path


def _stage_tune(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, model_path: Path, devices: str,
    runtime_profile: campaign_config.RuntimeProfile,
    screen_samples: int, final_samples: int, workdir: Path,
) -> tuple[CampaignLaneResult, Path]:
    gpu_mod.preflight_context(profile=runtime_profile, devices=_devices_tuple(devices), stage="tune")
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="tune", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
        inputs_by_build={"tune": (("inventory", inventory_path),)},
    )
    binary_path = lane_result.binary_ref.path
    tune_db_path = workdir / "tune"
    runner = ServerRunner(
        binary=binary_path, model=model_path,
        extra_args=("-ngl", "99", "-c", str(runtime_profile.tune_context), *runtime_profile.server_args),
        env_overrides={
            "HIP_VISIBLE_DEVICES": devices,
            "GGML_HIP_DISPATCH_MODE": "tune",
            "GGML_HIP_DISPATCH_DB": str(tune_db_path),
            "GGML_HIP_TUNE_SCREEN_SAMPLES": str(screen_samples),
            "GGML_HIP_TUNE_FINAL_SAMPLES": str(final_samples),
            "GGML_CUDA_DISABLE_GRAPHS": "1",
        },
        log_path=workdir / "tune-server.log",
    )
    with runner:
        runner.run_completion("Write a short paragraph about the ocean.", n_predict=96)
    measurements_path = Path(f"{tune_db_path}.measurements.jsonl")
    if not measurements_path.is_file():
        raise TuneCampaignError(f"tune stage produced no measurements at {measurements_path}")
    return lane_result, measurements_path


def _gpu_scoped_test_backend_ops_runner(devices: str):
    """A subprocess runner for signature_digest_verification's real
    test-backend-ops calls, scoped to this campaign's own GPU selection --
    correctness_evidence.run_test_backend_ops() passes an explicit ``env``
    dict straight to subprocess.run(), which REPLACES the ambient
    environment entirely rather than merging with it, so without this the
    verifier subprocess would not inherit HIP_VISIBLE_DEVICES the way
    ServerRunner's own env_overrides do for the record/tune stages."""

    def _runner(argv, **kwargs):
        env = dict(kwargs.get("env") or {})
        # Assignment, not setdefault: the verifier's device scoping is a
        # promise this runner exists to make, not a fallback -- a future
        # caller supplying its own (wrong) HIP_VISIBLE_DEVICES in env=
        # must not be able to silently defeat it.
        env["HIP_VISIBLE_DEVICES"] = devices
        kwargs["env"] = env
        return subprocess.run(argv, **kwargs)

    return _runner


def _stage_signature_verifier(
    *, context, cfg, store, run_id, platform_name, source_name, devices: str, seed: int = 1,
) -> tuple[CampaignLaneResult, Callable[[dict[str, Any]], str]]:
    """HI125 close-out step 6: a DEDICATED lane building a RECORD=ON
    test-backend-ops under the hi105-correctness experiment (which carries
    the additional patches signature_digest_verification's MUL_MAT_ID and
    routed-GLU mappers need -- a plain record-lane test-backend-ops is
    insufficient for HI121's full audited domain).

    Deliberately NOT built by adding extra_cmake_targets to the real
    record lane (build_name="record", binary_relative_path="bin/llama-
    server"): campaign/lane.py's _execute_build_phase() folds every
    requested cmake target -- including any extras -- into
    BuildPlan.requested_targets, which is part of that lane's own declared
    build identity, and the worker also includes extra binaries in the
    runtime artifact/bundle hash. Adding test-backend-ops there would
    change the REAL production record lane's build_plan_id/runtime bundle
    identity for a reason that has nothing to do with what that lane
    actually produces for the campaign. A separate lane keeps the trust-
    bearing verifier binary's own identity auditable (see
    WorkflowReceipt.signature_verifier) without perturbing production
    record-lane provenance at all.

    ``devices`` is the campaign's full device selection (e.g. "0,1"), but
    the verifier is scoped to just its FIRST device (adversarial-review
    follow-up, 2026-08-27): test-backend-ops loops over every visible
    backend device and requires aggregate success across all of them, so
    passing the whole multi-device string would run every unique canonical
    on EVERY selected GPU (wasted duplicate work) and could abort
    verification entirely on an unrelated failure on a second/heterogeneous
    device -- HI125's verifier only needs to prove canonical->C++ digest
    correspondence once, on any one real device, not hardware equivalence
    across the whole campaign's device set.
    """
    verifier_device = str(_devices_tuple(devices)[0])
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="record", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/test-backend-ops",
        experiment="hi105-correctness",
    )
    verifier = sdv.make_signature_digest_verifier(
        binary=lane_result.binary_ref.path,
        vendor_root=lane_result.source_root,
        seed=seed,
        runner=_gpu_scoped_test_backend_ops_runner(verifier_device),
    )
    return lane_result, verifier


def _count_missing_correctness_evidence(promoted_path: Path) -> int:
    """How many rows are stuck on `rejected_no_correctness_evidence` --
    gpt review (2026-08-27): `promote()`'s own return dict has no such
    count, and `promoted == 0` is the wrong trigger for re-running the
    correctness-evidence stage -- a dispatch_db that already had SOME
    reusable evidence from a prior run can promote a few candidates
    immediately while leaving others stuck, and that partial case must
    still trigger evidence generation for the rest, not be silently
    skipped because promoted > 0."""
    count = 0
    with promoted_path.open(encoding="utf-8") as f:
        next(f, None)  # header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("promotion_status") == "rejected_no_correctness_evidence":
                count += 1
    return count


def _stage_load_and_promote(
    *, tune_measurements: Path, tune_manifest_path: Path, workdir: Path,
    q: float, threshold_pct: float, resamples: int,
    signature_digest_verifier: Callable[[dict[str, Any]], str],
) -> tuple[Path, dict, int, int]:
    """Ingest + promote for the real campaign path -- ``signature_digest_
    verifier`` is a REQUIRED keyword (no default) so a future third call
    site here cannot silently revert to unverified ingestion by omission;
    inv_mod.load_measurements() itself still defaults it to None for the
    offline/manual-CLI path, which stays unaffected."""
    dispatch_db = workdir / "tune.sqlite"
    try:
        ingest_counts = inv_mod.load_measurements(
            tune_measurements, dispatch_db, paths.SQL / "dispatch-db.sql",
            manifest_path=tune_manifest_path,
            signature_digest_verifier=signature_digest_verifier,
            unsupported_signature_policy="quarantine",
            # adversarial-review follow-up (2026-08-27): without this, a
            # missing/unreadable tune_manifest_path or an older header
            # predating producer_capabilities would silently commit an
            # UNATTESTED load even though a verifier was supplied --
            # exactly the "believes it's strengthened but isn't" gap this
            # whole wiring pass exists to close.
            require_strengthened_ingest=True,
        )
        if "winner_verifications" not in ingest_counts:
            raise TuneCampaignError(
                "strengthened ingest did not report verified winner counts"
            )
        if int(ingest_counts["winner_verifications"]) <= 0:
            raise TuneCampaignError(
                "strengthened ingest produced zero verified winners"
            )
    except inv_mod.RecordError as exc:
        # Fail closed, never silently retry unverified (HI125 close-out
        # step 6's explicit policy) -- an UnsupportedSignatureDomain or a
        # genuine verification failure aborts the whole campaign rather
        # than degrading to un-attested ingestion an operator would not
        # know to distrust.
        raise TuneCampaignError(f"strengthened ingest failed: {exc}") from exc
    promoted_path = workdir / "promoted.jsonl"
    result = tune_promotion.promote(
        tune_measurements, promoted_path,
        dispatch_db=dispatch_db, q=q, threshold_pct=threshold_pct, resamples=resamples,
    )
    # Promotion finalizes the winner identity (including challenger -> native
    # fallback). Re-ingest that finalized artifact through the strengthened
    # path so the dispatch DB remains the authoritative source for export.
    try:
        final_ingest_counts = inv_mod.load_measurements(
            promoted_path, dispatch_db, paths.SQL / "dispatch-db.sql",
            manifest_path=tune_manifest_path,
            signature_digest_verifier=signature_digest_verifier,
            unsupported_signature_policy="quarantine",
            require_strengthened_ingest=True,
        )
    except inv_mod.RecordError as exc:
        raise TuneCampaignError(f"strengthened finalized ingest failed: {exc}") from exc
    result["winner_verifications"] = int(final_ingest_counts.get("winner_verifications", 0))
    result["quarantined_unsupported_winners"] = int(
        final_ingest_counts.get("quarantined_unsupported_winners", 0)
    )
    missing_evidence = _count_missing_correctness_evidence(promoted_path)
    return dispatch_db, result, int(result.get("promoted", 0)), missing_evidence


def _stage_correctness_evidence(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, tune_measurements: Path, dispatch_db: Path,
    seeds: tuple[int, ...],
) -> CampaignLaneResult:
    lane_result = _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name="tune", platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/test-backend-ops",
        inputs_by_build={"tune": (("inventory", inventory_path),)},
        experiment="hi105-correctness",
    )
    _header, results = hi80._read_measurements(tune_measurements)
    rows = hi80.find_candidate_rows(results)
    import sqlite3
    conn = sqlite3.connect(str(dispatch_db))
    try:
        for row in rows:
            try:
                hi80.generate_for_row(
                    conn, row,
                    binary=lane_result.binary_ref.path,
                    vendor_root=lane_result.source_root,
                    seeds=seeds,
                    headroom_fraction=hi80.ce.DEFAULT_HEADROOM_FRACTION,
                    contract_version=hi80.ce.CONTRACT_VERSION,
                    tool_version="hi130-tune-campaign-v1",
                )
            except hi80.scm.SignatureMappingError:
                continue  # unsupported signature domain -- same as the CLI's own honest skip
    finally:
        conn.close()
    return lane_result


def _stage_replay_export(
    *, promoted_path: Path, target_manifest_path: Path, target_source_root: Path,
    dispatch_db: Path, workdir: Path,
) -> Path:
    """Export the dispatch cache against the REPLAY build's own manifest (not
    tune's) -- gpt review (2026-08-27, req_ec659ded425c4335): replay.build()
    already rebinds every winner to whatever manifest it's given (tune and
    replay legitimately have different manifest_hash by design, since
    variant_set is baked into the hash), so the cache must be stamped with
    the manifest of whichever binary will actually load it, or every entry
    reads as stale/RERUN_REQUIRED against that binary's own compiled hash."""
    ggml_h = target_source_root / "ggml" / "include" / "ggml.h"
    cache_bytes = replay_mod.build(
        promoted_path, target_manifest_path, ggml_h, dispatch_db=dispatch_db,
        require_winner_verification=True,
    )
    # HI143 (gpt review, 2026-08-29): write to a provisional path and only
    # atomically rename to the real dispatch.cache once the behavioral
    # gate + coverage check both pass (_stage_replay_validate). Writing
    # straight to dispatch.cache here meant a campaign that later failed
    # validation still left a plausible-looking, unvalidated cache file
    # sitting in the workdir.
    cache_path = workdir / "dispatch.cache.provisional"
    cache_path.write_bytes(cache_bytes)
    return cache_path


def _stage_replay_build(
    *, context, cfg, store, run_id, platform_name, source_name,
    inventory_path: Path, winners_path: Path,
    build_name: str = "replay",
) -> CampaignLaneResult:
    if build_name not in ("replay", "replay-diagnostic"):
        raise TuneCampaignError("replay stage requires replay or replay-diagnostic build")
    return _plan_and_run_one_lane(
        context=context, cfg=cfg, store=store, source_name=source_name,
        build_name=build_name, platform_name=platform_name, run_id=run_id,
        binary_relative_path="bin/llama-server",
        inputs_by_build={
            build_name: (("inventory", inventory_path), ("promoted-winners", winners_path)),
        },
    )


def _verify_replay_companion(production: CampaignLaneResult, diagnostic: CampaignLaneResult) -> None:
    """Fail closed before validating a production cache with another binary.

    Use existing source/catalog/generated-input identities, not a second
    compatibility scheme. The differing diagnostics flags are deliberate;
    generated registry and candidate compile inputs must remain identical.
    """
    if production.source_slice_id != diagnostic.source_slice_id:
        raise TuneCampaignError("replay companion source composition differs")
    manifests = []
    generated_inputs = []
    common_options = []
    for result, require_diagnostics in ((production, False), (diagnostic, True)):
        options = dict(result.build_plan.cmake_options)
        common_options.append({key: value for key, value in options.items() if key not in {
            "GGML_HIP_DISPATCH_DIAGNOSTICS", "GGML_HIP_REPLAY_DIAGNOSTICS",
        }})
        enabled = lambda name: str(options.get(name, "OFF")).upper() in ("ON", "TRUE", "1", "YES")
        if (not enabled("GGML_HIP_DISPATCH_REPLAY") or enabled("GGML_HIP_AUTOTUNE")
                or enabled("GGML_HIP_AUTOTUNE_RECORD")
                or enabled("GGML_HIP_DISPATCH_DIAGNOSTICS") != require_diagnostics
                or enabled("GGML_HIP_REPLAY_DIAGNOSTICS") != require_diagnostics):
            raise TuneCampaignError("replay companion build roles/configuration are invalid")
        if result.manifest_ref is None or result.generated_tree_ref is None:
            raise TuneCampaignError("replay companion lacks manifest/generated-tree evidence")
        try:
            manifest = json.loads(result.manifest_ref.path.read_text(encoding="utf-8"))
            if manifest["manifest_hash"] != catalog_mod.manifest_hash(manifest):
                raise TuneCampaignError("replay companion manifest hash does not recompute")
            descriptor = catalog_mod.build_descriptor(manifest)
            if descriptor != manifest["build_descriptor"]:
                raise TuneCampaignError("replay companion descriptor does not recompute")
            manifests.append(descriptor)
            tree = json.loads(result.generated_tree_ref.path.read_text(encoding="utf-8"))
            compiled_inputs_digest = generated_tree.compile_inputs_digest(tree)
            bundle_bytes = result.runtime_bundle_ref.path.read_bytes()
            if ArtifactStore.digest(bundle_bytes) != result.runtime_bundle_ref.content_hash:
                raise TuneCampaignError("replay companion runtime bundle evidence hash mismatch")
            bundle = json.loads(bundle_bytes)
            if (bundle.get("generated_inputs_verification") != "compiled-copy-v1"
                    or bundle.get("generated_compile_inputs_hash") != compiled_inputs_digest):
                raise TuneCampaignError("replay companion lacks matching build-bound compiled-input evidence")
            inputs = {name: tree["files"][name] for name in tree["compile_inputs"]}
            if not inputs or "hip-autotune-registry.inc" not in inputs:
                raise TuneCampaignError("replay companion lacks generated registry evidence")
            generated_inputs.append(inputs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TuneCampaignError(f"invalid replay companion evidence: {exc}") from exc
    if common_options[0] != common_options[1]:
        raise TuneCampaignError("replay companion non-diagnostic requested CMake options differ")
    if manifests[0] != manifests[1] or generated_inputs[0] != generated_inputs[1]:
        raise TuneCampaignError("replay companion catalog/registry compile inputs differ")


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _resolve_default_corpus(
    runtime_profile: campaign_config.RuntimeProfile,
) -> tuple[list[behavioral_gate_mod.BehavioralVector], behavioral_corpus_mod.CorpusEdition | None, tuple[behavioral_corpus_mod.CorpusVectorSpec, ...]]:
    """HTR03: resolve the applicable corpus for this runtime profile via
    its own configured edition reference (RuntimeProfile.
    behavioral_corpus_edition) + explicit behavioral_classes -- replacing
    the old hardcoded _default_behavioral_corpus() + implicit '--spec-type'
    string-matching, AND (GPT review round 2, 2026-08-30) replacing a
    hardcoded module-level manifest-path constant that would have required
    a code change to publish a new edition, defeating HTR03's actual goal.

    A profile with no behavioral_corpus_edition AND no behavioral_classes
    legitimately has nothing decision-sensitive to check (generalizes the
    old require_mtp=False case). A profile declaring behavioral_classes
    WITHOUT a corpus edition is a real config error -- fail closed rather
    than silently checking nothing against a class the operator explicitly
    asked for."""
    if runtime_profile.behavioral_corpus_edition is None:
        if runtime_profile.behavioral_classes:
            raise TuneCampaignError(
                f"runtime profile {runtime_profile.name!r} declares "
                f"behavioral_classes {list(runtime_profile.behavioral_classes)!r} "
                f"but no behavioral_corpus_edition -- nothing to resolve them against"
            )
        return [], None, ()
    manifest_path = behavioral_corpus_mod.resolve_manifest_path(
        runtime_profile.behavioral_corpus_edition, _FIXTURES_DIR
    )
    edition = behavioral_corpus_mod.load_corpus_edition(manifest_path, _FIXTURES_DIR)
    specs = behavioral_corpus_mod.resolve_applicable_vectors(edition, runtime_profile.behavioral_classes)
    vectors = [behavioral_corpus_mod.to_behavioral_vector(spec, _FIXTURES_DIR) for spec in specs]
    return vectors, edition, specs


def _load_signature_assignments(promoted_path: Path) -> dict[str, recovery_mod.SignatureAssignment]:
    """HTR01: build the recovery search's starting point directly from this
    campaign's OWN promoted.jsonl -- every non-native promoted signature's
    current winner, plus its already-measured alternatives (from its own
    ranking_decisions, best-first by effective_us) so recovery never needs
    a new GPU timing measurement to try a different candidate.

    KNOWN LIMITATION (not yet closed): alternatives are ordered purely by
    this campaign's own ranking_decisions effective_us; whether each
    alternative already has correctness evidence is checked lazily by
    AssignmentExecutor.ensure_correctness_evidence() when a proposal
    actually tries to use it (HTR01's real-hardware validation, 2026-08-30,
    confirmed this is the common case, not an edge case -- see HTR01's
    plan-item notes)."""
    assignments: dict[str, recovery_mod.SignatureAssignment] = {}
    with promoted_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("promotion_status") != "promoted":
                continue
            dispatch = row.get("dispatch")
            signature = row.get("signature")
            current = row.get("winner")
            # row["native"] is ALREADY the real native candidate stable_name
            # (e.g. "mmvq:native:v1", confirmed via real campaign data,
            # 2026-08-30) -- the same field promotion_gate.resolve_
            # promotion_identity() itself resolves via native_name -- not
            # the literal string "native". No derivation/regex-matching
            # against ranking_decisions is needed (an earlier version of
            # this function did that unnecessarily before this was found).
            native_candidate = row.get("native")
            if not dispatch or not signature or not current or not native_candidate:
                continue
            ranking = row.get("ranking_decisions") or []
            names_by_us: list[tuple[float, str]] = []
            for decision in ranking:
                for entry in decision.get("candidates", []):
                    name = entry.get("name")
                    if not name or name == current or name == native_candidate:
                        continue
                    effective_us = entry.get("effective_us")
                    if effective_us is None:
                        continue
                    names_by_us.append((float(effective_us), name))
            names_by_us.sort(key=lambda pair: pair[0])
            seen: set[str] = set()
            alternatives = []
            for _, name in names_by_us:
                if name in seen:
                    continue
                seen.add(name)
                alternatives.append(name)
            assignments[dispatch] = recovery_mod.SignatureAssignment(
                dispatch=dispatch, signature=signature, current_candidate=current,
                native_candidate=native_candidate, alternatives=tuple(alternatives),
            )
    return assignments


def _stage_replay_validate(
    *, lane_result: CampaignLaneResult, model_path: Path, devices: str,
    runtime_profile: campaign_config.RuntimeProfile, provisional_cache: Path, workdir: Path,
    corpus: list[behavioral_gate_mod.BehavioralVector] | None = None,
    promoted_path: Path | None = None, manifest_path: Path | None = None,
    ggml_h_path: Path | None = None, dispatch_db: Path | None = None, allow_recovery: bool = True,
    correctness_binary_path: Path | None = None, correctness_vendor_root: Path | None = None,
    correctness_seeds: tuple[int, ...] = (1, 2, 3), campaign_run_id: str | None = None,
    max_recovery_evaluations: int = recovery_mod.DEFAULT_MAX_RECOVERY_EVALUATIONS,
    max_new_correctness_candidates: int = recovery_mod.DEFAULT_MAX_NEW_CORRECTNESS_CANDIDATES,
) -> dict:
    """HI143: the real pre-promotion behavioral regression gate, wired into
    the actual campaign path (gpt-negotiated integration, 2026-08-29,
    session ses_330ae3c055084f38) -- replaces the old _stage_replay_verify,
    which only checked cache coverage (dispatched==executed, no stale/
    rerun entries) and would have shipped HI141's real regression without
    ever noticing it, since a stale/miss check says nothing about whether
    the CONTENT of what a hit dispatches to is correct.

    Runs the SAME replay-capable binary in native mode first (one real
    server load), then in replay mode against the provisional cache (a
    second real server load, which also carries the existing coverage
    check) -- never two 27B servers concurrently. Every corpus vector is
    compared native-vs-candidate via behavioral_gate.compare_traces; any
    hard_fail (generated output diverged) or behavior_changed (same
    output, different MTP accept trace -- requires a throughput
    adjudication step this module does not yet implement) fails the whole
    campaign closed. Only when every vector is exact_pass AND the existing
    replay-coverage check passes does the provisional cache get atomically
    renamed to the real dispatch.cache."""
    corpus_edition: behavioral_corpus_mod.CorpusEdition | None = None
    corpus_vector_specs: tuple[behavioral_corpus_mod.CorpusVectorSpec, ...] = ()
    if corpus is None:
        corpus_list, corpus_edition, corpus_vector_specs = _resolve_default_corpus(runtime_profile)
        corpus = tuple(corpus_list)
        # A runtime profile declaring NO behavioral_classes has nothing
        # decision-sensitive to check (the direct generalization of the
        # old require_mtp=False path) -- this is a legitimate, silent-by-
        # design skip, NOT the "corpus is empty" failure below, which
        # exists for an EXPLICITLY-supplied empty corpus (a caller bug).
        if not corpus and not runtime_profile.behavioral_classes:
            corpus = ()
        elif not corpus:
            raise TuneCampaignError(
                "behavioral gate corpus resolved to zero vectors for a runtime "
                "profile that DOES declare behavioral_classes -- this should "
                "already have failed loud inside resolve_applicable_vectors()"
            )
    else:
        corpus = tuple(corpus)
        if not corpus:
            raise TuneCampaignError("behavioral gate corpus is empty -- refusing to export unvalidated")
    binary_path = lane_result.binary_ref.path
    coverage_path = workdir / "coverage.json"
    report_path = workdir / "behavioral-gate.json"
    final_cache_path = workdir / "dispatch.cache"
    # Same stale-artifact defense as the old _stage_replay_verify (GPT
    # deep-review P2, 2026-08-29), extended to the FINALIZED cache itself
    # (gpt review, 2026-08-29): a reused workdir/run-id must never let a
    # prior run's coverage/report/dispatch.cache be read/seen as this
    # run's result -- an old finalized cache surviving a failed rerun
    # would otherwise remain visible even though THIS validation rejected
    # its provisional replacement.
    for stale in (coverage_path, report_path, final_cache_path):
        if stale.exists():
            stale.unlink()

    common_args = (
        "-ngl", "99", "-c", str(runtime_profile.production_context), *runtime_profile.server_args,
    )
    # HI143 (gpt review, 2026-08-29): the launched process otherwise
    # inherits the full ambient environment -- a leftover
    # GGML_HIP_FORCE_CANDIDATE(_STRICT)/DISPATCH_DB/DISPATCH_CACHE/
    # DISPATCH_COVERAGE from an unrelated earlier command could silently
    # contaminate either leg. Strip them; each leg then sets only what it
    # actually intends.
    env_unset = (
        "GGML_HIP_FORCE_CANDIDATE", "GGML_HIP_FORCE_CANDIDATE_STRICT",
        "GGML_HIP_DISPATCH_DB", "GGML_HIP_DISPATCH_CACHE", "GGML_HIP_DISPATCH_COVERAGE",
    )

    def _run_leg(*, dispatch_mode: str, log_name: str, extra_env: dict[str, str]) -> list[behavioral_gate_mod.BehavioralTrace]:
        env = {"HIP_VISIBLE_DEVICES": devices, "GGML_HIP_DISPATCH_MODE": dispatch_mode, **extra_env}
        runner = ServerRunner(
            binary=binary_path, model=model_path, extra_args=common_args,
            env_overrides=env, env_unset=env_unset, log_path=workdir / log_name,
        )
        try:
            with runner:
                return [
                    behavioral_gate_mod.run_vector(runner, vector, require_mtp=vector.requires_mtp)
                    for vector in corpus
                ]
        except (behavioral_gate_mod.BehavioralGateError, ServerError) as exc:
            raise TuneCampaignError(f"behavioral gate: {exc}") from exc

    native_traces = _run_leg(dispatch_mode="native", log_name="behavioral-native.log", extra_env={})

    candidate_env = {
        "GGML_HIP_DISPATCH_CACHE": str(provisional_cache),
        "GGML_HIP_DISPATCH_COVERAGE": str(coverage_path),
    }
    candidate_runner = ServerRunner(
        binary=binary_path, model=model_path, extra_args=common_args,
        env_overrides={"HIP_VISIBLE_DEVICES": devices, "GGML_HIP_DISPATCH_MODE": "replay", **candidate_env},
        env_unset=env_unset, log_path=workdir / "behavioral-candidate.log",
    )
    try:
        with candidate_runner:
            candidate_traces = [
                behavioral_gate_mod.run_vector(candidate_runner, vector, require_mtp=vector.requires_mtp)
                for vector in corpus
            ]
            candidate_runner.run_completion("Explain how a compass works.", n_predict=96)
    except (behavioral_gate_mod.BehavioralGateError, ServerError) as exc:
        raise TuneCampaignError(f"behavioral gate: {exc}") from exc

    # strict=True (gpt review, 2026-08-29): corpus is a materialized tuple
    # (not a one-shot iterator) so this can never silently truncate, but
    # strict=True is kept as a hard guarantee against a future refactor
    # reintroducing that class of bug -- a length mismatch here must be a
    # loud error, not a quietly short zip producing zero verdicts for the
    # missing vectors (which would let replay coverage pass unchallenged).
    report = behavioral_gate_mod.BehavioralGateReport(verdicts=[
        behavioral_gate_mod.compare_traces(vector.name, native, candidate)
        for vector, native, candidate in zip(corpus, native_traces, candidate_traces, strict=True)
    ])
    report_document = report.summary()
    # HTR03: persist exactly which corpus edition/vectors were checked --
    # so a FUTURE deep-dive reviewer (per real HTR01 experience: this is
    # not hypothetical, it's exactly what caught three real bugs) can
    # reconstruct precisely what this run validated without re-deriving it
    # from whatever the manifest/parser looks like AT REVIEW TIME.
    if corpus_edition is not None:
        report_document["corpus_edition_id"] = corpus_edition.edition
        report_document["corpus_schema_version"] = corpus_edition.schema_version
        report_document["corpus_content_digest"] = corpus_edition.content_digest
        # GPT review round 2 (2026-08-30): the promised per-vector snapshot
        # also includes the ACTUAL comparison result (verdict, draft
        # traces, first_output_divergence, exact token digests) -- not
        # just the manifest's own static parameters -- so a future
        # reviewer never needs to re-derive "what happened" from the
        # manifest/parser as it exists at review time. verdicts_by_name is
        # keyed by vector name (== spec.id) rather than assuming zip order,
        # since corpus_vector_specs and report.verdicts are populated via
        # independent code paths that happen to share ordering today but
        # need not forever.
        verdicts_by_name = {v.vector_name: v for v in report.verdicts}
        selected_vectors = []
        for spec in corpus_vector_specs:
            entry = {
                "id": spec.id, "content_digest": spec.content_digest,
                "prompt_sha256": spec.prompt_sha256, "n_predict": spec.n_predict, "seed": spec.seed,
                "applies_to": list(spec.applies_to), "requirements": list(spec.requirements),
                "scenario": spec.scenario, "provenance": spec.provenance,
            }
            verdict = verdicts_by_name.get(spec.id)
            if verdict is not None:
                entry.update({
                    "verdict": verdict.verdict,
                    "native_draft": [verdict.native.draft_n, verdict.native.draft_n_accepted],
                    "candidate_draft": [verdict.candidate.draft_n, verdict.candidate.draft_n_accepted],
                    "first_output_divergence": verdict.first_output_divergence,
                    "native_token_digest": behavioral_gate_mod.token_digest(verdict.native.generated_token_ids),
                    "candidate_token_digest": behavioral_gate_mod.token_digest(verdict.candidate.generated_token_ids),
                    "token_count": len(verdict.native.generated_token_ids),
                })
            selected_vectors.append(entry)
        report_document["selected_vectors"] = selected_vectors
    report_document["behavioral_gate_contract_version"] = behavioral_gate_mod.BEHAVIORAL_GATE_CONTRACT_VERSION
    report_document["runtime_profile_name"] = runtime_profile.name
    report_document["runtime_profile_digest"] = runtime_profile.digest
    atomic_write_json(report_path, report_document)

    if report.hard_fail or report.needs_throughput_adjudication:
        # HTR01: rather than immediately discarding the ENTIRE campaign's
        # yield (every other candidate's real, already-measured speed win)
        # over one bad signature, attempt a bounded recovery search using
        # ONLY data this campaign already collected -- no new GPU tuning.
        # Real hardware proof this matters: run-id hi141-proof-20260829-2231
        # (2026-08-29) hard-failed on exactly one of 41 promoted candidates
        # and shipped nothing at all.
        recovered = False
        if (allow_recovery and promoted_path is not None and manifest_path is not None
                and ggml_h_path is not None and dispatch_db is not None
                and correctness_binary_path is not None and correctness_vendor_root is not None):
            try:
                assignments = _load_signature_assignments(promoted_path)
                executor = recovery_mod.AssignmentExecutor(
                    binary_path=binary_path, model_path=model_path, devices=devices,
                    common_args=common_args, measurements_path=promoted_path,
                    manifest_path=manifest_path, ggml_h_path=ggml_h_path, workdir=workdir,
                    dispatch_db=dispatch_db,
                    correctness_binary_path=correctness_binary_path,
                    vendor_root=correctness_vendor_root, campaign_run_id=campaign_run_id,
                    recovery_run_id=f"{campaign_run_id}-recovery" if campaign_run_id else None,
                    correctness_seeds=correctness_seeds,
                    max_new_correctness_candidates=max_new_correctness_candidates,
                )
                strategy = recovery_mod.BoundedPairedBisectionStrategy()
                # KNOWN LIMITATION (not yet closed, tracked for follow-up):
                # precise dispatch-hit scoping via GGML_HIP_DISPATCH_HIT_LOG
                # cross-referencing is not wired here -- recovery searches
                # over every promoted non-native signature rather than only
                # the ones the failing vector(s) actually exercised. Correct
                # but less efficient than the design's intent.
                dispatch_hits = frozenset(assignments)
                result = recovery_mod.run_recovery(
                    executor=executor, strategy=strategy, initial_assignments=assignments,
                    initial_report=report, full_corpus=list(corpus), dispatch_hits=dispatch_hits,
                    max_evaluations=max_recovery_evaluations,
                )
            except recovery_mod.RecoveryError as exc:
                raise TuneCampaignError(
                    f"behavioral gate failed and recovery search could not produce a "
                    f"publishable assignment: {exc}. Original gate report: {report.summary()!r}"
                ) from exc
            recovery_report_path = workdir / "recovery-result.json"
            atomic_write_json(recovery_report_path, {
                "published": result.published, "final_overrides": result.final_overrides,
                "evaluations_used": result.evaluations_used, "stop_reason": result.stop_reason,
                # HTR04: structured evidence only -- no code path here or in
                # recovery.py acts on these. An operator (or a future,
                # explicit GPU-budget-governed policy that does not exist
                # yet) reads this to decide whether a targeted retune is
                # worth its real GPU cost.
                "retune_recommendations": [
                    {
                        "signature_dispatch": r.signature_dispatch, "reason": r.reason,
                        "current_assignment": r.current_assignment,
                        "exhausted_candidates": list(r.exhausted_candidates),
                    }
                    for r in result.retune_recommendations
                ],
            })
            if not result.published:
                raise TuneCampaignError(
                    f"behavioral gate failed and recovery search exhausted its budget "
                    f"without a publishable assignment ({result.stop_reason}). Original "
                    f"gate report: {report.summary()!r}"
                )
            provisional_cache = result.cache_path
            recovered = True
        if not recovered:
            verb = "hard-fail" if report.hard_fail else "requires throughput adjudication (not yet implemented)"
            raise TuneCampaignError(
                f"behavioral gate {verb}: refusing to ship this cache and recovery "
                f"was not attempted: {report.summary()!r}"
            )

    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    replay_coverage = coverage.get("replay", coverage)
    if not isinstance(replay_coverage, dict):
        replay_coverage = {}
    if (
        replay_coverage.get("stale")
        or replay_coverage.get("rerun_required", 0)
        or replay_coverage.get("exact", 0) <= 0
    ):
        raise TuneCampaignError(
            "replay verification requires exact cache hits with no stale or "
            "rerun_required entries -- exported cache was not proven usable: "
            f"{coverage!r}"
        )

    provisional_cache.replace(final_cache_path)
    # HTR03 provenance (GPT point A): bind this validation to the EXACT
    # cache artifact and report bytes, not just a path -- a path alone can
    # be silently overwritten by a later run reusing the same workdir.
    import hashlib
    coverage["validated_cache_digest"] = hashlib.sha256(final_cache_path.read_bytes()).hexdigest()
    coverage["behavioral_gate_report_path"] = str(report_path)
    coverage["behavioral_gate_report_digest"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    coverage["runtime_profile_digest"] = runtime_profile.digest
    coverage["behavioral_gate_contract_version"] = behavioral_gate_mod.BEHAVIORAL_GATE_CONTRACT_VERSION
    if corpus_edition is not None:
        coverage["corpus_edition_id"] = corpus_edition.edition
        coverage["corpus_content_digest"] = corpus_edition.content_digest
    return coverage


def run_tune_campaign(
    *,
    context: ProjectContext,
    cfg: campaign_config.Config,
    store: ArtifactStore,
    model_path: Path,
    platform_name: str,
    devices: str,
    runtime_profile_name: str,
    source_name: str = "bigcherry",
    run_id: str | None = None,
    workdir: Path | None = None,
    tune_screen_samples: int = 3,
    tune_final_samples: int = 15,
    correctness_seeds: tuple[int, ...] = (1, 2, 3),
    promotion_q: float = 0.05,
    promotion_threshold_pct: float = 1.0,
    promotion_resamples: int = 10_000,
) -> WorkflowReceipt:
    import uuid

    profile = cfg.runtime_profiles.get(runtime_profile_name)
    if profile is None:
        raise TuneCampaignError(
            f"no runtime-profile named {runtime_profile_name!r} -- known: "
            f"{sorted(cfg.runtime_profiles)}"
        )
    campaign_run_id = run_id or uuid.uuid4().hex[:12]
    workdir = workdir or (context.work_root / "tune-campaigns" / campaign_run_id)
    workdir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    devices_tuple = _devices_tuple(devices)
    gpu_mod.preflight_context(profile=profile, devices=devices_tuple, stage="record")

    record_result, record_db_path = _stage_record(
        context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-record",
        platform_name=platform_name, source_name=source_name, model_path=model_path,
        devices=devices, runtime_profile=profile, workdir=workdir,
    )
    inventory_path, _inventory_db_path = _stage_inventory_record(
        record_db_path=record_db_path, workdir=workdir,
    )
    tune_result, tune_measurements = _stage_tune(
        context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-tune",
        platform_name=platform_name, source_name=source_name,
        inventory_path=inventory_path, model_path=model_path, devices=devices,
        runtime_profile=profile, screen_samples=tune_screen_samples,
        final_samples=tune_final_samples, workdir=workdir,
    )
    if tune_result.manifest_ref is None:
        # HI125 close-out step 6: strengthened ingest needs a real manifest
        # to establish inventory.verify_hip_build_artifacts()'s capability
        # proof -- without one, build_attested is always False and
        # winner_verification is never written, silently defeating the
        # whole point of wiring a verifier in at all.
        raise TuneCampaignError(
            "tune build produced no manifest_ref -- strengthened ingest is impossible"
        )
    tune_manifest_path = Path(tune_result.manifest_ref.path)

    signature_verifier_result, signature_digest_verifier = _stage_signature_verifier(
        context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-signature-verifier",
        platform_name=platform_name, source_name=source_name, devices=devices,
    )

    dispatch_db, _first_promote_result, promoted_before, missing_evidence = (
        _stage_load_and_promote(
            tune_measurements=tune_measurements, tune_manifest_path=tune_manifest_path,
            workdir=workdir, q=promotion_q, threshold_pct=promotion_threshold_pct,
            resamples=promotion_resamples,
            signature_digest_verifier=signature_digest_verifier,
        )
    )

    correctness_result: CampaignLaneResult | None = None
    promoted_after = promoted_before
    # GPT deep-review P2 (2026-08-29): the receipt's verified_winners/
    # quarantined_unsupported_winners counts must reflect the FINAL ingest
    # of the dispatch_db that actually ships, not necessarily the first
    # pass -- when correctness evidence triggers a second
    # _stage_load_and_promote, that reingest can change which winners are
    # verified/quarantined, and _final_promote_result is updated below only
    # when that second pass actually runs.
    _final_promote_result = _first_promote_result
    if missing_evidence > 0:
        # gpt review (2026-08-27): triggering only on promoted_before == 0
        # missed the partial-evidence case -- a dispatch_db that already
        # had SOME reusable evidence from a prior run promotes those
        # candidates immediately while others stay
        # rejected_no_correctness_evidence; that case must still generate
        # evidence for the REMAINING candidates, not be skipped just
        # because promoted_before > 0. Re-running is cheap and always
        # safe (idempotent over already-decided rows).
        correctness_result = _stage_correctness_evidence(
            context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-correctness",
            platform_name=platform_name, source_name=source_name,
            inventory_path=inventory_path, tune_measurements=tune_measurements,
            dispatch_db=dispatch_db, seeds=correctness_seeds,
        )
        _dispatch_db2, _promote_result2, promoted_after, _missing_after = (
            _stage_load_and_promote(
                tune_measurements=tune_measurements, tune_manifest_path=tune_manifest_path,
                workdir=workdir, q=promotion_q, threshold_pct=promotion_threshold_pct,
                resamples=promotion_resamples,
                # Same memoized verifier instance reused across both loads --
                # correctness-evidence generation changes promotion evidence/
                # state, not canonical->digest identity, so the first pass's
                # already-verified canonicals hit the memo cache for free.
                signature_digest_verifier=signature_digest_verifier,
            )
        )
        _final_promote_result = _promote_result2

    replay_result: CampaignLaneResult | None = None
    replay_validation_result: CampaignLaneResult | None = None
    replay_coverage: dict | None = None
    dispatch_cache_path: Path | None = None
    if promoted_after > 0:
        # Build the replay binary FIRST, then export the cache against ITS
        # OWN manifest_ref -- not tune's. tune (workload-max) and replay
        # (replay-full) legitimately have different manifest_hash by design
        # (see _stage_replay_export docstring); exporting against tune's
        # manifest stamps every cache entry with a hash the replay binary
        # never has, so it always reads as stale. gpt review confirmed this
        # ordering bug is the actual HI130 root cause (req_ec659ded425c4335).
        replay_result = _stage_replay_build(
            context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-replay",
            platform_name=platform_name, source_name=source_name,
            inventory_path=inventory_path, winners_path=workdir / "promoted.jsonl",
        )
        if replay_result.manifest_ref is None:
            raise TuneCampaignError("replay build produced no manifest_ref -- cannot export cache")
        replay_validation_result = _stage_replay_build(
            context=context, cfg=cfg, store=store, run_id=f"{campaign_run_id}-replay-diagnostic",
            platform_name=platform_name, source_name=source_name,
            inventory_path=inventory_path, winners_path=workdir / "promoted.jsonl",
            build_name="replay-diagnostic",
        )
        _verify_replay_companion(replay_result, replay_validation_result)
        provisional_cache_path = _stage_replay_export(
            promoted_path=workdir / "promoted.jsonl",
            target_manifest_path=Path(replay_result.manifest_ref.path),
            target_source_root=replay_result.source_root,
            dispatch_db=dispatch_db, workdir=workdir,
        )
        # HI143: _stage_replay_validate raises TuneCampaignError (never
        # returns) unless every corpus vector's behavioral trace matches
        # native AND the existing coverage check passes -- only then does
        # the provisional cache get atomically renamed to dispatch.cache,
        # which is the path this receipt records below.
        replay_coverage = _stage_replay_validate(
            lane_result=replay_validation_result, model_path=model_path, devices=devices,
            runtime_profile=profile, provisional_cache=provisional_cache_path, workdir=workdir,
            promoted_path=workdir / "promoted.jsonl",
            manifest_path=Path(replay_result.manifest_ref.path),
            ggml_h_path=replay_result.source_root / "ggml" / "include" / "ggml.h",
            dispatch_db=dispatch_db,
            # HTR01: only available for lazy correctness-qualification if
            # THIS run actually built a test-backend-ops lane (it doesn't
            # when missing_evidence == 0 -- all evidence already existed
            # from a prior run). Recovery gracefully disables its lazy-
            # qualification path (falls through to the original hard_fail)
            # when these are None, rather than requiring a rebuild.
            correctness_binary_path=(
                correctness_result.binary_ref.path if correctness_result is not None else None
            ),
            correctness_vendor_root=(
                correctness_result.source_root if correctness_result is not None else None
            ),
            correctness_seeds=correctness_seeds, campaign_run_id=campaign_run_id,
        )
        replay_coverage["observation_role"] = "diagnostic-companion"
        replay_coverage["validation_build_plan_id"] = replay_validation_result.build_plan_id
        replay_coverage["production_build_plan_id"] = replay_result.build_plan_id
        dispatch_cache_path = workdir / "dispatch.cache"

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt = WorkflowReceipt(
        schema_version=4,
        campaign_run_id=campaign_run_id,
        model_path=str(model_path),
        platform_name=platform_name,
        devices=devices,
        runtime_profile_name=runtime_profile_name,
        effective_tune_context=profile.tune_context,
        record=_stage_identity(record_result),
        tune=_stage_identity(tune_result),
        signature_verifier=_stage_identity(signature_verifier_result),
        correctness=(
            _stage_identity(correctness_result)
            if correctness_result is not None else None
        ),
        replay=(
            _stage_identity(replay_result)
            if replay_result is not None else None
        ),
        replay_validation=(
            _stage_identity(replay_validation_result)
            if replay_validation_result is not None else None
        ),
        promoted_before_evidence=promoted_before,
        promoted_after_evidence=promoted_after,
        verified_winners=int(_final_promote_result.get("winner_verifications", 0)),
        quarantined_unsupported_winners=int(
            _final_promote_result.get("quarantined_unsupported_winners", 0)
        ),
        dispatch_cache_path=str(dispatch_cache_path) if dispatch_cache_path else None,
        replay_coverage=replay_coverage,
        started_at=started_at,
        finished_at=finished_at,
    )
    receipt_path = workdir / "tune-campaign-receipt.json"
    atomic_write_json(receipt_path, asdict(receipt))
    store.publish_json(f"tune-campaigns/{campaign_run_id}/receipt.json", asdict(receipt))
    return receipt
