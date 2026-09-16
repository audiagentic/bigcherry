"""PA36 sub-slice 2 (dev-gpt-agent req_2ecda033763949a9, T6): patch-local
validation producer for 1205 (RD12-PAIRED-MMVQ-DUAL).

Mechanically migrated off validation_campaign.py's
run_rd12_correctness_check() -- that function (and the --run-rd12-contract
CLI path) is DELETED from shared code in the same change (no compatibility
layer, per the project's migrate-up doctrine).

The producer owns ONLY the measurement: the paired test-backend-ops
build pair (fat multi-arch, built once via ``ctx.runtime.build_pair()``),
the per-(lane, seed) exact-output comparison, the raw per-arm activation
logs, and the raw correctness artifact. It returns semantic evidence
(``correctness`` = {disposition, mechanism, detail}, the per-arm
``trace_evidence`` artifact refs, the ``activation_evidence``) -- every
canonical identity field (patch digest, source tree, campaign identity,
root correctness.json / activation.json) is owned by the shared binder in
the generic dispatcher, never here.

Control:
  normal BigCherry composition
  + deterministic test-backend-ops seed support (1222)
  + machine-readable correctness metrics (1223)
  + paired plain-MUL_MAT whole-graph probe (1258)

Subject:
  exact same composition
  + 1205_rd12_paired_mmvq_dual_output

A normal signature_to_test_file_line()/--test-file case is intentionally
NOT used: it represents one isolated MUL_MAT and cannot satisfy RD12's
production selector, which requires two distinct adjacent MUL_MAT nodes
sharing the exact src1 tensor. 1236 is not needed -- this graph contains
no MUL_MAT_ID routing tensor.

Exact equality is required independently for the K and V outputs:
  * CPU-reference output digest,
  * HIP output digest,
  * output element count.

The focal patch's existing BIGCHERRY_PATCH_TRACE marker is also
mandatory. This prevents a false green result if control and subject
happen to produce identical bytes because the dual-output fusion never
activated.
"""

from __future__ import annotations

import math
import os
import subprocess

from bigcherry.patch import validation_producer as vp

# One contract architecture per run (the historical RD12 rule): the
# operator names it via --amdgpu-targets; the binary itself is built ONCE
# as the production-matching fat multi-arch set and the real device is
# selected at run time.
_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030")

_EVIDENCE_PATCHES: tuple[str, ...] = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
    "1258_rd12_paired_mul_mat_test_case",
)

# The registered 1258 test case is deliberately unique under -p.
_SHAPE: dict[str, object] = {
    "name": "q6_k-kv-decode1-1024x2560",
    "weight_type": "Q6_K",
    "m": 1024,
    "n": 1,
    "k": 2560,
    "params_filter": "bigcherry_rd12=1",
    "digest_tensor": "rd12_x",
}
_LANES: tuple[tuple[str, str], ...] = (("k", "rd12_k_out"), ("v", "rd12_v_out"))
_SEEDS: tuple[int, ...] = (1, 2, 3)
_TRACE_MARKER = "BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion"


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.patch import activation as patch_activation
    from bigcherry.patch import source as psi
    from bigcherry.tuning import correctness_evidence

    if (
        len(ctx.fat_targets.targets) != 1
        or ctx.fat_targets.targets[0] not in _CONTRACT_ARCHITECTURES
    ):
        raise vp.ValidationProducerError(
            "RD12 correctness requires exactly one contract architecture "
            "per run: gfx1100, gfx1201, or gfx1030; "
            f"got targets={ctx.fat_targets.targets!r}"
        )
    architecture = ctx.fat_targets.targets[0]
    # Real production builds compile ONE fat multi-arch binary and select
    # the real device to run it against at runtime -- match that instead
    # of rebuilding per architecture (the build-once-fat-multiarch rule).
    fat_targets = ";".join(_CONTRACT_ARCHITECTURES)

    if (
        not _SEEDS
        or any(seed == 0 for seed in _SEEDS)
        or len(set(_SEEDS)) != len(_SEEDS)
    ):
        raise vp.ValidationProducerError(
            "RD12 correctness requires a non-empty set of unique nonzero seeds"
        )

    # The one sanctioned pair authority: control = baseline + common
    # evidence patches; subject = baseline + common + focal (1205).
    pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="test-backend-ops",
        common_extra_patches=_EVIDENCE_PATCHES,
        baseline_source="bigcherry",
    )
    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    device = next((d for d in devices if d.architecture == architecture), None)
    if device is None:
        raise vp.ValidationProducerError(
            f"RD12 correctness: no device mapped for run architecture "
            f"{architecture!r}; device_map={ {k: tuple(v) for k, v in ctx.device_map.items()}!r} "
            "-- --device-map must select a real device for it"
        )

    activation_observations: list[dict[str, object]] = []
    activation_log_records: list[dict[str, object]] = []

    def _correctness_runner(argv, **kwargs):
        # correctness_evidence invokes the runner as
        # runner(argv, capture_output=True, text=True, env=run_env) where
        # run_env carries its deterministic seed + dispatch mode -- MERGE
        # it onto the sanctioned HIP-only device selector env instead of
        # forwarding it as a second env keyword (subprocess.run would
        # raise "multiple values for keyword argument 'env'"). The
        # selector env replaces any ambient selector (never ROCR
        # double-filtering) and enables the focal patch's existing
        # activation marker.
        env = {**os.environ, **dict(device.env_overrides), "BIGCHERRY_PATCH_TRACE": "1"}
        for key in device.env_unset:
            env.pop(key, None)
        env.update(dict(kwargs.pop("env", None) or {}))

        completed = subprocess.run(argv, env=env, **kwargs)

        executable = str(argv[0])
        if executable == str(control_binary):
            arm = "control"
        elif executable == str(subject_binary):
            arm = "subject"
        else:
            arm = "unknown"

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        seed = env.get("BIGCHERRY_TEST_DETERMINISTIC_SEED")
        activation_observations.append(
            {
                "arm": arm,
                "seed": seed,
                # Both streams: the per-arm logs below concatenate stdout+stderr
                # and _builtin_trace_marker() re-searches the WHOLE file, so the
                # producer's own hit observation must use the same search space
                # (GPT review req_243e3fcd3d684077: stderr-only here would
                # disagree with the validator if logging were ever redirected
                # to stdout).
                "hit": _TRACE_MARKER in stdout or _TRACE_MARKER in stderr,
            }
        )
        # Keep each invocation's raw streams for the per-arm activation
        # logs written below: the declared trace-marker check's validator
        # re-reads those files and re-verifies the marker itself, so the
        # "hit" observation above is NOT what the validator trusts.
        activation_log_records.append(
            {
                "arm": arm,
                "seed": seed,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        return completed

    def _finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(float(value)) else None

    rows: list[dict[str, object]] = []

    for lane, target_tensor in _LANES:
        for seed in _SEEDS:
            control = correctness_evidence.collect_native_seed_evidence(
                control_binary,
                op_filter=_SHAPE["params_filter"],
                target_tensor=target_tensor,
                digest_tensor=_SHAPE["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )
            subject = correctness_evidence.collect_native_seed_evidence(
                subject_binary,
                op_filter=_SHAPE["params_filter"],
                target_tensor=target_tensor,
                digest_tensor=_SHAPE["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )

            control_backend_ok = (
                control.native_execution_status == "ok"
                and math.isfinite(control.e_n_nmse)
                and math.isfinite(control.threshold_t)
                and control.e_n_nmse <= control.threshold_t
            )
            subject_backend_ok = (
                subject.native_execution_status == "ok"
                and math.isfinite(subject.e_n_nmse)
                and math.isfinite(subject.threshold_t)
                and subject.e_n_nmse <= subject.threshold_t
            )

            reference_equal = (
                control.reference_output_digest is not None
                and control.reference_output_digest == subject.reference_output_digest
            )
            output_equal = (
                control.native_output_digest is not None
                and control.native_output_digest == subject.native_output_digest
            )
            nels_equal = (
                control.output_nels is not None
                and control.output_nels == subject.output_nels
            )

            exact_equal = (
                control.native_execution_status == "ok"
                and subject.native_execution_status == "ok"
                and reference_equal
                and output_equal
                and nels_equal
            )

            rows.append(
                {
                    "shape": _SHAPE["name"],
                    "weight_type": _SHAPE["weight_type"],
                    "lane": lane,
                    "target_tensor": target_tensor,
                    "seed": seed,
                    "control_status": control.native_execution_status,
                    "subject_status": subject.native_execution_status,
                    "control_input_digest": control.reference_digest,
                    "subject_input_digest": subject.reference_digest,
                    "control_reference_output_digest": control.reference_output_digest,
                    "subject_reference_output_digest": subject.reference_output_digest,
                    "control_output_digest": control.native_output_digest,
                    "subject_output_digest": subject.native_output_digest,
                    "control_output_nels": control.output_nels,
                    "subject_output_nels": subject.output_nels,
                    "control_nmse": _finite_or_none(control.e_n_nmse),
                    "subject_nmse": _finite_or_none(subject.e_n_nmse),
                    "control_threshold": _finite_or_none(control.threshold_t),
                    "subject_threshold": _finite_or_none(subject.threshold_t),
                    "control_max_abs": _finite_or_none(control.max_abs_native),
                    "subject_max_abs": _finite_or_none(subject.max_abs_native),
                    "reference_equal": reference_equal,
                    "output_equal": output_equal,
                    "nels_equal": nels_equal,
                    "backend_reference_ok": control_backend_ok and subject_backend_ok,
                    "bit_identical": exact_equal,
                }
            )

    # One raw per-arm log each (RD08/RD73 precedent): these artifacts are
    # the ones the declared trace-marker check binds and re-verifies, so
    # they carry the real, unfiltered subprocess output. Written even when
    # activation fails, so a failed run's logs remain inspectable. Names
    # are namespaced by architecture (the historical standalone lab driver
    # shared one run_dir across all three contract architectures).
    subject_log_ref = ctx.runtime.write_text_artifact(
        name=f"activation-rd12-{architecture}-subject.log",
        text="\n---\n".join(
            f"{record['arm']}-seed{record['seed']}:\n{record['stdout']}\n{record['stderr']}"
            for record in activation_log_records
            if record["arm"] == "subject"
        ),
    )
    control_log_ref = ctx.runtime.write_text_artifact(
        name=f"activation-rd12-{architecture}-control.log",
        text="\n---\n".join(
            f"{record['arm']}-seed{record['seed']}:\n{record['stdout']}\n{record['stderr']}"
            for record in activation_log_records
            if record["arm"] == "control"
        ),
    )

    expected_runs_per_arm = len(_LANES) * len(_SEEDS)
    control_activation_runs = [
        observation
        for observation in activation_observations
        if observation["arm"] == "control"
    ]
    subject_activation_runs = [
        observation
        for observation in activation_observations
        if observation["arm"] == "subject"
    ]

    activation_ok = (
        len(control_activation_runs) == expected_runs_per_arm
        and len(subject_activation_runs) == expected_runs_per_arm
        and not any(observation["hit"] for observation in control_activation_runs)
        and all(observation["hit"] for observation in subject_activation_runs)
    )

    activation_result = experiment_contract.CorrectnessResult(
        check="activation",
        passed=activation_ok,
        detail=(
            f"RD12 dual-output MMVQ activation proven for all {expected_runs_per_arm} "
            "subject runs; control emitted no focal marker"
            if activation_ok
            else (
                "RD12 activation attestation failed: "
                f"control_runs={len(control_activation_runs)} "
                f"control_hits={sum(bool(row['hit']) for row in control_activation_runs)} "
                f"subject_runs={len(subject_activation_runs)} "
                f"subject_hits={sum(bool(row['hit']) for row in subject_activation_runs)} "
                f"expected_per_arm={expected_runs_per_arm}"
            )
        ),
    )

    first_exact_failure = next((row for row in rows if not row["bit_identical"]), None)

    if first_exact_failure is None and activation_result.passed:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=True,
            detail=(
                f"{len(rows)} RD12 (projection,seed) rows produced byte-identical "
                "CPU-reference and HIP outputs with focal fusion activation proven"
            ),
        )
    elif first_exact_failure is not None:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=False,
            detail=(
                f"RD12 exact-output mismatch for shape={first_exact_failure['shape']!r} "
                f"lane={first_exact_failure['lane']!r} seed={first_exact_failure['seed']}: "
                f"control_status={first_exact_failure['control_status']} "
                f"subject_status={first_exact_failure['subject_status']} "
                f"reference_equal={first_exact_failure['reference_equal']} "
                f"output_equal={first_exact_failure['output_equal']} "
                f"nels_equal={first_exact_failure['nels_equal']}"
            ),
        )
    else:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=False,
            detail=(
                "RD12 exact output digests matched, but the focal dual-output MMVQ "
                f"path was not proven active: {activation_result.detail}"
            ),
        )

    first_backend_failure = next(
        (row for row in rows if not row["backend_reference_ok"]), None
    )
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=first_backend_failure is None,
        detail=(
            f"{len(rows)} subject/control rows stayed within each emitted "
            "backend-reference threshold"
            if first_backend_failure is None
            else (
                f"RD12 backend-reference failure for shape={first_backend_failure['shape']!r} "
                f"lane={first_backend_failure['lane']!r} seed={first_backend_failure['seed']}: "
                f"control_nmse={first_backend_failure['control_nmse']} "
                f"control_threshold={first_backend_failure['control_threshold']} "
                f"subject_nmse={first_backend_failure['subject_nmse']} "
                f"subject_threshold={first_backend_failure['subject_threshold']}"
            )
        ),
    )

    # The RAW producer artifact: contract-internal measurements
    # (bit_identical / backend_reference) stay HERE -- they are not
    # validation-plan check ids, so they never become ProducerCheckResults.
    artifact_doc = {
        "schema_version": 1,
        "contract_id": "RD12-PAIRED-MMVQ-DUAL",
        "check": "bit_identical",
        "passed": bit_identical_result.passed,
        # Contract-internal measurements, persisted for audit -- they are
        # NOT validation-plan check ids, so they never become
        # ProducerCheckResults (dev-gpt-agent req_2ecda033763949a9 ruling).
        "contract_results": {
            "bit_identical": {
                "check": bit_identical_result.check,
                "passed": bit_identical_result.passed,
                "detail": bit_identical_result.detail,
            },
            "backend_reference": {
                "check": backend_reference_result.check,
                "passed": backend_reference_result.passed,
                "detail": backend_reference_result.detail,
            },
            "activation": {
                "check": activation_result.check,
                "passed": activation_result.passed,
                "detail": activation_result.detail,
            },
        },
        "base_revision": ctx.base_revision,
        "architecture": architecture,
        "compiled_targets": fat_targets,
        "mechanism": (
            "registered whole-graph paired MUL_MAT test-backend-ops subject/control "
            "CPU-reference + backend1 digest equality"
        ),
        "evidence_patches": list(_EVIDENCE_PATCHES),
        "subject_patch": "1205_rd12_paired_mmvq_dual_output",
        "seeds": list(_SEEDS),
        "shape": {
            "name": _SHAPE["name"],
            "weight_type": _SHAPE["weight_type"],
            "m": _SHAPE["m"],
            "n": _SHAPE["n"],
            "k": _SHAPE["k"],
            "params_filter": _SHAPE["params_filter"],
            "digest_tensor": _SHAPE["digest_tensor"],
            "target_tensors": [target_tensor for _, target_tensor in _LANES],
        },
        "activation": {
            "marker": _TRACE_MARKER,
            "passed": activation_result.passed,
            "subject_log": subject_log_ref.path,
            "control_log": control_log_ref.path,
            "observations": activation_observations,
        },
        "control_source_tree": psi.git_worktree_tree(pair.control_source),
        "subject_source_tree": psi.git_worktree_tree(pair.subject_source),
        "control_build_identity": pair.validation_build_identities["control"],
        "subject_build_identity": pair.validation_build_identities["subject"],
        "rows": rows,
    }
    correctness_artifact_ref = ctx.runtime.write_artifact(
        name=f"rd12-correctness-{architecture}.json",
        payload=artifact_doc,
    )

    # Semantic evidence for the shared binder (T3): exactly
    # {disposition, mechanism, detail} -- no identity fields.
    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if bit_identical_result.passed else "failed",
            "mechanism": "rd12-paired-mmvq-bit-identical",
            "detail": bit_identical_result.detail,
        },
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=patch_activation.ActivationEvidence(
            status="executed" if activation_result.passed else "not_executed",
            mechanism="rd12-trigger-marker",
            detail=activation_result.detail,
        ),
        performance_evidence={},
        # The producer owns ONLY the per-arm artifact refs -- the plan
        # owns the marker semantics (the shared binder injects the
        # marker_regex from validation.toml's trace-marker check).
        trace_evidence={
            "positive": {
                "artifact": {
                    "path": subject_log_ref.path,
                    "sha256": subject_log_ref.sha256,
                },
            },
            "negative": {
                "artifact": {
                    "path": control_log_ref.path,
                    "sha256": control_log_ref.sha256,
                },
            },
        },
        check_results=(),
        lane_effects=(),
        emitted_artifacts=frozenset(
            {
                correctness_artifact_ref.name,
                subject_log_ref.name,
                control_log_ref.name,
            }
        ),
    )
