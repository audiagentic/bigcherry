"""PA36 migration #9 (RD30/1237): patch-local validation producer.

Mechanically migrated off validation_campaign.py's
run_rd30_correctness_check() -- that function and the --run-rd30-correctness
CLI path are DELETED from shared code in the same change (no
compatibility layer, per the project's migrate-up doctrine).

The producer owns ONLY the measurement:
- The exact-output correctness check (256-expert MUL_MAT_ID shapes)
- bit_identical gate (byte-identical HIP outputs)
- backend_reference diagnostic

The dispatcher owns:
- The final promotion verdict

RD30's check (bit_identical):
- Control = normal BigCherry source + deterministic test-backend-ops evidence chain
- Subject = same composition + 1237
- Each arm runs the same exact 256-expert MUL_MAT_ID --test-file shapes
- The bit-identical gate requires, for every shape/seed:
  * successful execution in both arms,
  * identical deterministic routing (leaf_2 digest),
  * identical CPU-reference output digest,
  * identical output element count, and
  * identical HIP backend output bytes (backend1_digest)
- Scoped to gfx1100 exactly
"""

from __future__ import annotations

from bigcherry.patch import validation_producer as vp


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run RD30's validation producer.

    Returns a ProducerResult with:
    - contract_correctness_results: the bit_identical and backend_reference results
    - emitted_artifacts: the required artifacts
    """
    import math as _math
    import os
    import subprocess

    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.patch import source as psi
    from bigcherry.tuning import correctness_evidence
    from bigcherry.tuning import signature_mapping

    # RD30 is scoped to gfx1100 exactly
    targets = tuple(
        target.strip()
        for target in ctx.amdgpu_targets.replace(",", ";").split(";")
        if target.strip()
    )
    if targets != ("gfx1100",):
        raise vp.ValidationProducerError(
            "RD30 correctness is scoped to gfx1100 exactly; "
            f"got AMDGPU_TARGETS={ctx.amdgpu_targets!r}"
        )

    seeds = (1, 2, 3)
    if not seeds or any(seed == 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise vp.ValidationProducerError(
            "RD30 correctness requires a non-empty set of unique nonzero seeds"
        )

    evidence_patches = (
        "1222_hi67_deterministic_test_backend_ops_seed",
        "1223_hi67_machine_readable_correctness_metrics",
        "1236_hi105_deterministic_mul_mat_id_ids",
    )
    subject_patch = "1237_rd30_moe_mmq_compact_grid"

    # Resolve compositions
    control_revision, control_composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=evidence_patches,
        base_ref=ctx.base_revision,
        base_repo=ctx.base_repo,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=(*evidence_patches, subject_patch),
        base_ref=ctx.base_revision,
        base_repo=ctx.base_repo,
    )
    if control_revision != subject_revision:
        raise vp.ValidationProducerError(
            "RD30 correctness: control and subject resolved different base revisions"
        )

    control_src = psi.materialize_composition(
        base_repo=ctx.base_repo,
        worktree_root=ctx.worktree_root / "rd30-correctness-control",
        resolved_revision=control_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=ctx.base_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=ctx.base_repo,
        worktree_root=ctx.worktree_root / "rd30-correctness-subject",
        resolved_revision=subject_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=ctx.base_revision,
    )

    # Build test-backend-ops for both
    exe = ".exe" if __import__("sys").platform == "win32" else ""
    correctness_build_root = ctx.build_root / "rd30-correctness"
    architecture_tag = ctx.amdgpu_targets.replace(";", "_").replace(",", "_")
    control_name = f"rd30-correctness-control-{architecture_tag}"
    subject_name = f"rd30-correctness-subject-{architecture_tag}"

    control_bin_dir = ctx.runtime.build_tree(
        name=control_name,
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=correctness_build_root,
        targets=["test-backend-ops"],
        source=control_src,
        extra_cmake_args=[],
    )
    subject_bin_dir = ctx.runtime.build_tree(
        name=subject_name,
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=correctness_build_root,
        targets=["test-backend-ops"],
        source=subject_src,
        extra_cmake_args=[],
    )
    control_binary = control_bin_dir / f"test-backend-ops{exe}"
    subject_binary = subject_bin_dir / f"test-backend-ops{exe}"

    # Load op/type names
    op_names = signature_mapping.load_ggml_op_names(control_src)
    type_names = signature_mapping.load_ggml_type_names(control_src)

    def _enum_id(names: dict[int, str], wanted: str, kind: str) -> int:
        for value, name in names.items():
            if name.upper() == wanted.upper():
                return int(value)
        raise vp.ValidationProducerError(
            f"RD30 correctness: {kind} enum {wanted!r} "
            "is absent from materialized source"
        )

    op_mul_mat_id = _enum_id(op_names, "MUL_MAT_ID", "ggml_op")
    type_f32 = _enum_id(type_names, "F32", "ggml_type")

    # 32 routed tokens is deliberately > upstream MMVQ_MAX_BATCH_SIZE (8).
    shape_specs = (
        ("q4_k-moe-prefill32", "Q4_K"),
        ("q8_0-moe-prefill32", "Q8_0"),
    )

    scratch_dir = ctx.run_dir / "scratch" / "rd30-correctness"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mapped_shapes: list[dict[str, object]] = []
    for shape_name, weight_type in shape_specs:
        signature = {
            "op": op_mul_mat_id,
            "flags": 0x0F,
            "ne0": [2048, 256, 256, 1],
            "ne1": [2048, 1, 32, 1],
            "ned": [256, 8, 32, 1],
            "n_expert": 256,
            "n_expert_used": 8,
            "src0_type": _enum_id(type_names, weight_type, "ggml_type"),
            "src1_type": type_f32,
            "dst_type": type_f32,
        }
        line, target_tensor, digest_tensor = (
            signature_mapping.signature_to_mul_mat_id_test_file_line(
                signature,
                vendor_root=control_src,
            )
        )
        test_file = scratch_dir / f"{shape_name}.txt"
        test_file.write_text(line + "\n", encoding="utf-8")
        mapped_shapes.append(
            {
                "name": shape_name,
                "weight_type": weight_type,
                "signature": signature,
                "test_file": test_file,
                "target_tensor": target_tensor,
                "digest_tensor": digest_tensor,
            }
        )

    runner = subprocess.run

    def _correctness_runner(argv, **kwargs):
        env = {**os.environ, **(kwargs.pop("env", None) or {})}
        return runner(argv, env=env, **kwargs)

    def _finite_or_none(value: float) -> float | None:
        return float(value) if _math.isfinite(float(value)) else None

    rows: list[dict[str, object]] = []
    for shape in mapped_shapes:
        for seed in seeds:
            control = correctness_evidence.collect_native_seed_evidence(
                control_binary,
                test_file=shape["test_file"],
                target_tensor=shape["target_tensor"],
                digest_tensor=shape["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )
            subject = correctness_evidence.collect_native_seed_evidence(
                subject_binary,
                test_file=shape["test_file"],
                target_tensor=shape["target_tensor"],
                digest_tensor=shape["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )

            control_backend_ok = (
                control.native_execution_status == "ok"
                and _math.isfinite(control.e_n_nmse)
                and _math.isfinite(control.threshold_t)
                and control.e_n_nmse <= control.threshold_t
            )
            subject_backend_ok = (
                subject.native_execution_status == "ok"
                and _math.isfinite(subject.e_n_nmse)
                and _math.isfinite(subject.threshold_t)
                and subject.e_n_nmse <= subject.threshold_t
            )

            bit_identical = (
                control.native_execution_status == "ok"
                and subject.native_execution_status == "ok"
                and control.reference_digest == subject.reference_digest
                and control.reference_output_digest is not None
                and control.reference_output_digest
                == subject.reference_output_digest
                and control.native_output_digest is not None
                and control.native_output_digest == subject.native_output_digest
                and control.output_nels is not None
                and control.output_nels == subject.output_nels
            )

            rows.append(
                {
                    "shape": shape["name"],
                    "weight_type": shape["weight_type"],
                    "seed": seed,
                    "control_status": control.native_execution_status,
                    "subject_status": subject.native_execution_status,
                    "control_ids_digest": control.reference_digest,
                    "subject_ids_digest": subject.reference_digest,
                    "control_output_digest": control.native_output_digest,
                    "subject_output_digest": subject.native_output_digest,
                    "control_reference_output_digest":
                        control.reference_output_digest,
                    "subject_reference_output_digest":
                        subject.reference_output_digest,
                    "control_output_nels": control.output_nels,
                    "subject_output_nels": subject.output_nels,
                    "control_nmse": _finite_or_none(control.e_n_nmse),
                    "subject_nmse": _finite_or_none(subject.e_n_nmse),
                    "control_threshold":
                        _finite_or_none(control.threshold_t),
                    "subject_threshold":
                        _finite_or_none(subject.threshold_t),
                    "control_max_abs":
                        _finite_or_none(control.max_abs_native),
                    "subject_max_abs":
                        _finite_or_none(subject.max_abs_native),
                    "backend_reference_ok":
                        control_backend_ok and subject_backend_ok,
                    "bit_identical": bit_identical,
                }
            )

    # Evaluate gates
    first_exact_failure = next(
        (row for row in rows if not row["bit_identical"]),
        None,
    )
    if first_exact_failure is None:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=True,
            detail=(
                f"{len(rows)} RD30 256-expert MUL_MAT_ID "
                "(shape,seed) pairs produced byte-identical HIP outputs"
            ),
        )
    else:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=False,
            detail=(
                "RD30 exact-output mismatch for "
                f"shape={first_exact_failure['shape']!r} "
                f"seed={first_exact_failure['seed']}: "
                f"control_status={first_exact_failure['control_status']} "
                f"subject_status={first_exact_failure['subject_status']} "
                "ids_equal="
                f"{first_exact_failure['control_ids_digest'] == first_exact_failure['subject_ids_digest']} "
                "reference_equal="
                f"{first_exact_failure['control_reference_output_digest'] == first_exact_failure['subject_reference_output_digest']} "
                "output_equal="
                f"{first_exact_failure['control_output_digest'] == first_exact_failure['subject_output_digest']} "
                "nels_equal="
                f"{first_exact_failure['control_output_nels'] == first_exact_failure['subject_output_nels']}"
            ),
        )

    first_backend_failure = next(
        (row for row in rows if not row["backend_reference_ok"]),
        None,
    )
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=first_backend_failure is None,
        detail=(
            f"{len(rows)} subject/control rows stayed within each emitted "
            "backend-reference threshold"
            if first_backend_failure is None
            else (
                "RD30 backend-reference failure for "
                f"shape={first_backend_failure['shape']!r} "
                f"seed={first_backend_failure['seed']}: "
                f"control_nmse={first_backend_failure['control_nmse']} "
                f"control_threshold={first_backend_failure['control_threshold']} "
                f"subject_nmse={first_backend_failure['subject_nmse']} "
                f"subject_threshold={first_backend_failure['subject_threshold']}"
            )
        ),
    )

    # Write the artifact
    artifact_doc = {
        "schema_version": 1,
        "contract_id": "RD30-MOE-MMQ-COMPACT-GRID",
        "check": "bit_identical",
        "passed": bit_identical_result.passed,
        "base_revision": ctx.base_revision,
        "architecture": "gfx1100",
        "mechanism": (
            "test-backend-ops MUL_MAT_ID native subject/control "
            "backend1_digest equality"
        ),
        "evidence_patches": list(evidence_patches),
        "subject_patch": subject_patch,
        "seeds": list(seeds),
        "shapes": [
            {
                "name": shape["name"],
                "weight_type": shape["weight_type"],
                "signature": shape["signature"],
                "target_tensor": shape["target_tensor"],
                "digest_tensor": shape["digest_tensor"],
            }
            for shape in mapped_shapes
        ],
        "control_source_tree": str(control_src),
        "subject_source_tree": str(subject_src),
        "rows": rows,
    }
    ctx.runtime.write_artifact(
        name="rd30-correctness.json",
        payload=artifact_doc,
    )

    return vp.ProducerResult(
        validation_build_identities=ctx.validation_build_identities,
        promotion_lane_effects={},
        promotion_target_metric={},
        promotion_trigger_evidence={},
        contract_correctness_results=(
            bit_identical_result,
            backend_reference_result,
        ),
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        correctness=bit_identical_result,
        activation_evidence=None,
        emitted_artifacts=frozenset({"rd30-correctness.json"}),
    )
