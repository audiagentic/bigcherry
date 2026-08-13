"""Offline BLAS-1 candidate-plan contracts (HI17)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import autotune_catalog as catalog  # noqa: E402
from bigcherry import autotune_schema as schema  # noqa: E402


def _forced_native() -> dict:
    candidates = catalog.enumerate_blas(["gfx1100"])
    assert len(candidates) == 1
    return candidates[0].to_dict()


def _signature(**overrides) -> dict:
    signature = {
        "src0_type": "q8_0",
        "src1_type": "f32",
        "dst_type": "f32",
        "prec": 0,
        "has_ids": False,
        "source_a_contiguous": True,
        "source_b_contiguous": True,
        "batched": False,
    }
    signature.update(overrides)
    return signature


def test_forced_native_name_contains_every_resolved_plan_field():
    candidate = _forced_native()
    plan = candidate["config"]["blas_plan"]
    schema.validate_candidate(candidate, "candidate.forced_native")
    schema.validate_blas_plan(plan, "candidate.config.blas_plan")

    expected_tokens = {
        "operand_type": f"operand-{plan['operand_type']}",
        "accumulation_type": f"accumulation-{plan['accumulation_type']}",
        "output_type": f"output-{plan['output_type']}",
        "source_a_conversion": f"source-a-{plan['source_a_conversion']}",
        "source_b_conversion": f"source-b-{plan['source_b_conversion']}",
        "output_conversion": f"output-conversion-{plan['output_conversion']}",
        "numerical_class": f"numerical-{plan['numerical_class']}",
    }
    assert candidate["stable_name"].startswith("blas:forced-native:")
    for token in expected_tokens.values():
        assert token in candidate["stable_name"]


def test_structured_blas_plan_must_match_stable_identity():
    candidate = _forced_native()
    candidate["config"]["blas_plan"]["operand_type"] = "f32"

    with pytest.raises(schema.SchemaError, match="resolved BLAS plan identity"):
        schema.validate_candidate(candidate, "candidate.mutated_plan")

    candidate = _forced_native()
    candidate["stable_name"] = candidate["stable_name"].replace(
        "operand-native", "operand-f32")

    with pytest.raises(schema.SchemaError, match="resolved BLAS plan identity"):
        schema.validate_candidate(candidate, "candidate.mutated_name")


def test_structured_blas_plan_requires_a_valid_mode():
    candidate = _forced_native()
    del candidate["config"]["mode"]

    with pytest.raises(schema.SchemaError, match="BLAS plan mode"):
        schema.validate_candidate(candidate, "candidate.missing_mode")


def test_native_blas_fallback_without_structured_plan_remains_valid():
    candidate = catalog.enumerate_natives(["gfx1100"], ["blas"])[0].to_dict()
    schema.validate_candidate(candidate, "candidate.native_fallback")
    assert candidate["stable_name"] == "blas:native:v1"


def test_strict_f32_rejects_reduced_precision_plans():
    plan = dict(_forced_native()["config"]["blas_plan"])
    plan["accumulation_type"] = "f16"
    plan["numerical_class"] = "reduced_precision"

    with pytest.raises(schema.SchemaError, match="GGML_PREC_F32"):
        schema.validate_blas_plan(
            plan, "candidate.config.blas_plan", prec=schema.GGML_PREC_F32)

    # The same plan remains a valid shape for the default precision class; the
    # strict gate is not a blanket ban on future reduced-precision experiments.
    schema.validate_blas_plan(plan, "candidate.config.blas_plan", prec=0)


def test_plan_must_name_all_conversion_and_numerical_fields():
    candidate = _forced_native()
    plan = candidate["config"]["blas_plan"]
    assert set(plan) == set(schema.BLAS_PLAN_FIELDS)
    for field in (
        "source_a_conversion", "source_b_conversion", "output_conversion",
        "numerical_class",
    ):
        assert isinstance(plan[field], str) and plan[field]


def test_plan_rejects_provider_or_api_identity_fields():
    plan = dict(_forced_native()["config"]["blas_plan"])
    plan["provider"] = "hipblas"
    with pytest.raises(schema.SchemaError, match="unexpected fields"):
        schema.validate_blas_plan(plan, "candidate.config.blas_plan")


def test_first_slice_emits_no_provider_or_api_candidates():
    candidates = catalog.enumerate_blas(["gfx1100"])
    assert [candidate.source_class for candidate in candidates] == ["existing_runtime"]
    for candidate in candidates:
        assert "provider" not in candidate.stable_name
        assert "api" not in candidate.stable_name
        assert "provider" not in candidate.config
        assert "api" not in candidate.config


def test_native_fallback_covers_every_requested_architecture():
    architectures = ["gfx1030", "gfx1100", "gfx1201"]
    natives = catalog.enumerate_natives(architectures, ["blas"])
    assert len(natives) == 1
    assert natives[0].stable_name == "blas:native:v1"
    assert set(natives[0].architectures) == set(architectures)


def test_resolver_normalizes_a_valid_native_plan_stably():
    plan = _forced_native()["config"]["blas_plan"]
    first = schema.resolve_blas_plan(plan, _signature())
    second = schema.resolve_blas_plan(dict(reversed(list(plan.items()))), _signature())

    assert first.rejection_reason is None
    assert second.rejection_reason is None
    assert first.plan == second.plan == plan
    assert list(first.plan) == list(schema.BLAS_PLAN_FIELDS)


def test_resolver_applies_strict_precision_before_runtime_selection():
    plan = dict(_forced_native()["config"]["blas_plan"])
    plan["numerical_class"] = "reduced_precision"

    strict = schema.resolve_blas_plan(plan, _signature(prec=schema.GGML_PREC_F32))
    default = schema.resolve_blas_plan(plan, _signature(prec=0))

    assert strict.plan is None
    assert strict.rejection_reason == "strict_precision_rejects_reduced_precision"
    assert default.plan == plan


def test_resolver_rejects_ids_but_accepts_batched_layout_for_native_auto():
    plan = _forced_native()["config"]["blas_plan"]
    ids = schema.resolve_blas_plan(plan, _signature(has_ids=True))
    batched = schema.resolve_blas_plan(plan, _signature(batched=True))

    assert ids.rejection_reason == "mul_mat_id_unsupported"
    assert batched.rejection_reason is None


def test_resolver_checks_source_and_output_conversion_routes():
    plan = {
        "operand_type": "f16",
        "accumulation_type": "f32",
        "output_type": "f16",
        "source_a_conversion": "contiguous",
        "source_b_conversion": "none",
        "output_conversion": "temporary_to_f32",
        "numerical_class": "equivalent_within_backend_tolerance",
    }
    valid = schema.resolve_blas_plan(
        plan, _signature(src0_type="f16", src1_type="f16",
                         source_a_contiguous=True, dst_type="f32"))
    bad_source = schema.resolve_blas_plan(
        plan, _signature(src0_type="f16", src1_type="f16",
                         source_a_contiguous=False, dst_type="f32"))
    bad_output = schema.resolve_blas_plan(
        plan, _signature(src0_type="f16", src1_type="f16",
                         source_a_contiguous=True, dst_type="f16"))

    assert valid.plan == plan
    assert bad_source.rejection_reason == "source_a_conversion_requires_contiguous_layout"
    assert bad_output.rejection_reason == "temporary_to_f32_requires_f16_or_bf16_output"
