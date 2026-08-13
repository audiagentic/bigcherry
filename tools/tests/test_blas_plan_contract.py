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


def test_forced_native_name_contains_every_resolved_plan_field():
    candidate = _forced_native()
    plan = candidate["config"]["blas_plan"]
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
