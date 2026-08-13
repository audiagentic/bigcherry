"""Source-level contracts for the HI19 identity namespace boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry import source_audit  # noqa: E402


_HEADER = """
struct ggml_hip_dispatch_signature_v1 {
    uint16_t schema_version;
    uint16_t op;
    uint8_t src0_type;
    uint8_t prec;
};

struct ggml_hip_candidate_descriptor {
    uint32_t runtime_id;
    const char * stable_name;
    uint8_t family;
    ggml_hip_variant_params variant;
};
"""


def _audit_result(tmp_path: Path, header: str):
    cuda = tmp_path / "ggml" / "src" / "ggml-cuda"
    cuda.mkdir(parents=True)
    (cuda / "hip-autotune-types.h").write_text(header, encoding="utf-8")
    ctx = source_audit.AuditContext(
        root=tmp_path, cuda=cuda, instances=cuda / "template-instances")
    source_audit.check_identity_namespace_separation(ctx)
    return ctx.results


def test_current_namespace_fixture_passes(tmp_path: Path):
    results = _audit_result(tmp_path, _HEADER)
    assert len(results) == 1
    assert results[0].id == "identity.signature_candidate_separation"
    assert results[0].ok


def test_signature_field_inserted_into_candidate_fails(tmp_path: Path):
    header = _HEADER.replace(
        "    ggml_hip_variant_params variant;",
        "    ggml_hip_variant_params variant;\n    uint8_t prec;",
    )
    results = _audit_result(tmp_path, header)
    assert not results[0].ok
    assert "prec" in results[0].detail


def test_candidate_field_inserted_into_signature_fails(tmp_path: Path):
    header = _HEADER.replace(
        "    uint8_t prec;",
        "    uint8_t prec;\n    const char * stable_name;",
    )
    results = _audit_result(tmp_path, header)
    assert not results[0].ok
    assert "stable_name" in results[0].detail


def test_identity_check_is_registered_in_full_audit():
    assert source_audit.check_identity_namespace_separation in source_audit.ALL_CHECKS
