"""Source-level contracts for the HI19 identity namespace boundary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry.source import audit as source_audit # noqa: E402


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
    ctx = source_audit.AuditContext(engine="llamacpp",
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



def _mmq_context(tmp_path: Path):
    cuda = tmp_path / "ggml" / "src" / "ggml-cuda"
    instances = cuda / "template-instances"
    instances.mkdir(parents=True)
    return source_audit.AuditContext(engine="llamacpp", root=tmp_path, cuda=cuda, instances=instances)


def _by_id(ctx, check_id: str):
    return next(result for result in ctx.results if result.id == check_id)


def test_mmq_generated_types_parser_accepts_b11552_explicit_q8(tmp_path: Path):
    ctx = _mmq_context(tmp_path)
    generator = ctx.instances / "generate_cu_files.py"
    generator.write_text(
        'TYPES_MMQ = ["GGML_TYPE_Q4_0", "GGML_TYPE_MXFP4"]\n',
        encoding="utf-8",
    )
    (ctx.instances / "mmq-instance-q4_0.cu").write_text("", encoding="utf-8")
    (ctx.instances / "mmq-instance-mxfp4.cu").write_text("", encoding="utf-8")
    (ctx.cuda / "mmq.cu").write_text(
        """
static void ggml_cuda_mul_mat_q_switch_type() {
    mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
    mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
    mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
}
""",
        encoding="utf-8",
    )

    source_audit.check_mmq_types(ctx)

    result = _by_id(ctx, "mmq.generated_types_match_runtime")
    assert result.ok
    assert result.actual == ["GGML_TYPE_Q4_0", "GGML_TYPE_MXFP4"]


def test_mmq_j_search_parser_reads_b11552_top_level_loop(tmp_path: Path):
    ctx = _mmq_context(tmp_path)
    cases = "\n".join(
        f"        launch_mul_mat_q<type, {j}, fallback, prec_src1>(ctx, args, stream);"
        for j in source_audit.MMQ_J_VALUES
    )
    (ctx.cuda / "mmq.cuh").write_text(
        f"""
template <ggml_type type, bool fallback, ggml_prec prec_src1>
void mul_mat_q_switch_J() {{
{cases}
}}
""",
        encoding="utf-8",
    )
    (ctx.cuda / "mmq.cu").write_text(
        """
void ggml_cuda_mul_mat_q() {
    int ntiles_J_best = 99;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        ntiles_J_best = J;
    }
}
""",
        encoding="utf-8",
    )

    source_audit.check_mmq_j(ctx)

    assert _by_id(ctx, "mmq.j_switch_values").ok
    assert _by_id(ctx, "mmq.j_search_loop_bounds").ok
