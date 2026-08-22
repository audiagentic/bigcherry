"""RE30 phase 1: the Vulkan source-audit profile.

Deliberately narrow scope: this profile can only verify structural facts
about a pristine checkout (no Vulkan BigCherry patch exists yet to define a
real anchor contract against). See ``source_audit.vulkan_audit``'s docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry import paths, source_audit  # noqa: E402


def test_vulkan_audit_runs_clean_on_the_real_vendored_tree():
    root = paths.llama_root()
    if not paths.vulkan_dir(root).is_dir():
        return  # vendor checkout not materialised in this environment
    report = source_audit.vulkan_audit(root)
    assert report["profile"] == "vulkan"
    assert report["summary"]["errors"] == 0, report["checks"]


def test_vulkan_audit_fails_closed_on_a_non_vulkan_checkout(tmp_path: Path):
    report = source_audit.vulkan_audit(tmp_path)
    assert report["summary"]["errors"] >= 1
    assert not source_audit.passed(report, strict=False)


def test_vulkan_audit_reports_missing_shader_dir(tmp_path: Path):
    vk_dir = tmp_path / "ggml" / "src" / "ggml-vulkan"
    vk_dir.mkdir(parents=True)
    (vk_dir / "CMakeLists.txt").write_text("# stub\n", encoding="utf-8")
    (vk_dir / "ggml-vulkan.cpp").write_text(
        "void ggml_vk_mul_mat_q_f16() {}\n"
        "void ggml_vk_guess_matmul_pipeline() {}\n"
        "void ggml_vk_dispatch_pipeline() {}\n"
        "void ggml_vk_create_pipeline() {}\n",
        encoding="utf-8",
    )
    report = source_audit.vulkan_audit(tmp_path)
    ids = {c["id"]: c for c in report["checks"]}
    assert ids["vulkan.backend_dir_present"]["ok"]
    assert ids["vulkan.host_source_present"]["ok"]
    assert ids["vulkan.dispatch_seam_symbols_present"]["ok"]
    assert not ids["vulkan.shaders_dir_present"]["ok"]


def test_vulkan_audit_reports_missing_dispatch_seam_symbols(tmp_path: Path):
    vk_dir = tmp_path / "ggml" / "src" / "ggml-vulkan"
    (vk_dir / "vulkan-shaders").mkdir(parents=True)
    (vk_dir / "CMakeLists.txt").write_text("# stub\n", encoding="utf-8")
    (vk_dir / "ggml-vulkan.cpp").write_text("// nothing here\n", encoding="utf-8")
    (vk_dir / "vulkan-shaders" / "mul_mat.comp").write_text("// stub\n", encoding="utf-8")
    (vk_dir / "vulkan-shaders" / "vulkan-shaders-gen.cpp").write_text("// stub\n", encoding="utf-8")
    report = source_audit.vulkan_audit(tmp_path)
    ids = {c["id"]: c for c in report["checks"]}
    assert not ids["vulkan.dispatch_seam_symbols_present"]["ok"]
    assert ids["vulkan.shader_sources_present"]["ok"]
    assert ids["vulkan.shader_generator_present"]["ok"]


def test_hip_audit_profile_is_unaffected_by_the_vulkan_profile_existing(tmp_path: Path):
    """Regression proof: adding VULKAN_CHECKS/vulkan_audit must not touch
    ALL_CHECKS or audit() -- the HIP profile stays exactly what it was."""
    report = source_audit.audit(tmp_path)
    assert report.get("profile") is None
    ids = {c["id"] for c in report["checks"]}
    assert not any(i.startswith("vulkan.") for i in ids)
