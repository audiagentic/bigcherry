"""Card-free contract tests for the native scheduler gate adapter."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "scheduler_burst", ROOT / "tools/lab/radiance/scheduler_burst.py")
burst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burst)


def test_stage_native_target_once(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "sched_test.cpp").write_text("RAD_TEST_MAIN()\n")
    (tests / "config_test.cpp").write_text("RAD_TEST_MAIN()\n")
    core = tmp_path / "core"
    core.mkdir()
    (core / "engine_bringup.cpp").write_text("    if (!cfg_.prefill_interleave_set) {\n    }\n\n    /* Control vectors are resolved before any rank declares")
    (tests / "CMakeLists.txt").write_text("add_executable(sched_test sched_test.cpp)\n")
    burst.stage_fixture(tmp_path)
    assert '#include "bc_scheduler_burst_cases.h"' in (tests / "sched_test.cpp").read_text()
    assert '#include "sched_test.cpp"' not in (tests / "bc_scheduler_burst_cases.h").read_text()
    cmake = (tests / "CMakeLists.txt").read_text()
    for case, name in zip(burst.CASES + burst.CONFIG_CASES, burst.TESTS):
        target = "sched_test" if case in burst.CASES else "config_test"
        assert f"add_test(NAME {name} COMMAND {target} {case})" in cmake
    assert "PASS_REGULAR_EXPRESSION" not in cmake
    with pytest.raises(ValueError, match="already staged"):
        burst.stage_fixture(tmp_path)


def test_copy_is_private_and_content_exact(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "core.cpp").write_text("original")
    (src / ".git").write_text("ignored")
    dst = tmp_path / "dst"
    burst.copy_snapshot(src, dst)
    assert not (dst / ".git").exists()
    (dst / "core.cpp").write_text("different")
    assert (src / "core.cpp").read_text() == "original"
    assert burst.tree_files(src) != burst.tree_files(dst)


def test_external_symlink_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="external"):
        burst.copy_snapshot(src, tmp_path / "dst")


def test_slice_mismatch_refused_before_attestation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source.with_name(source.name + ".metadata.json").write_text(json.dumps(
        {"source_slice_id": "other"}))
    with pytest.raises(ValueError, match="source slice"):
        burst.attestation(source, "expected")


@pytest.fixture
def junit(tmp_path):
    root = ET.Element("testsuite")
    for name, case in zip(burst.TESTS, burst.CASES + burst.CONFIG_CASES):
        node = ET.SubElement(root, "testcase", {"name": name})
        ET.SubElement(node, "system-out").text = "  ok   " + case + "\n1 check(s)"
    path = tmp_path / "tests.xml"
    return root, path


def test_valid_native_pass(junit):
    root, path = junit
    ET.ElementTree(root).write(path)
    burst.validate_results(path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "skip", "failure", "vacuous"])
def test_invalid_native_result(junit, mutation):
    root, path = junit
    case = root[0]
    if mutation == "missing":
        root.remove(case)
    elif mutation == "duplicate":
        root.append(case)
    elif mutation == "skip":
        ET.SubElement(case, "skipped")
    elif mutation == "failure":
        ET.SubElement(case, "failure")
    else:
        case.find("system-out").text = "no case matches"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError):
        burst.validate_results(path)


def test_missing_default_seam_refuses():
    with pytest.raises(ValueError, match="seam changed"):
        burst.extract_interleave_block("not a supported source")
