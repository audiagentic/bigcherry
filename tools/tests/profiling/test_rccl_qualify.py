"""HI138/RQ03: no-GPU unit tests for the RCCL qualification harness.

Gate: no GPU work until this suite passes (RQ03). Uses fake child
programs (python -c '...') to exercise classification paths, never a
real GPU or RCCL binary.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from bigcherry.profiling import rccl_qualify as rq
from bigcherry.profiling import rccl_schema as rs


XTX_XTX = rq.RcclTopology(topology_id="xtx_xtx", device_arches=("gfx1100", "gfx1100"))
XTX_6900XT = rq.RcclTopology(topology_id="xtx_6900xt", device_arches=("gfx1100", "gfx1030"))
XTX_R9700 = rq.RcclTopology(topology_id="xtx_r9700", device_arches=("gfx1100", "gfx1201"))
XTX_XTX_6900XT = rq.RcclTopology(
    topology_id="xtx_xtx_6900xt", device_arches=("gfx1100", "gfx1100", "gfx1030")
)


def _fake_case(topology: rq.RcclTopology = XTX_XTX, **overrides) -> rq.RcclCase:
    kwargs = dict(topology=topology, element_count=8192, dtype="float")
    kwargs.update(overrides)
    return rq.RcclCase(**kwargs)


def _py(code: str) -> str:
    return textwrap.dedent(code).strip()


# ---------------------------------------------------------------------------
# RQ02: command construction
# ---------------------------------------------------------------------------


def test_command_contains_exact_size():
    case = _fake_case(element_count=8192, dtype="float")  # 8192 * 4 bytes
    command, _env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert "-b" in command and "-e" in command
    b_idx = command.index("-b")
    e_idx = command.index("-e")
    assert command[b_idx + 1] == "32768"
    assert command[e_idx + 1] == "32768"


def test_command_enables_correctness_check():
    case = _fake_case()
    command, _env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert "-c" in command
    assert command[command.index("-c") + 1] == "1"


def test_command_uses_internal_timeout():
    case = _fake_case()
    command, _env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert "-T" in command
    assert command[command.index("-T") + 1] == "20"


def test_command_requests_algo_proto_channel_reporting():
    case = _fake_case(algorithm="Tree", protocol="LL128")
    command, env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert "-M" in command
    assert command[command.index("-M") + 1] == "1"
    assert env["RCCL_OVERRIDE_ALGO"] == "Tree"
    assert env["RCCL_OVERRIDE_PROTO"] == "LL128"


def test_forced_algorithm_added_to_environment():
    case = _fake_case(algorithm="Ring")
    _command, env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert env["RCCL_OVERRIDE_ALGO"] == "Ring"


def test_forced_protocol_added_to_environment():
    case = _fake_case(protocol="LL")
    _command, env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(0, 1),
        rccl_output_path="/tmp/out.json",
    )
    assert env["RCCL_OVERRIDE_PROTO"] == "LL"


def test_visible_devices_set_from_diagnostic_binding_only():
    case = _fake_case()
    _command, env = rq.build_command(
        case, binary="all_reduce_perf", visible_devices=(2, 0),
        rccl_output_path="/tmp/out.json",
    )
    assert env["HIP_VISIBLE_DEVICES"] == "2,0"


# ---------------------------------------------------------------------------
# RQ03: process isolation / classification, using fake child programs
# ---------------------------------------------------------------------------


def _run_fake(tmp_path: Path, script: str, *, case=None, outer_timeout=10.0, name: str = "fake"):
    """Write `script` as a standalone python file and run it through
    rccl_qualify.run_case via a thin wrapper (run_case takes a single
    executable path, not "python script.py"), simulating one case's
    all_reduce_perf invocation."""
    fake = tmp_path / f"{name}.py"
    fake.write_text(_py(script))

    if sys.platform.startswith("win"):
        wrapper = tmp_path / "fake_all_reduce_perf.bat"
        wrapper.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
        binary = str(wrapper)
    else:
        wrapper = tmp_path / "fake_all_reduce_perf.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        wrapper.chmod(0o755)
        binary = str(wrapper)

    used_case = case or _fake_case()
    return rq.run_case(
        used_case, binary=binary, visible_devices=(0, 1),
        output_dir=tmp_path / "out", outer_timeout=outer_timeout,
    )


def test_clean_pass_is_classified_pass(tmp_path: Path):
    # Real -Z json output (verified on real hardware) is a JSON ARRAY of
    # per-pass records with a "wrong" string field; algo/proto/channels
    # only appear in -M 1's human-readable stdout table.
    result = _run_fake(tmp_path, """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}, {"wrong": "0"}], f)
        print("    RING    SIMPLE           2")
        sys.exit(0)
    """)
    assert result.classification == rq.PASS
    assert result.correct is True
    assert result.observed_algorithm == "RING"
    assert result.observed_channels == 2


def test_wrong_result_classified_from_rccl_wrong_field(tmp_path: Path):
    result = _run_fake(tmp_path, """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "3"}, {"wrong": "0"}], f)
        sys.exit(0)
    """)
    assert result.classification == rq.WRONG_RESULT
    assert result.correct is False


def test_timeout_kills_process_group(tmp_path: Path):
    result = _run_fake(
        tmp_path,
        "import time; time.sleep(30)",
        outer_timeout=0.5,
    )
    assert result.classification == rq.TIMEOUT


def test_rccl_internal_test_timeout_classified_timeout_not_init_failure(tmp_path: Path):
    # Regression: real hardware (RQ08) hit this exact case -- RCCL Tests'
    # own "-T" internal timeout prints routine ncclCommInitAll trace lines
    # (which appear on every run, successful or not) before "Test timeout",
    # and a too-loose INIT_FAILURE marker previously misclassified this.
    result = _run_fake(tmp_path, """
        import sys
        print("NCCL INFO ncclCommInitAll_impl comm rank 0 nranks 2 - Init COMPLETE")
        print("brutus: Test timeout (20s) common.cu.cpp:558")
        sys.exit(3)
    """)
    assert result.classification == rq.TIMEOUT


def test_nonzero_exit_classified_launch_failure(tmp_path: Path):
    result = _run_fake(tmp_path, "import sys; sys.exit(7)")
    assert result.classification == rq.LAUNCH_FAILURE
    assert result.returncode == 7


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX signals only")
def test_signal_exit_classified_signal(tmp_path: Path):
    result = _run_fake(tmp_path, """
        import os, signal
        os.kill(os.getpid(), signal.SIGSEGV)
    """)
    assert result.classification == rq.SIGNAL
    assert result.terminating_signal is not None


def test_gpu_fault_text_classified_gpu_fault(tmp_path: Path):
    result = _run_fake(tmp_path, """
        import sys
        print("ROCm error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)")
        sys.exit(1)
    """)
    assert result.classification == rq.GPU_FAULT


def test_device_lost_text_classified_device_lost_over_gpu_fault(tmp_path: Path):
    # Even when a GPU_FAULT-looking marker is also present, DEVICE_LOST must
    # take precedence -- it drives the campaign safety-stop path.
    result = _run_fake(tmp_path, """
        import sys
        print("amdgpu: GPU reset begin!")
        print("ROCm error: unhandled cuda error")
        sys.exit(1)
    """)
    assert result.classification == rq.DEVICE_LOST


def test_unsupported_text_classified_unsupported(tmp_path: Path):
    result = _run_fake(tmp_path, """
        import sys
        print("RCCL_OVERRIDE_PROTO=LL128 not supported on this topology")
        sys.exit(1)
    """)
    assert result.classification == rq.UNSUPPORTED


def test_symmetric_memory_benign_line_does_not_classify_unsupported(tmp_path: Path):
    # Real hardware regression (2026-09-02): "Symmetric memory is not
    # supported. cuMemEnable 0, ..." is a routine NCCL_DEBUG=INFO
    # capability-negotiation trace line printed on every run on this
    # hardware, successful or not -- it previously misclassified a clean
    # PASS as UNSUPPORTED via the bare "not supported" substring marker.
    result = _run_fake(tmp_path, """
        import sys
        print("brutus:1:1 [0] NCCL INFO Symmetric memory is not supported. "
              "cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1")
        print('    RING    SIMPLE           2')
        sys.exit(0)
    """)
    assert result.classification != rq.UNSUPPORTED


def test_real_unsupported_line_alongside_benign_line_still_classified_unsupported(tmp_path: Path):
    # The benign-line exclusion must not swallow a genuine decline that
    # happens to share a process with the routine trace line.
    result = _run_fake(tmp_path, """
        import sys
        print("brutus:1:1 [0] NCCL INFO Symmetric memory is not supported. "
              "cuMemEnable 0, globalGinSupport 0, globalNicFused 0 cuMemGdrSupport 1")
        print("RCCL_OVERRIDE_PROTO=LL128 not supported on this topology")
        sys.exit(1)
    """)
    assert result.classification == rq.UNSUPPORTED


def test_missing_binary_classified_harness_failure(tmp_path: Path):
    case = _fake_case()
    result = rq.run_case(
        case, binary=str(tmp_path / "does_not_exist_binary"),
        visible_devices=(0, 1), output_dir=tmp_path / "out", outer_timeout=5.0,
    )
    assert result.classification == rq.HARNESS_FAILURE


def test_clean_exit_without_json_output_is_harness_failure(tmp_path: Path):
    result = _run_fake(tmp_path, "import sys; sys.exit(0)")
    assert result.classification == rq.HARNESS_FAILURE


# ---------------------------------------------------------------------------
# Identity / determinism / topology-exclusion contract
# ---------------------------------------------------------------------------


def test_case_id_is_deterministic():
    case = _fake_case(algorithm="Ring", protocol="Simple")
    assert case.case_id == _fake_case(algorithm="Ring", protocol="Simple").case_id


def test_case_id_excludes_visible_devices_and_arches_stay_ordered():
    case_a = rq.RcclCase(
        topology=rq.RcclTopology(topology_id="t1", device_arches=("gfx1100", "gfx1030")),
        element_count=1024,
    )
    case_b = rq.RcclCase(
        topology=rq.RcclTopology(topology_id="t1", device_arches=("gfx1100", "gfx1030")),
        element_count=1024,
    )
    assert case_a.case_id == case_b.case_id
    # case_id is derived purely from topology_id/element_count/dtype/algo/proto,
    # never from any per-run device-ordinal binding.


def test_topology_identity_unaffected_by_diagnostic_visible_devices(tmp_path: Path):
    script = """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        sys.exit(0)
    """
    # Same case run twice with different diagnostic device bindings --
    # topology_id/case_id must be identical; only diagnostic_visible_devices
    # (evidence-only, never identity) is allowed to differ.
    fake = tmp_path / "fake.py"
    fake.write_text(_py(script))
    if sys.platform.startswith("win"):
        wrapper = tmp_path / "fake_all_reduce_perf.bat"
        wrapper.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        wrapper = tmp_path / "fake_all_reduce_perf.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        wrapper.chmod(0o755)

    result_a = rq.run_case(
        _fake_case(), binary=str(wrapper), visible_devices=(0, 1),
        output_dir=tmp_path / "out_a", outer_timeout=10.0,
    )
    result_b = rq.run_case(
        _fake_case(), binary=str(wrapper), visible_devices=(5, 9),
        output_dir=tmp_path / "out_b", outer_timeout=10.0,
    )
    assert result_a.topology_id == result_b.topology_id
    assert result_a.case_id == result_b.case_id
    assert result_a.diagnostic_visible_devices != result_b.diagnostic_visible_devices


def test_rccl_topology_rejects_no_ordinal_fields():
    # RcclTopology's only fields are topology_id and device_arches -- this
    # test documents/enforces that contract at the dataclass level.
    fields = {f for f in rq.RcclTopology.__dataclass_fields__}
    assert fields == {"topology_id", "device_arches"}


def test_rccl_case_result_rejects_unknown_classification(tmp_path: Path):
    with pytest.raises(ValueError):
        rq.RcclCaseResult(
            schema_version=1, case_id="x", topology_id="t", device_arches=("gfx1100",),
            diagnostic_visible_devices=(0,), element_count=1, dtype="float", byte_count=4,
            algorithm="Ring", protocol="Simple", requested_channels=None,
            observed_algorithm=None, observed_protocol=None, observed_channels=None,
            returncode=0, terminating_signal=None, elapsed_seconds=0.1,
            classification="not_a_real_state", correct=True, detail="",
            rccl_output_path="", stdout_path="", stderr_path="",
        )


# ---------------------------------------------------------------------------
# RQ15: append-only results file
# ---------------------------------------------------------------------------


def test_results_jsonl_append_only(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    r1 = rq.RcclCaseResult(
        schema_version=1, case_id="a", topology_id="t", device_arches=("gfx1100",),
        diagnostic_visible_devices=(0,), element_count=1, dtype="float", byte_count=4,
        algorithm="Ring", protocol="Simple", requested_channels=None,
        observed_algorithm=None, observed_protocol=None, observed_channels=None,
        returncode=0, terminating_signal=None, elapsed_seconds=0.1,
        classification=rq.PASS, correct=True, detail="", rccl_output_path="",
        stdout_path="", stderr_path="",
    )
    r2 = r1
    rq.append_result(r1, path)
    rq.append_result(r2, path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["case_id"] == "a"


def test_output_files_are_unique_per_case(tmp_path: Path):
    case_a = _fake_case(algorithm="Ring")
    case_b = _fake_case(algorithm="Tree")
    _cmd_a, _env_a = rq.build_command(
        case_a, binary="x", visible_devices=(0, 1), rccl_output_path="/tmp/a.json",
    )
    out_dir = tmp_path / "out"
    result_a = rq.run_case(case_a, binary=str(tmp_path / "missing"), visible_devices=(0, 1), output_dir=out_dir)
    result_b = rq.run_case(case_b, binary=str(tmp_path / "missing"), visible_devices=(0, 1), output_dir=out_dir)
    assert result_a.stdout_path != result_b.stdout_path
    assert result_a.rccl_output_path != result_b.rccl_output_path


# ---------------------------------------------------------------------------
# Topology fixtures sanity (gfx1030 must be present per explicit user
# instruction -- not just gfx1100/gfx1201)
# ---------------------------------------------------------------------------


def test_gfx1030_topology_fixtures_exist():
    assert "gfx1030" in XTX_6900XT.device_arches
    assert "gfx1030" in XTX_XTX_6900XT.device_arches


def test_byte_count_derivation_matches_dtype():
    case_f32 = _fake_case(element_count=8192, dtype="float")
    assert case_f32.byte_count == 8192 * 4
    case_f16 = _fake_case(element_count=8192, dtype="float16")
    assert case_f16.byte_count == 8192 * 2


def test_byte_count_rejects_unknown_dtype():
    case = _fake_case(dtype="not_a_real_dtype")
    with pytest.raises(ValueError):
        _ = case.byte_count


# ---------------------------------------------------------------------------
# GP07: plan-verification fail-closed behaviour, compatibility identity,
# attempt artifact isolation (gpt-dev-agent review, req_acf7c8da985f4f17)
# ---------------------------------------------------------------------------


def test_plan_substitution_downgrades_pass_to_unsupported(tmp_path: Path):
    # Requested Ring/Simple, RCCL actually reports Tree/LL128 -- must not
    # be a silent PASS.
    case = _fake_case(algorithm="Ring", protocol="Simple")
    result = _run_fake(tmp_path, """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        print("    TREE    LL128           4")
        sys.exit(0)
    """, case=case)
    assert result.classification == rq.UNSUPPORTED
    assert result.plan_verification == rq.PLAN_SUBSTITUTED


def test_missing_plan_observation_fails_closed_not_pass(tmp_path: Path):
    # Clean/correct exit but no -M 1 table line at all -- the requested
    # plan was never actually confirmed, so this must NOT default to PASS.
    case = _fake_case(algorithm="Ring", protocol="Simple")
    result = _run_fake(tmp_path, """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        sys.exit(0)
    """, case=case)
    assert result.classification == rq.HARNESS_FAILURE
    assert result.plan_verification == rq.PLAN_UNVERIFIED


def test_matched_plan_classified_pass_with_verified_marker(tmp_path: Path):
    case = _fake_case(algorithm="Ring", protocol="Simple")
    result = _run_fake(tmp_path, """
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        print("    RING    SIMPLE           2")
        sys.exit(0)
    """, case=case)
    assert result.classification == rq.PASS
    assert result.plan_verification == rq.PLAN_VERIFIED


def test_explicit_decline_marked_explicit_declined(tmp_path: Path):
    result = _run_fake(tmp_path, """
        import sys
        print("RCCL_OVERRIDE_PROTO=LL128 not supported on this topology")
        sys.exit(1)
    """)
    assert result.classification == rq.UNSUPPORTED
    assert result.plan_verification == rq.PLAN_EXPLICIT_DECLINE


_DURABLE_REVISION = rs.RcclCompatibilityRevision(
    rccl_version="2.28.3", rccl_source_revision="57e58688f44c77076ad536ef1f6b68741fc6e694",
)


def test_revision_id_requires_durable_identity():
    bare_version_only = rs.RcclCompatibilityRevision(rccl_version="2.30.4")
    with pytest.raises(rs.InsufficientCompatibilityIdentity):
        _ = bare_version_only.revision_id


def test_revision_id_prefers_library_build_id_over_source_revision():
    rev = rs.RcclCompatibilityRevision(
        rccl_version="2.30.4", rccl_source_revision="abc123",
        library_build_id="sha256:deadbeef",
    )
    assert rev.revision_id == "sha256:deadbeef"


def test_run_case_with_durable_compatibility_records_revision_and_key(tmp_path: Path):
    case = _fake_case(algorithm="Ring", protocol="Simple")
    fake = tmp_path / "fake.py"
    fake.write_text(_py("""
        import sys, json
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        print("    RING    SIMPLE           2")
        sys.exit(0)
    """))
    if sys.platform.startswith("win"):
        wrapper = tmp_path / "fake_all_reduce_perf.bat"
        wrapper.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        wrapper = tmp_path / "fake_all_reduce_perf.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        wrapper.chmod(0o755)

    result = rq.run_case(
        case, binary=str(wrapper), visible_devices=(0, 1),
        output_dir=tmp_path / "out", compatibility=_DURABLE_REVISION,
    )
    assert result.compatibility_revision_id == _DURABLE_REVISION.revision_id
    assert result.qualification_key == rs.qualification_key(_DURABLE_REVISION, case.case_id)
    # Enforced namespacing: artifacts land under output_dir/<revision_id>/.
    assert _DURABLE_REVISION.revision_id in result.stdout_path


def test_run_case_rejects_insufficient_compatibility_identity(tmp_path: Path):
    case = _fake_case()
    bare_version_only = rs.RcclCompatibilityRevision(rccl_version="2.30.4")
    with pytest.raises(rs.InsufficientCompatibilityIdentity):
        rq.run_case(
            case, binary=str(tmp_path / "does_not_exist"), visible_devices=(0, 1),
            output_dir=tmp_path / "out", compatibility=bare_version_only,
        )


def test_compatibility_manifest_mismatch_rejected(tmp_path: Path):
    # Same revision_id, different recorded fields -- must not silently
    # share a namespaced directory.
    rev_a = rs.RcclCompatibilityRevision(
        rccl_version="2.28.3", rccl_source_revision="samecommit",
        build_config="Release",
    )
    rev_b = rs.RcclCompatibilityRevision(
        rccl_version="2.28.3", rccl_source_revision="samecommit",
        build_config="Debug",
    )
    case = _fake_case()
    out_dir = tmp_path / "out"
    rq.run_case(
        case, binary=str(tmp_path / "missing"), visible_devices=(0, 1),
        output_dir=out_dir, compatibility=rev_a,
    )
    with pytest.raises(rq.CompatibilityManifestMismatch):
        rq.run_case(
            case, binary=str(tmp_path / "missing"), visible_devices=(0, 1),
            output_dir=out_dir, compatibility=rev_b,
        )


def test_attempt_suffix_isolates_repeated_case_artifacts(tmp_path: Path):
    case = _fake_case()
    fake = tmp_path / "fake.py"
    fake.write_text(_py("""
        import sys, json, random
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        print("    RING    SIMPLE           2")
        sys.exit(0)
    """))
    if sys.platform.startswith("win"):
        wrapper = tmp_path / "fake_all_reduce_perf.bat"
        wrapper.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        wrapper = tmp_path / "fake_all_reduce_perf.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        wrapper.chmod(0o755)

    out_dir = tmp_path / "out"
    result_1 = rq.run_case(
        case, binary=str(wrapper), visible_devices=(0, 1), output_dir=out_dir, attempt=1,
    )
    result_2 = rq.run_case(
        case, binary=str(wrapper), visible_devices=(0, 1), output_dir=out_dir, attempt=2,
    )
    assert result_1.stdout_path != result_2.stdout_path
    assert result_1.rccl_output_path != result_2.rccl_output_path
    assert Path(result_1.stdout_path).exists()
    assert Path(result_2.stdout_path).exists()
