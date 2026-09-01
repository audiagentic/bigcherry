"""GP07: no-GPU unit tests for the RCCL qualification campaign driver.

Uses fake child programs, never a real GPU or RCCL binary -- same pattern
as test_rccl_qualify.py.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from bigcherry.profiling import rccl_qualify as rq
from bigcherry.profiling import rccl_qualify_campaign as campaign
from bigcherry.profiling import rccl_schema as rs

_TEST_REVISION = rs.RcclCompatibilityRevision(
    rccl_version="2.28.3", rccl_source_revision="57e58688f44c77076ad536ef1f6b68741fc6e694",
)


def _py(code: str) -> str:
    return textwrap.dedent(code).strip()


def _make_fake_binary(tmp_path: Path, script: str, name: str = "fake") -> str:
    fake = tmp_path / f"{name}.py"
    fake.write_text(_py(script))
    if sys.platform.startswith("win"):
        wrapper = tmp_path / f"{name}_wrapper.bat"
        wrapper.write_text(f'@"{sys.executable}" "{fake}" %*\r\n')
    else:
        wrapper = tmp_path / f"{name}_wrapper.sh"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        wrapper.chmod(0o755)
    return str(wrapper)


_ALWAYS_PASS = """
    import sys, json
    out = sys.argv[sys.argv.index("-x") + 1]
    with open(out, "w") as f:
        json.dump([{"wrong": "0"}], f)
    # RCCL_OVERRIDE_ALGO/PROTO are always honored by this fake, so the
    # observed table always matches whatever was requested.
    import os
    print(f"    {os.environ['RCCL_OVERRIDE_ALGO'].upper():7s} {os.environ['RCCL_OVERRIDE_PROTO'].upper():7s}       2")
    sys.exit(0)
"""

_ALWAYS_GPU_FAULT = """
    import sys
    print("ROCm error: unhandled cuda error (run with NCCL_DEBUG=INFO for details)")
    sys.exit(1)
"""


def test_run_matrix_small_all_pass(tmp_path: Path):
    binary = _make_fake_binary(tmp_path, _ALWAYS_PASS)
    topologies = ((rq.RcclTopology(topology_id="xtx_xtx", device_arches=("gfx1100", "gfx1100")), (0, 1)),)
    results = campaign.run_matrix(
        topologies=topologies, element_counts=(1024,), algorithms=("Ring",),
        protocols=("Simple",), repetitions=3, binary=binary,
        compatibility=_TEST_REVISION, output_dir=tmp_path / "out",
    )
    assert len(results) == 3
    assert all(r.classification == rq.PASS for r in results)
    assert {r.attempt for r in results} == {1, 2, 3}


def test_run_matrix_covers_full_cartesian_product(tmp_path: Path):
    binary = _make_fake_binary(tmp_path, _ALWAYS_PASS)
    topologies = (
        (rq.RcclTopology(topology_id="xtx_xtx", device_arches=("gfx1100", "gfx1100")), (0, 1)),
        (rq.RcclTopology(topology_id="xtx0_r9700", device_arches=("gfx1100", "gfx1201")), (0, 2)),
    )
    results = campaign.run_matrix(
        topologies=topologies, element_counts=(1024, 2048), algorithms=("Ring", "Tree"),
        protocols=("Simple",), repetitions=1, binary=binary,
        compatibility=_TEST_REVISION, output_dir=tmp_path / "out",
    )
    # 2 topologies x 2 element_counts x 2 algorithms x 1 protocol x 1 rep
    assert len(results) == 8


def test_post_fault_triggers_control_recheck(tmp_path: Path):
    # A GPU_FAULT-only fake, used on a non-control topology, must trigger
    # a recheck of the control topology per the runbook's safety rule --
    # and since this fake always faults, the recheck itself also fails,
    # so the whole campaign must abort.
    binary = _make_fake_binary(tmp_path, _ALWAYS_GPU_FAULT)
    topologies = ((rq.RcclTopology(topology_id="xtx0_6900xt", device_arches=("gfx1100", "gfx1030")), (0, 3)),)
    with pytest.raises(campaign.CampaignAborted):
        campaign.run_matrix(
            topologies=topologies, element_counts=(1024,), algorithms=("Ring",),
            protocols=("Simple",), repetitions=1, binary=binary,
            compatibility=_TEST_REVISION, output_dir=tmp_path / "out",
        )


def test_control_recheck_success_allows_campaign_to_continue(tmp_path: Path):
    # A fake that GPU_FAULTs on the FIRST invocation only, then passes --
    # simulates one real transient fault whose control recheck succeeds,
    # so the campaign must continue rather than abort.
    binary = _make_fake_binary(tmp_path, """
        import sys, json, os
        marker = os.path.join(os.path.dirname(sys.argv[0]), "..", "invoked_once")
        marker = os.path.abspath(marker)
        if not os.path.exists(marker):
            open(marker, "w").close()
            print("ROCm error: unhandled cuda error")
            sys.exit(1)
        out = sys.argv[sys.argv.index("-x") + 1]
        with open(out, "w") as f:
            json.dump([{"wrong": "0"}], f)
        print(f"    {os.environ['RCCL_OVERRIDE_ALGO'].upper():7s} {os.environ['RCCL_OVERRIDE_PROTO'].upper():7s}       2")
        sys.exit(0)
    """)
    topologies = ((rq.RcclTopology(topology_id="xtx0_r9700", device_arches=("gfx1100", "gfx1201")), (0, 2)),)
    results = campaign.run_matrix(
        topologies=topologies, element_counts=(1024,), algorithms=("Ring",),
        protocols=("Simple",), repetitions=1, binary=binary,
        compatibility=_TEST_REVISION, output_dir=tmp_path / "out",
    )
    # First case (GPU_FAULT) + the control recheck (PASS, since it's the
    # binary's second invocation) = 2 results.
    assert len(results) == 2
    assert results[0].classification == rq.GPU_FAULT
    assert results[1].classification == rq.PASS
    assert results[1].topology_id == campaign.CONTROL_TOPOLOGY[0].topology_id


def test_summarize_counts_per_topology_classification(tmp_path: Path):
    binary = _make_fake_binary(tmp_path, _ALWAYS_PASS)
    topologies = ((rq.RcclTopology(topology_id="xtx_xtx", device_arches=("gfx1100", "gfx1100")), (0, 1)),)
    results = campaign.run_matrix(
        topologies=topologies, element_counts=(1024,), algorithms=("Ring",),
        protocols=("Simple",), repetitions=5, binary=binary,
        compatibility=_TEST_REVISION, output_dir=tmp_path / "out",
    )
    summary = campaign.summarize(results)
    assert summary["xtx_xtx"] == {rq.PASS: 5}


def test_parse_compatibility_rejects_bare_version():
    parser_args = campaign.argparse.Namespace(
        rccl_version="2.30.4", rccl_source_revision=None, library_build_id=None,
        rocm_install_label=None, build_config=None,
    )
    with pytest.raises(SystemExit):
        campaign._parse_compatibility(parser_args)


def test_parse_compatibility_accepts_source_revision():
    parser_args = campaign.argparse.Namespace(
        rccl_version="2.28.3", rccl_source_revision="57e58688f4", library_build_id=None,
        rocm_install_label="vendor/rocm/7.2.4", build_config=None,
    )
    rev = campaign._parse_compatibility(parser_args)
    assert rev.revision_id == "57e58688f4"


def test_default_topologies_include_control_and_negative_control():
    ids = {t.topology_id for t, _ in campaign.DEFAULT_TOPOLOGIES}
    assert campaign.CONTROL_TOPOLOGY[0].topology_id in ids
    # A device-3 negative control must be present by default -- GP06's
    # whole point was that this needs re-testing per RCCL revision, not
    # just the RCCL-viable pairs.
    assert any("6900xt" in topology_id for topology_id in ids)


def test_default_element_counts_match_gp06_matrix():
    assert campaign.DEFAULT_ELEMENT_COUNTS == (30720, 2621440)
