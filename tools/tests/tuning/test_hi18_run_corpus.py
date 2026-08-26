"""HI18: offline test for tools/bigcherry/hi18_run_corpus.py's orchestration
-- corpus generation, probe invocation, evaluation, and JSONL output --
against a fake `test-hip-reduce` stand-in so this layer's own wiring is
exercised without a compiled probe or real HIP hardware. The fake probe
computes the real CPU-double reduction and reports it as every provider's
"output", so a correct run of this module against the fake probe must pass
every case; a deliberately wrong fake probe must fail the run.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import hi18_run_corpus as runner  # noqa: E402
from bigcherry.tuning import reduction as rc # noqa: E402

FAKE_PROBE = textwrap.dedent(r"""
    import json
    import struct
    import sys
    from pathlib import Path

    def f32_bytes(values):
        return b"".join(struct.pack("<f", v) for v in values)

    def from_f32_bytes(data):
        n = len(data) // 4
        return list(struct.unpack(f"<{n}f", data))

    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    case_dir = Path(args["--case"])
    plan = args["--plan"]
    devices = [int(x) for x in args["--devices"].split(",")]
    out_path = Path(args["--out"])

    manifest = json.loads((case_dir / "case.json").read_text())
    device_count = manifest["device_count"]
    element_count = manifest["element_count"]

    rank_values = []
    for d in range(device_count):
        rank_values.append(from_f32_bytes((case_dir / f"rank-{d}.f32").read_bytes()))

    reference = [0.0] * element_count
    for d in range(device_count):
        for i in range(element_count):
            reference[i] += rank_values[d][i]

    outputs = []
    for rank, dev in enumerate(devices):
        out_data = f32_bytes(reference)
        out_file = out_path.with_name(out_path.stem + f"-rank-{rank}.f32")
        out_file.write_bytes(out_data)
        import hashlib
        outputs.append({
            "device": dev,
            "path": str(out_file),
            "sha256": hashlib.sha256(out_data).hexdigest(),
        })

    result = {
        "schema_version": 1,
        "case_id": manifest["case_id"],
        "plan": plan,
        "probe_valid": True,
        "completion_synchronized": True,
        "reduction_signature_matches_case": True,
        "reduction_signature_key": manifest["reduction_signature_key"],
        "reduction_signature": {
            "slice_shape": manifest["slice_shape"],
            "topology_key": manifest["topology_key"],
            "peer_access": manifest["peer_access"],
            "element_type": "f32",
        },
        "requested_provider": plan,
        "effective_provider": plan,
        "provider_succeeded": True,
        "handoff": "none",
        "fallback_depth": 0,
        "devices": devices,
        "input_digests": manifest["input_digests"],
        "outputs": outputs,
    }
    out_path.write_text(json.dumps(result))
""")


def _write_fake_probe(tmp_path: Path) -> Path:
    probe = tmp_path / "fake_probe.py"
    probe.write_text(FAKE_PROBE)
    return probe


def _run_main(tmp_path: Path, probe_script: Path, *, seeds="1,2") -> int:
    out_dir = tmp_path / "corpus"
    jsonl_path = tmp_path / "reduce-correctness.jsonl"
    return runner.main([
        "--probe", str(probe_script),
        "--element-count", "8", "--slice-shape", "4,2,1,1",
        "--topology-key", "n2:peer1001", "--peer-access", "partial",
        "--devices", "0,1", "--seeds", seeds,
        "--source-revision", "deadbeef", "--manifest-hash", "cafef00d",
        "--out-dir", str(out_dir), "--jsonl", str(jsonl_path),
    ])


def test_slice_shape_must_have_four_components(tmp_path, capsys):
    probe = _write_fake_probe(tmp_path)
    try:
        runner.main([
            "--probe", str(probe), "--element-count", "8", "--slice-shape", "4,2,1",
            "--topology-key", "n2:peer1001", "--peer-access", "partial",
            "--devices", "0,1", "--seeds", "1",
            "--source-revision", "x", "--manifest-hash", "y",
            "--out-dir", str(tmp_path / "c"), "--jsonl", str(tmp_path / "j.jsonl"),
        ])
        assert False, "expected SystemExit for a malformed slice-shape"
    except SystemExit as exc:
        assert "exactly 4 components" in str(exc)


def test_end_to_end_against_a_correct_fake_probe(tmp_path, monkeypatch):
    """A fake probe that always reports the true CPU-double reduction must
    produce a fully passing run through the real reduce_correctness.py
    evaluation path -- exercising corpus generation, subprocess wiring,
    load_probe_run() ingestion, evaluation, and JSONL writing end to end."""
    probe = _write_fake_probe(tmp_path)
    real_run = runner.subprocess.run

    def fake_run(cmd, env, capture_output, text):
        return real_run([sys.executable, str(probe)] + cmd[1:], env=env,
                         capture_output=capture_output, text=text)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc_code = _run_main(tmp_path, probe, seeds="1,2")
    assert rc_code == 0

    jsonl_path = tmp_path / "reduce-correctness.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    # 6 patterns x 2 seeds x 3 providers
    assert len(rows) == 6 * 2 * 3
    assert all(row["valid"] and row["correct"] for row in rows)
    assert all(row["reduction_signature_key"] == rc.make_reduction_signature_key(
        element_type="f32", element_count=8, slice_shape=(4, 2, 1, 1),
        topology_key="n2:peer1001",
    ) for row in rows)


def test_end_to_end_fails_closed_against_a_wrong_fake_probe(tmp_path, monkeypatch):
    """A fake probe that reports a corrupted result must fail the run
    (nonzero exit, at least one invalid/incorrect row) -- proving this
    orchestration layer does not silently pass a bad provider result."""
    probe = _write_fake_probe(tmp_path)
    broken_probe_src = FAKE_PROBE.replace(
        'reference[i] += rank_values[d][i]',
        'reference[i] += rank_values[d][i] + 1000.0',
    )
    broken_probe = tmp_path / "broken_probe.py"
    broken_probe.write_text(broken_probe_src)
    real_run = runner.subprocess.run

    def fake_run(cmd, env, capture_output, text):
        return real_run([sys.executable, str(broken_probe)] + cmd[1:], env=env,
                         capture_output=capture_output, text=text)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc_code = _run_main(tmp_path, broken_probe, seeds="1")
    assert rc_code == 1

    jsonl_path = tmp_path / "reduce-correctness.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert any(not row["valid"] or not row["correct"] for row in rows)
