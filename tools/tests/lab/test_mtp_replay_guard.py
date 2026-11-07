"""The launcher refuses a start where MTP speculation runs but no draft/verify pass was recorded (rad0013's runtime guard)."""
import re
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[3] / "tools/lab/radiance/launch-flashnext-best.sh"


def guard_functions() -> str:
    text = LAUNCHER.read_text()
    start = text.index("mtp_replay_recorded() {")
    end = text.index("mtp_replay_check || exit 95")
    return text[start:end]


def run(tmp_path: Path, log_lines: list[str], canary_adds: str = "") -> subprocess.CompletedProcess:
    log = tmp_path / "engine.log"
    log.write_text("".join(f"{line}\n" for line in log_lines))
    # The re-check's extra canary request may cause the engine to record the pass; the stub appends that line.
    stub = f'start_canary() {{ printf "%s" "{canary_adds}" >> "$LOG"; return 0; }}\n'
    script = f'LOG="{log}"\n{stub}{guard_functions()}mtp_replay_check\n'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


MTP = "I speculation: 'mtp' drafts 3 token(s) a step in 3 serial pass(es); prompt lookup fills in for any sequence it declines"
RECORDED = "I recorded pass: draft pass 2 at 1 token(s) -- 37 records"
PASS0_ONLY = "I recorded pass: draft pass 0 at 1 token(s) -- 1549 records"
HISTORY = "I recorded pass: draft pass -1 at 5 token(s) -- 50 records"


def test_recorded_draft_pass_passes(tmp_path):
    assert run(tmp_path, [MTP, RECORDED]).returncode == 0


def test_mtp_without_recorded_pass_refuses_start(tmp_path):
    r = run(tmp_path, [MTP])
    assert r.returncode == 1
    assert "RADIANCE_EXIT=95" in r.stderr and "rad0013" in r.stderr and "depth 3" in r.stderr


def test_only_pass_zero_recorded_refuses_start(tmp_path):
    # v1.17 without rad0013 (hardware, rad0013-validate main arms): draft pass 0 recorded, every other pass refused.
    r = run(tmp_path, [MTP, PASS0_ONLY])
    assert r.returncode == 1 and "beyond pass 0" in r.stderr


def test_history_pass_counts(tmp_path):
    assert run(tmp_path, [MTP, PASS0_ONLY, HISTORY]).returncode == 0


def test_depth_one_is_not_checked(tmp_path):
    one = MTP.replace("drafts 3 token", "drafts 1 token")
    assert run(tmp_path, [one, PASS0_ONLY]).returncode == 0


def test_recording_after_second_canary_passes(tmp_path):
    assert run(tmp_path, [MTP], canary_adds=RECORDED + "\\n").returncode == 0


def test_no_speculation_is_not_checked(tmp_path):
    assert run(tmp_path, ["I chained launches: off"]).returncode == 0


def test_depth_zero_is_not_checked(tmp_path):
    zero = MTP.replace("drafts 3 token", "drafts 0 token")
    assert run(tmp_path, [zero]).returncode == 0


def test_launcher_runs_the_guard_after_the_canary():
    text = LAUNCHER.read_text()
    assert re.search(r"start_canary \|\| exit 96\n(?:.*\n){0,6}mtp_replay_recorded\(\)", text)
