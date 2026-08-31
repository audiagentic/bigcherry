"""Tests for report.py — measurement analysis and reporting (HI21).

Run with: python -m unittest tools.tests.test_report
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import inventory  # noqa: E402
from bigcherry.analysis import report as _report  # noqa: E402


# ------------------------------------------------------------------ fixtures

TUNING_HEADER = {
    "kind": "header",
    "artifact_version": 1,
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "variant_set": "workload-max",
    "build_descriptor_hash": "build-descriptor-test",
}

TUNING_RESULT_NATIVE = {
    "kind": "result",
    "dispatch": "e" * 32,
    "winner": "mmq:native:v1",
    "improvement_pct": 0.0,
    "generated": 3,
    "eligible": 3,
    "measured": 2,
    "reason": "native retained",
    "candidates": [
        {
            "name": "mmq:native:v1",
            "status": "ok",
            "median_us": 1.500,
            "mad_us": 0.010,
            "p95_us": 1.600,
            "host_median_us": 0.400,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 10,
        },
        {
            "name": "mmq:generated:j4",
            "status": "ok",
            "median_us": 1.520,
            "mad_us": 0.008,
            "p95_us": 1.580,
            "host_median_us": 0.390,
            "nmse": 1e-6,
            "max_abs": 0.001,
            "workspace": 4096,
            "samples": 10,
        },
        {
            "name": "mmq:generated:j8",
            "status": "architecture",
            "median_us": 0.0,
            "mad_us": 0.0,
            "p95_us": 0.0,
            "host_median_us": 0.0,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 0,
        },
    ],
}

TUNING_RESULT_IMPROVED = {
    "kind": "result",
    "dispatch": "f" * 32,
    "winner": "mmf:generated:nw4",
    "improvement_pct": 5.0,
    "generated": 4,
    "eligible": 3,
    "measured": 3,
    "reason": "measured winner",
    "candidates": [
        {
            "name": "mmf:native:v1",
            "status": "ok",
            "median_us": 2.000,
            "mad_us": 0.020,
            "p95_us": 2.200,
            "host_median_us": 0.600,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 10,
        },
        {
            "name": "mmf:generated:nw4",
            "status": "ok",
            "median_us": 1.900,
            "mad_us": 0.015,
            "p95_us": 2.050,
            "host_median_us": 0.550,
            "nmse": 2e-6,
            "max_abs": 0.002,
            "workspace": 8192,
            "samples": 10,
        },
        {
            "name": "mmf:generated:nw8",
            "status": "workspace",
            "median_us": 0.0,
            "mad_us": 0.0,
            "p95_us": 0.0,
            "host_median_us": 0.0,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 0,
        },
    ],
}

MANIFEST = {
    "manifest_hash": "deadbeef00112233",
    "candidates": [
        {
            "stable_name": "mmq:native:v1",
            "family": "mmq",
            "source_class": "native_wrapper",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": True,
            "deterministic": True,
            "config": {"j": 8},
        },
        {
            "stable_name": "mmq:generated:j4",
            "family": "mmq",
            "source_class": "new_generated_variant",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": False,
            "deterministic": True,
            "config": {"j": 4},
        },
        {
            "stable_name": "mmf:native:v1",
            "family": "mmf",
            "source_class": "native_wrapper",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": True,
            "deterministic": True,
            "config": {},
        },
        {
            "stable_name": "mmf:generated:nw4",
            "family": "mmf",
            "source_class": "new_generated_variant",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": False,
            "deterministic": True,
            "config": {"nwarps": 4},
        },
    ],
}

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"


# ------------------------------------------------------------------ helpers

def make_jsonl(*records) -> Path:
    p = tempfile.mktemp(suffix=".jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            if isinstance(rec, str):
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return Path(p)


class TempDB:
    """Temp SQLite DB initialized with project schema."""

    def __init__(self):
        self._dir = tempfile.TemporaryDirectory(prefix="bigcherry_test_")
        self.db_path = Path(self._dir.name) / "test.sqlite"
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            conn.commit()
        finally:
            conn.close()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._dir.cleanup()


def populate_db(db: TempDB) -> Path:
    """Load tuning results into the DB and return the measurements JSONL path."""
    meas_path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED)
    manifest_path = tempfile.mktemp(suffix=".json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f)

    inventory.load_measurements(
        meas_path, db.db_path, SCHEMA_PATH, manifest_path=Path(manifest_path),
    )
    os.unlink(manifest_path)
    return meas_path


def _insert_screen_duplicate(db: TempDB) -> None:
    """Clone an existing 'final'-stage measurement row as a 'screen'-stage
    duplicate with a suspiciously faster median (fewer samples) -- the
    real shape sql/dispatch-db.sql's UNIQUE constraint allows and the
    exact shape that used to inflate report.py's counts/comparisons."""
    row = db.query(
        "SELECT build_id, hardware_id, signature_id, dispatch_digest, "
        "candidate_id, run_id, objective, accepted, samples, median_us, "
        "gpu_mad_us, p95_us, host_median_us, workspace_bytes, effective_us "
        "FROM measurement WHERE stage = 'final' LIMIT 1"
    )[0]
    db_conn = sqlite3.connect(str(db.db_path))
    try:
        db_conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, "
            "dispatch_digest, candidate_id, run_id, objective, stage, "
            "accepted, samples, median_us, gpu_mad_us, p95_us, "
            "host_median_us, workspace_bytes, effective_us) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, 'screen', ?, 1, 0.001, 0.0, 0.001, 0.0, ?, 0.001)",
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6],
             row[7], row[13]),
        )
        db_conn.commit()
    finally:
        db_conn.close()


# ------------------------------------------------------------------ tests

class TestReadMeasurementsJSONL(unittest.TestCase):
    """Parse .measurements.jsonl files."""

    def test_parse_results_only(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED)
        try:
            results = _report.read_measurements_jsonl(path)
        finally:
            os.unlink(path)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["winner"], "mmq:native:v1")
        self.assertEqual(results[1]["winner"], "mmf:generated:nw4")

    def test_header_not_included(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)
        try:
            results = _report.read_measurements_jsonl(path)
        finally:
            os.unlink(path)

        # Header line has kind="header", not "result" — should be excluded.
        self.assertEqual(len(results), 1)

    def test_truncated_line_ignored(self):
        # A genuinely torn write has NO trailing newline on the final line
        # -- make_jsonl always appends one, so this writes the file
        # directly to preserve that real-world distinction (gpt-dev-agent
        # review round 2, 2026-08-31: keepends() makes this matter).
        path = Path(tempfile.mktemp(suffix=".jsonl"))
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(TUNING_HEADER, separators=(",", ":")) + "\n")
            f.write(json.dumps(TUNING_RESULT_NATIVE, separators=(",", ":")) + "\n")
            f.write('{"kind":"result","dispatch":"x')  # truncated, no trailing newline
        try:
            results = _report.read_measurements_jsonl(path)
        finally:
            os.unlink(path)

        self.assertEqual(len(results), 1)

    def test_newline_terminated_corrupt_final_line_raises_not_tolerated(self):
        """gpt-dev-agent review round 2, 2026-08-31: a fully-written
        (newline-terminated) but corrupt final record is NOT a torn write
        and must not be silently tolerated the way a genuine truncation is."""
        path = make_jsonl(
            TUNING_HEADER,
            TUNING_RESULT_NATIVE,
            '{"kind":"result","dispatch":"x',  # corrupt, but make_jsonl adds \n
        )
        try:
            with self.assertRaises(_report.ReportError):
                _report.read_measurements_jsonl(path)
        finally:
            os.unlink(path)

    def test_interior_corruption_raises_instead_of_silently_dropping(self):
        """gpt-dev-agent review, 2026-08-31: a malformed line that is NOT
        the final line used to be silently skipped -- valid A, corrupt
        interior B, valid C would report A and C with no indication B was
        lost. Only a truncated FINAL line is tolerable."""
        path = make_jsonl(
            TUNING_HEADER,
            TUNING_RESULT_NATIVE,
            '{"kind":"result","dispatch":"corrupt-interior-line',  # NOT the last line
            TUNING_RESULT_IMPROVED,
        )
        try:
            with self.assertRaises(_report.ReportError):
                _report.read_measurements_jsonl(path)
        finally:
            os.unlink(path)


class TestFmtUs(unittest.TestCase):
    """Formatting helper for microsecond values."""

    def test_none(self):
        self.assertEqual(_report._fmt_us(None), "-")

    def test_sub_microsecond(self):
        self.assertEqual(_report._fmt_us(0.5), "0.500")

    def test_one_us(self):
        self.assertEqual(_report._fmt_us(1.234), "1.23")

    def test_large(self):
        self.assertEqual(_report._fmt_us(123.456), "123.46")


class TestReadMeasurementsSQLite(unittest.TestCase):
    """Reconstruct result dicts from SQLite (round-trip with load_measurements)."""

    def test_round_trip(self):
        with TempDB() as db:
            meas_path = populate_db(db)

            # Read back via report
            results = _report.read_measurements_sqlite(db.db_path)
        os.unlink(meas_path)

        self.assertEqual(len(results), 2)

        # First result (dispatch "e"*32)
        e_result = [r for r in results if r["dispatch"] == "e" * 32][0]
        self.assertEqual(e_result["winner"], "mmq:native:v1")
        self.assertEqual(len(e_result["candidates"]), 3)

        # Check candidate statuses are mapped from reject_reason
        statuses = {c["name"]: c["status"] for c in e_result["candidates"]}
        self.assertEqual(statuses["mmq:native:v1"], "ok")
        self.assertEqual(statuses["mmq:generated:j4"], "ok")
        # architecture-rejected → status "architecture"
        self.assertEqual(statuses["mmq:generated:j8"], "architecture")

    def test_dispatch_filter(self):
        with TempDB() as db:
            meas_path = populate_db(db)
            results = _report.read_measurements_sqlite(db.db_path, dispatch_filter="f" * 32)
        os.unlink(meas_path)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["dispatch"], "f" * 32)

    def test_dispatch_not_found(self):
        with TempDB() as db:
            meas_path = populate_db(db)
            results = _report.read_measurements_sqlite(db.db_path, dispatch_filter="9" * 32)
        os.unlink(meas_path)

        self.assertEqual(results, [])

    def test_screen_stage_duplicate_excluded_from_default_final_view(self):
        """gpt-dev-agent review, 2026-08-31: sql/dispatch-db.sql makes stage
        (screen|final) part of measurement's UNIQUE constraint -- the same
        candidate can legitimately have both a cheap early screening row and
        the authoritative final row. Reading without a stage filter used to
        emit BOTH as separate candidate rows for one dispatch, so a
        fewer-samples screening result could sort as "best" purely from
        having been measured on far less evidence."""
        with TempDB() as db:
            meas_path = populate_db(db)
            _insert_screen_duplicate(db)

            default_results = _report.read_measurements_sqlite(db.db_path)
            all_stage_results = _report.read_measurements_sqlite(db.db_path, stage=None)
        os.unlink(meas_path)

        default_dispatch = [r for r in default_results if r["dispatch"] == "e" * 32][0]
        all_dispatch = [r for r in all_stage_results if r["dispatch"] == "e" * 32][0]
        # The default (stage="final") view must be unaffected by the
        # screen-stage duplicate; the explicit stage=None view sees it.
        self.assertEqual(len(default_dispatch["candidates"]), 3)
        self.assertEqual(len(all_dispatch["candidates"]), 4)
        self.assertTrue(all(c["median_us"] != 0.001 for c in default_dispatch["candidates"]))


class TestCmdSignatures(unittest.TestCase):
    """Per-signature detail tables via CLI."""

    def test_no_source_errors(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)
        try:
            import argparse
            args = argparse.Namespace(
                measurements=str(path), database=None,
                dispatch=None, limit=0, json=False,
            )
            status = _report.cmd_signatures(args)
            self.assertEqual(status, 0)
        finally:
            os.unlink(path)

    def test_json_output(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)
        try:
            import argparse
            args = argparse.Namespace(
                measurements=str(path), database=None,
                dispatch=None, limit=0, json=True,
            )
            status = _report.cmd_signatures(args)
            self.assertEqual(status, 0)
        finally:
            os.unlink(path)

    def test_limit(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED)
        try:
            import argparse
            args = argparse.Namespace(
                measurements=str(path), database=None,
                dispatch=None, limit=1, json=False,
            )
            status = _report.cmd_signatures(args)
            self.assertEqual(status, 0)
        finally:
            os.unlink(path)


class TestReadTuningOverview(unittest.TestCase):
    """gpt-dev-agent review round 2, 2026-08-31: fixing
    read_measurements_sqlite's stage filtering alone was not enough --
    read_tuning_overview() ran its OWN independent, unfiltered
    `measurement` queries for pipeline generated/measured/rejection
    counts, so a screen-stage duplicate row still inflated those."""

    def test_screen_stage_duplicate_does_not_inflate_pipeline_counts(self):
        with TempDB() as db:
            meas_path = populate_db(db)
            before = _report.read_tuning_overview(db.db_path)
            _insert_screen_duplicate(db)
            after = _report.read_tuning_overview(db.db_path)
        os.unlink(meas_path)

        self.assertEqual(before["pipeline"], after["pipeline"])

    def test_stage_none_opts_into_seeing_the_duplicate(self):
        with TempDB() as db:
            meas_path = populate_db(db)
            before = _report.read_tuning_overview(db.db_path, stage=None)
            _insert_screen_duplicate(db)
            after = _report.read_tuning_overview(db.db_path, stage=None)
        os.unlink(meas_path)

        self.assertEqual(after["pipeline"]["generated"], before["pipeline"]["generated"] + 1)


class TestCmdSummary(unittest.TestCase):
    """Aggregate statistics via CLI."""

    def test_no_source_errors(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED)
        try:
            import argparse
            args = argparse.Namespace(measurements=str(path), database=None, json=False)
            status = _report.cmd_summary(args)
            self.assertEqual(status, 0)
        finally:
            os.unlink(path)

    def test_empty_results(self):
        path = make_jsonl(TUNING_HEADER)  # header only, no results
        try:
            import argparse
            args = argparse.Namespace(measurements=str(path), database=None, json=False)
            status = _report.cmd_summary(args)
            self.assertEqual(status, 0)  # should print "no tuning results found"
        finally:
            os.unlink(path)

    def test_no_source_given(self):
        import argparse
        args = argparse.Namespace(measurements=None, database=None, json=False)
        status = _report.cmd_summary(args)
        self.assertEqual(status, 2)


class TestCmdFamilies(unittest.TestCase):
    """Cross-family comparison for one dispatch digest."""

    def test_dispatch_found(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)
        try:
            import argparse
            args = argparse.Namespace(
                measurements=str(path), database=None,
                dispatch="e" * 32, json=False,
            )
            status = _report.cmd_families(args)
            self.assertEqual(status, 0)
        finally:
            os.unlink(path)

    def test_dispatch_not_found(self):
        path = make_jsonl(TUNING_HEADER, TUNING_RESULT_NATIVE)
        try:
            import argparse
            args = argparse.Namespace(
                measurements=str(path), database=None,
                dispatch="9" * 32, json=False,
            )
            status = _report.cmd_families(args)
            self.assertEqual(status, 1)
        finally:
            os.unlink(path)

    def test_dispatch_not_found_via_database_does_not_crash(self):
        """gpt-dev-agent review, 2026-08-31: the JSONL path already returned
        a clean exit 1 for an unknown digest; the database path skipped
        that check entirely and crashed on results[0] (IndexError)."""
        with TempDB() as db:
            meas_path = populate_db(db)
            import argparse
            args = argparse.Namespace(
                measurements=None, database=str(db.db_path),
                dispatch="9" * 32, json=False,
            )
            status = _report.cmd_families(args)
        os.unlink(meas_path)
        self.assertEqual(status, 1)

    def test_no_dispatch_given(self):
        import argparse
        args = argparse.Namespace(
            measurements=None, database=None,
            dispatch=None, json=False,
        )
        status = _report.cmd_families(args)
        self.assertEqual(status, 2)


class TestCmdHot(unittest.TestCase):
    """Top-N signatures by call count."""

    def test_no_database_given(self):
        import argparse
        args = argparse.Namespace(database=None, limit=20, json=False)
        status = _report.cmd_hot(args)
        self.assertEqual(status, 2)

    def test_empty_observation_table(self):
        with TempDB() as db:
            import argparse
            args = argparse.Namespace(
                database=str(db.db_path), limit=10, json=False,
            )
            status = _report.cmd_hot(args)
            self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
