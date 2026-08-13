"""Current tuning journal durability and recovery tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import tune_journal  # noqa: E402


class TuneJournalTests(unittest.TestCase):
    def writer(self, path: Path) -> tune_journal.JournalWriter:
        return tune_journal.JournalWriter(
            path, "experiment", "a" * 40, "b" * 32, "c" * 32,
            durability_mode="durable_each",
        )

    def test_complete_roundtrip_and_deterministic_compaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.result({"kind": "result", "dispatch": "b", "generation": 1, "winner": "x"})
            writer.result({"kind": "result", "dispatch": "a", "generation": 1, "winner": "y"})
            writer.complete({"results": 2})
            recovered = tune_journal.read_current(writer.path)
            self.assertTrue(recovered["complete"])
            header = {"kind": "header", "schema_version": 1}
            first, second = root / "first.jsonl", root / "second.jsonl"
            one = tune_journal.compact(writer.path, first, header)
            two = tune_journal.compact(writer.path, second, header)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["content_hash"], two["content_hash"])
            self.assertIn(b'"dispatch":"a"', first.read_bytes().splitlines()[1])

    def test_compact_preserves_start_provenance_for_replay_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.result({"kind": "result", "dispatch": "a", "winner": "native"})
            writer.complete()
            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header", "schema_version": 1})
            header = json.loads(output.read_bytes().splitlines()[0])
            self.assertEqual(header["source_revision"], "a" * 40)
            self.assertEqual(header["manifest_hash"], "b" * 32)
            self.assertEqual(header["hardware_digest"], "c" * 32)

    def test_truncated_and_corrupt_tail_recover_acknowledged_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = self.writer(path)
            writer.result({"kind": "result", "dispatch": "a"})
            writer.acknowledge()
            writer._handle.write(b'{"schema_version":1')
            writer._handle.close()
            recovered = tune_journal.read_current(path)
            self.assertTrue(recovered["corrupt_tail"])
            self.assertEqual(recovered["last_durable_sequence"], 2)
            self.assertIn("truncated", recovered["corruption"])

    def test_unknown_schema_and_interior_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            event = tune_journal._checked_event({
                "schema_version": 2, "experiment_id": "x", "sequence": 1,
                "kind": "start", "payload": {},
            })
            path.write_bytes(tune_journal.canonical(event) + b"\n")
            with self.assertRaisesRegex(tune_journal.JournalError, "external_conversion"):
                tune_journal.read_current(path)
            path.write_bytes(b"not-json\n{}\n")
            with self.assertRaisesRegex(tune_journal.JournalError, "line 1"):
                tune_journal.read_current(path)

    def test_compact_accepts_string_payload_from_cpp_writer(self):
        # hip-autotune-journal.cpp (HI48 Phase 1) stores a "result" event's
        # payload as an opaque JSON *string*, not a nested object -- see its
        # own header comment for why. compact() must transparently accept
        # either shape; this simulates the C++ wire format by writing a raw
        # string payload the same way JournalWriter.append() would if a
        # caller bypassed the dict-only .result() convenience method.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            payload_str = json.dumps({"kind": "result", "dispatch": "cpp-a",
                                      "generation": 1, "winner": "z",
                                      "signature": "d" * 32,
                                      "hardware": "e" * 32})
            writer.append("result", payload_str)
            writer.complete()
            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header"})
            result = json.loads(output.read_bytes().splitlines()[1])
            self.assertEqual(result["dispatch"], "cpp-a")
            self.assertEqual(result["winner"], "z")
            self.assertEqual(result["signature"], "d" * 32)
            self.assertEqual(result["hardware"], "e" * 32)

    def test_compact_recovers_unambiguous_legacy_native_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.append("result", json.dumps({
                "kind": "result", "dispatch": "native-a",
                "winner": "mmvq:native:v1",
            }))
            writer.complete()
            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header"})
            result = json.loads(output.read_bytes().splitlines()[1])
            self.assertEqual(result["native"], "mmvq:native:v1")
            self.assertEqual(result["promotion_status"], "native")

    def test_compact_rejects_malformed_string_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.append("result", "not valid json")
            writer.complete()
            with self.assertRaisesRegex(tune_journal.JournalError, "not valid JSON"):
                tune_journal.compact(writer.path, root / "out.jsonl", {"kind": "header"})

    def test_duplicate_dispatch_uses_generation_then_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.result({"kind": "result", "dispatch": "a", "generation": 2, "winner": "new"})
            writer.result({"kind": "result", "dispatch": "a", "generation": 1, "winner": "late-old"})
            writer.complete()
            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header"})
            result = json.loads(output.read_bytes().splitlines()[1])
            self.assertEqual(result["winner"], "new")

    def test_trailing_attempt_is_compactable_and_surfaced_as_last_attempt(self):
        """A crash mid-candidate leaves a trailing 'attempt' with no 'result'
        after it. Every prior result must still recover exactly as if the run
        had been killed cleanly between candidates -- can_compact must not
        regress just because this run's terminal event now names a suspect
        instead of being silent about it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.result({"kind": "result", "dispatch": "a", "generation": 1, "winner": "ok"})
            writer.append("attempt", {"candidate": "mmq:q6_k:j112:...:v1", "t_us": 123})
            writer._handle.close()
            recovered = tune_journal.read_current(writer.path)
            self.assertTrue(recovered["can_compact"])
            self.assertFalse(recovered["complete"])
            self.assertFalse(recovered["interrupted"])
            self.assertEqual(recovered["last_attempt"]["candidate"], "mmq:q6_k:j112:...:v1")

            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header"})
            result = json.loads(output.read_bytes().splitlines()[1])
            self.assertEqual(result["dispatch"], "a")

    def test_attempt_with_cpp_string_payload_is_parsed(self):
        """The C++ writer stores payload as an escaped JSON string (same
        convention as 'result', see append_result/append_attempt in
        hip-autotune-journal.cpp), not a nested object."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.append("attempt", json.dumps({"candidate": "cpp-candidate"}))
            writer._handle.close()
            recovered = tune_journal.read_current(writer.path)
            self.assertEqual(recovered["last_attempt"]["candidate"], "cpp-candidate")

    def test_attempt_only_journal_without_any_result_still_compacts_to_empty(self):
        """The narrow documented gap: the very first candidate of a run has
        no attempt coverage until a result opens the picture, but if the
        journal itself only ever reaches 'start' -> 'attempt', compaction
        must still succeed -- there is nothing to recover, not an error."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.writer(root / "events.jsonl")
            writer.append("attempt", {"candidate": "first-ever", "t_us": 1})
            writer._handle.close()
            recovered = tune_journal.read_current(writer.path)
            self.assertTrue(recovered["can_compact"])
            output = root / "measurements.jsonl"
            tune_journal.compact(writer.path, output, {"kind": "header"})
            header = json.loads(output.read_bytes().splitlines()[0])
            self.assertEqual(header["kind"], "header")
            self.assertEqual(header["source_revision"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
