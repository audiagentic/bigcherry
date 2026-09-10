from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bigcherry.environment_inputs import (
    EnvironmentInput,
    EnvironmentInputError,
    EnvironmentInputSet,
    action_input,
    artifact_input,
    observation_input,
)


class EnvironmentInputTests(unittest.TestCase):
    def test_observation_round_trips_and_is_deterministic(self) -> None:
        a = observation_input(name="driver", scope="run", values={"version": "6.8.12"})
        b = observation_input(name="driver", scope="run", values={"version": "6.8.12"})
        self.assertEqual(a.digest(), b.digest())

    def test_rejects_bad_name(self) -> None:
        with self.assertRaises(EnvironmentInputError):
            observation_input(name="Bad Name", scope="run", values={})

    def test_rejects_secret_pattern_value(self) -> None:
        with self.assertRaises(EnvironmentInputError):
            observation_input(name="env", scope="run", values={"API_TOKEN": "x"})

    def test_rejects_invalid_scope(self) -> None:
        with self.assertRaises(EnvironmentInputError):
            EnvironmentInput(kind="observation", name="x", scope="nope")  # type: ignore[arg-type]

    def test_artifact_input_keys_by_role_and_content_digest_not_path(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "solutions.csv"
            path.write_text("a,b,c\n1,2,3\n")
            expected = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
            item = artifact_input(name="rocblas-gemm-tune", logical_role="gemm-solutions", path=path)
            self.assertEqual(item.files, (("gemm-solutions", expected),))
            self.assertNotIn(str(path), item.canonical_document().values())

    def test_artifact_input_rejects_path_as_logical_role(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "solutions.csv"
            path.write_text("x")
            with self.assertRaises(EnvironmentInputError):
                artifact_input(name="n", logical_role="a/b", path=path)

    def test_reversible_action_requires_restored_outcome(self) -> None:
        pre = observation_input(name="aspm-pre", scope="evidence_only", values={"policy": "default"})
        with self.assertRaises(EnvironmentInputError):
            EnvironmentInput(
                kind="action", name="aspm-write", scope="run",
                values=(("policy", "powersave"),), reversible=True,
                pre=pre, post=None, restored=None,
            )

    def test_action_input_records_pre_post_restored(self) -> None:
        pre = observation_input(name="aspm-pre", scope="evidence_only", values={"policy": "default"})
        post = observation_input(name="aspm-post", scope="evidence_only", values={"policy": "default"})
        action = action_input(
            name="aspm-write", scope="run", values={"policy": "powersave"},
            pre=pre, post=post, restored=True,
        )
        doc = action.canonical_document()
        self.assertEqual(doc["restored"], True)
        self.assertIsNotNone(doc["pre"])
        self.assertIsNotNone(doc["post"])

    def test_set_rejects_duplicate_names(self) -> None:
        a = observation_input(name="dup", scope="run", values={"x": "1"})
        b = observation_input(name="dup", scope="build", values={"x": "2"})
        with self.assertRaises(EnvironmentInputError):
            EnvironmentInputSet(inputs=(a, b))

    def test_scope_digests_are_partitioned(self) -> None:
        build_only = observation_input(name="compiler", scope="build", values={"v": "1"})
        run_only = observation_input(name="driver", scope="run", values={"v": "1"})
        both = observation_input(name="rocm", scope="both", values={"v": "1"})
        evidence_only = observation_input(name="hostname", scope="evidence_only", values={"v": "brutus"})
        eset = EnvironmentInputSet(inputs=(build_only, run_only, both, evidence_only))

        build_digest = eset.build_digest()
        run_digest = eset.run_digest()
        self.assertNotEqual(build_digest, run_digest)

        # Removing an evidence_only input must not change either scoped digest.
        eset_without_evidence = EnvironmentInputSet(inputs=(build_only, run_only, both))
        self.assertEqual(build_digest, eset_without_evidence.build_digest())
        self.assertEqual(run_digest, eset_without_evidence.run_digest())

        # Changing the build-only input must not change the run digest.
        changed_build = observation_input(name="compiler", scope="build", values={"v": "2"})
        eset_changed = EnvironmentInputSet(inputs=(changed_build, run_only, both, evidence_only))
        self.assertNotEqual(build_digest, eset_changed.build_digest())
        self.assertEqual(run_digest, eset_changed.run_digest())

    def test_to_document_is_deterministic_regardless_of_input_order(self) -> None:
        a = observation_input(name="a", scope="run", values={"x": "1"})
        b = observation_input(name="b", scope="build", values={"x": "1"})
        doc1 = EnvironmentInputSet(inputs=(a, b)).to_document()
        doc2 = EnvironmentInputSet(inputs=(b, a)).to_document()
        self.assertEqual(doc1, doc2)


if __name__ == "__main__":
    unittest.main()
