from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.jobs.executor import ResourceRequest
from bigcherry.jobs.operations import (
    ArtifactBinding,
    OperationError,
    OperationSpec,
    begin_operation,
    bind_artifact,
    durable_state,
    execution_hash,
    operation_spec_hash,
    publish_result,
    rehydrate_succeeded,
)


def spec(*, command_value: str = "a") -> OperationSpec:
    return OperationSpec(
        operation_id="build",
        kind="build",
        command_semantics={"mode": command_value},
        environment_semantics=(("A", "1"),),
        resources=ResourceRequest(2, None, "build", None, 60),
        dependencies=("prepare",),
        declared_outputs=("binary",),
    )


def input_binding(content: str = "a" * 64) -> ArtifactBinding:
    return ArtifactBinding("source", "source.bin", "b" * 64, content, 1)


class OperationTests(unittest.TestCase):
    def test_spec_and_execution_hash_have_distinct_change_domains(self):
        base = spec()
        changed = spec(command_value="b")
        self.assertNotEqual(operation_spec_hash(base), operation_spec_hash(changed))
        one = execution_hash(operation_spec_hash(base), (input_binding("1" * 64),))
        two = execution_hash(operation_spec_hash(base), (input_binding("2" * 64),))
        self.assertNotEqual(one, two)
        self.assertEqual(operation_spec_hash(base), operation_spec_hash(spec()))

    def test_success_rehydrates_only_when_inputs_and_output_bytes_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "out.bin").write_bytes(b"original")
            current = spec()
            inputs = (input_binding(),)
            begin_operation(root, current, inputs)
            output = bind_artifact(root, "binary", "out.bin", descriptor={"kind": "exe"})
            publish_result(
                root,
                spec=current,
                inputs=inputs,
                state="succeeded",
                returncode=0,
                outputs=(output,),
            )
            self.assertIsNotNone(rehydrate_succeeded(root, current, inputs))
            self.assertIsNone(
                rehydrate_succeeded(root, current, (input_binding("c" * 64),))
            )
            (root / "out.bin").write_bytes(b"tampered")
            self.assertIsNone(rehydrate_succeeded(root, current, inputs))

    def test_running_without_terminal_result_is_interrupted_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            begin_operation(root, spec(), (input_binding(),))
            self.assertEqual(durable_state(root), "interrupted")
            self.assertIsNone(rehydrate_succeeded(root, spec(), (input_binding(),)))

    def test_success_requires_declared_outputs_and_result_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = spec()
            inputs = (input_binding(),)
            begin_operation(root, current, inputs)
            with self.assertRaises(OperationError):
                publish_result(
                    root,
                    spec=current,
                    inputs=inputs,
                    state="succeeded",
                    returncode=0,
                    outputs=(),
                )
            (root / "out.bin").write_bytes(b"ok")
            output = bind_artifact(root, "binary", "out.bin")
            publish_result(
                root,
                spec=current,
                inputs=inputs,
                state="succeeded",
                returncode=0,
                outputs=(output,),
            )
            with self.assertRaises(OperationError):
                publish_result(
                    root,
                    spec=current,
                    inputs=inputs,
                    state="succeeded",
                    returncode=0,
                    outputs=(output,),
                )

    def test_artifact_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root.parent / "outside-operation-test.bin"
            outside.write_bytes(b"x")
            try:
                with self.assertRaises(OperationError):
                    bind_artifact(root, "x", "../outside-operation-test.bin")
            finally:
                outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
