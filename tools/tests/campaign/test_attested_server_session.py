"""VA25: AttestedServerSession is the structural seam every server-driven
measurement lane must go through -- these tests prove attestation cannot
be silently skipped, that it fails closed on mismatch, and that it forces
the verbosity level the real fallback attestation evidence needs (see
server_execution.py's own module docstring for the real hardware finding
this last point is based on)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import attestation as att  # noqa: E402
from bigcherry.experiment import server_execution as se  # noqa: E402


_GOOD_LOG = (
    "I load_tensors: layer   0 assigned to device ROCm0, is_swa = 0\n"
    "D load_tensors: layer   1 assigned to device ROCm0, is_swa = 0\n"
)
_WRONG_ARCH_LOG = _GOOD_LOG  # architecture comes from architecture_by_locator; see test


class AttestedServerSessionTests(unittest.TestCase):
    def _session(self, *, log_text: str, expected: att.ExecutionIdentity, tmp_path: Path):
        log_path = tmp_path / "server.log"
        log_path.write_text(log_text, encoding="utf-8")
        session = se.AttestedServerSession(
            binary=Path("/fake/bin/llama-server"), model=Path("/fake/model.gguf"),
            expected=expected, log_path=log_path,
        )
        return session, log_path

    def test_forces_verbosity_5_regardless_of_caller_args(self):
        with patch.object(se, "ServerRunner") as fake_runner_cls:
            se.AttestedServerSession(
                binary=Path("/fake/bin/llama-server"), model=Path("/fake/model.gguf"),
                expected=att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",)),
                log_path=Path("/fake/server.log"), extra_args=("--parallel", "1"),
            )
        _, kwargs = fake_runner_cls.call_args
        self.assertEqual(kwargs["extra_args"], ("--parallel", "1", "--verbosity", "5"))

    def test_matching_identity_enters_cleanly_and_records_attestation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            expected = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",))
            session, _ = self._session(log_text=_GOOD_LOG, expected=expected, tmp_path=tmp_path)
            fake_runner = MagicMock()
            session._runner = fake_runner
            with patch.object(
                se, "parse_llama_server_attestation",
                return_value=att.ExecutionAttestation(
                    backend="ROCm", devices=(att.ObservedDevice(architecture="gfx1100", locator=None),),
                ),
            ):
                with session:
                    pass
            fake_runner.launch.assert_called_once()
            fake_runner.wait_healthy.assert_called_once()
            fake_runner.shutdown.assert_called_once()
            self.assertIsNotNone(session.attestation)
            self.assertEqual(session.attestation.backend, "ROCm")

    def test_mismatched_identity_raises_and_shuts_down(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            expected = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1201",))
            session, _ = self._session(log_text=_GOOD_LOG, expected=expected, tmp_path=tmp_path)
            fake_runner = MagicMock()
            session._runner = fake_runner
            with patch.object(
                se, "parse_llama_server_attestation",
                return_value=att.ExecutionAttestation(
                    backend="ROCm", devices=(att.ObservedDevice(architecture="gfx1100", locator=None),),
                ),
            ):
                with self.assertRaises(att.AttestationError):
                    with session:
                        pass
            # Never usable for measurement, and the process must not leak.
            fake_runner.shutdown.assert_called_once()

    def test_missing_attestation_raises_and_shuts_down(self):
        # The CPU-fallback case this whole item exists for: output that
        # parses to no positive device evidence at all.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            expected = att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",))
            session, _ = self._session(log_text="", expected=expected, tmp_path=tmp_path)
            fake_runner = MagicMock()
            session._runner = fake_runner
            with patch.object(se, "parse_llama_server_attestation", return_value=None):
                with self.assertRaises(att.AttestationError):
                    with session:
                        pass
            fake_runner.shutdown.assert_called_once()

    def test_health_check_failure_shuts_down_and_never_reaches_attestation(self):
        with patch.object(se, "ServerRunner") as fake_runner_cls:
            fake_runner = fake_runner_cls.return_value
            fake_runner.wait_healthy.side_effect = RuntimeError("server never came up")
            session = se.AttestedServerSession(
                binary=Path("/fake/bin/llama-server"), model=Path("/fake/model.gguf"),
                expected=att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",)),
                log_path=Path("/fake/server.log"),
            )
            with patch.object(se, "parse_llama_server_attestation") as fake_parse:
                with self.assertRaises(RuntimeError):
                    with session:
                        pass
                fake_parse.assert_not_called()
            fake_runner.shutdown.assert_called_once()

    def test_run_completion_and_post_json_delegate_to_the_wrapped_runner(self):
        with patch.object(se, "ServerRunner") as fake_runner_cls:
            fake_runner = fake_runner_cls.return_value
            session = se.AttestedServerSession(
                binary=Path("/fake/bin/llama-server"), model=Path("/fake/model.gguf"),
                expected=att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",)),
                log_path=Path("/fake/server.log"),
            )
            session.run_completion("hello", n_predict=8)
            fake_runner.run_completion.assert_called_once_with("hello", n_predict=8)
            session.post_json("/v1/x", {"a": 1})
            fake_runner.post_json.assert_called_once_with("/v1/x", {"a": 1})

    def test_host_port_base_url_delegate_to_the_wrapped_runner(self):
        with patch.object(se, "ServerRunner") as fake_runner_cls:
            fake_runner = fake_runner_cls.return_value
            fake_runner.host = "127.0.0.1"
            fake_runner.port = 12345
            session = se.AttestedServerSession(
                binary=Path("/fake/bin/llama-server"), model=Path("/fake/model.gguf"),
                expected=att.ExecutionIdentity(backend="ROCm", architectures=("gfx1100",)),
                log_path=Path("/fake/server.log"),
            )
            self.assertEqual(session.host, "127.0.0.1")
            self.assertEqual(session.port, 12345)
            self.assertEqual(session.base_url, "http://127.0.0.1:12345")


if __name__ == "__main__":
    unittest.main()
