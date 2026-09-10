"""VA25: the ONE structural seam every server-driven measurement lane must
go through, so execution-identity attestation cannot be omitted by a new
lane forgetting to call it -- the exact failure mode that let RD73's three
server lanes (run_rd73_mtp_server_lane, run_rd73_decode_control_lane,
run_rd73_resource_burst_session) ship unattested despite the underlying
attestation.py machinery (ExecutionIdentity/compare_execution_identity/
parse_llama_server_attestation) already existing.

WHY A WRAPPER, NOT A CALL SITE CONVENTION. dev-gpt-agent review (VA25
follow-up consultation, 2026-09-08): three individual require_execution_
identity() call sites just moves the same "field exists, nothing joins it
to the path that needs it" defect one level down -- the fourth lane
someone writes next will forget too. A structural seam that MUST be used
to get a measurable server at all closes the class of gap, not one
instance of it.

WHY NOT IN tuning/server_runner.py::ServerRunner ITSELF. ServerRunner is
also used for discovery/smoke/non-measured operation (see tuning/
workflow.py's record/tune stages) where mandatory attestation would be
unwanted policy bleeding into a general-purpose lifecycle primitive.
AttestedServerSession composes ServerRunner rather than replacing it.

REAL BLOCKER FOUND (not just "nobody called it yet"), 2026-09-08: RD73's
lanes pass ``-sm tensor --fit off`` (required for that model/topology,
patches/1233's own README documents the real hardware crash without
--fit off). That combination skips the device-fitting code path that
emits llama-server's "using device ROCm0 (...)" line -- attestation.py's
own module comment documents this exact caveat, and states server lanes
"cannot attest" until fixed. The fallback evidence (per-layer "assigned
to device" lines, LLAMA_LOG_DEBUG in src/llama-model.cpp) exists but is
filtered below llama-server's default verbosity threshold. This session
forces --verbosity 5 on every attested launch so that fallback evidence
is always present, rather than trusting each RD73 call site to remember
to add it (the log volume is bounded per session, since RD73's own
design launches one fresh server per single measured/warmup request --
see run_rd73_mtp_server_lane's docstring).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .attestation import (
    ExecutionAttestation,
    ExecutionIdentity,
    parse_llama_server_attestation,
    require_execution_identity,
)
from ..tuning.server_runner import ServerRunner


class AttestedServerSession:
    """A ServerRunner that refuses to become usable for measurement unless
    its actual execution identity (backend/device-count/architecture,
    optionally device locators) matches what the caller declared it
    expects. Use exactly like ServerRunner as a context manager; the
    measured-request interface only becomes meaningful once ``__enter__``
    has returned, which it does not do if attestation fails.

    Lifecycle (dev-gpt-agent review): attest ONCE per server process,
    after health succeeds and before the first measured request -- never
    per-request. A process that attested correctly at startup does not
    need to re-attest for every subsequent completion; if the environment
    could change mid-process that is a different, currently unaddressed
    hazard (VA25 scopes this to "the process that ran is the process we
    expected", not "the process is stable device-migration-free
    throughout its lifetime").
    """

    def __init__(
        self,
        *,
        binary: Path,
        model: Path,
        expected: ExecutionIdentity,
        log_path: Path,
        extra_args: tuple[str, ...] = (),
        env_overrides: dict[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
        host: str = "127.0.0.1",
        port: int | None = None,
        shutdown_method: str = "http",
        architecture_by_locator: Mapping[str, str] | None = None,
    ) -> None:
        self._log_path = Path(log_path)
        self._expected = expected
        self._architecture_by_locator = architecture_by_locator
        self.attestation: ExecutionAttestation | None = None
        self._runner = ServerRunner(
            binary=binary,
            model=model,
            host=host,
            port=port,
            # See module docstring: forced unconditionally, not left to the
            # caller, because the whole point of this class is that a lane
            # cannot forget the one thing that makes attestation possible.
            extra_args=(*extra_args, "--verbosity", "5"),
            env_overrides=env_overrides,
            env_unset=env_unset,
            log_path=self._log_path,
            shutdown_method=shutdown_method,
        )

    @property
    def host(self) -> str:
        return self._runner.host

    @property
    def port(self) -> int:
        return self._runner.port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "AttestedServerSession":
        self._runner.launch()
        try:
            self._runner.wait_healthy()
            observed = parse_llama_server_attestation(
                self._log_path.read_text(encoding="utf-8", errors="replace"),
                architecture_by_locator=self._architecture_by_locator,
            )
            require_execution_identity(
                self._expected, observed,
                context=f"attested server session ({self._log_path.name})",
            )
            self.attestation = observed
        except BaseException:
            # Same reasoning as ServerRunner.__enter__ itself (HI143):
            # without this, a health/attestation failure leaves the
            # just-launched process -- and its real GPU allocation --
            # running forever, since __exit__ is never called when
            # __enter__ raises.
            self._runner.shutdown()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._runner.shutdown()

    def run_completion(self, *args, **kwargs):
        return self._runner.run_completion(*args, **kwargs)

    def post_json(self, *args, **kwargs):
        return self._runner.post_json(*args, **kwargs)
