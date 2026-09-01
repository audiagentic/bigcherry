"""HI138/RQ01-03: standalone, crash-isolated RCCL heterogeneous-topology
qualification harness.

Diagnostic tooling only -- see docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md
for the governing procedure. This module does not touch, select, or
influence BigCherry's production reduction-provider selection
(GGML_HIP_REDUCE_PLAN / patch 1225 stay untouched); it drives an external
``all_reduce_perf`` (RCCL Tests) binary in its own subprocess, one case per
process, so a GPU fault or hard abort in RCCL cannot take this harness (or
a sibling case) down with it.

Deliberately independent of tuning/catalog.py, tuning/inventory.py,
promotion, replay, the dispatch ABI, and the hip-autotune runtime -- this
is qualification evidence, not a tuning candidate source.

Topology identity (``RcclTopology.topology_id``) is a persistent semantic
key (ordered device architectures) and must never be derived from or
depend on a HIP ordinal, PCI BDF, GPU UUID, serial, hostname, or
``/dev/dri`` path -- those are runtime/diagnostic-only and appear (if at
all) only in ``RcclCaseResult.diagnostic_visible_devices``, which must
never be hashed into any identity.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from bigcherry.profiling.rccl_schema import RcclCompatibilityRevision
from bigcherry.profiling.rccl_schema import qualification_key

SCHEMA_VERSION = 1

# Runbook P1.6/P2.4 superset (10 states) -- gpt-agreed (session
# ses_b1c4f602d7644d44, 2026-08-29): UNSUPPORTED and DEVICE_LOST must stay
# distinct from LAUNCH_FAILURE/GPU_FAULT. UNSUPPORTED is actionable RCCL
# capability evidence (a plan RCCL itself declines, not a crash).
# DEVICE_LOST is materially different from a per-process SIGNAL/GPU_FAULT
# because it is what triggers the campaign-level safety stop/recheck path
# (RQ08's "after GPU_FAULT/SIGNAL/TIMEOUT, recheck GPU health" rule).
PASS = "pass"
WRONG_RESULT = "wrong_result"
UNSUPPORTED = "unsupported"
INIT_FAILURE = "init_failure"
LAUNCH_FAILURE = "launch_failure"
GPU_FAULT = "gpu_fault"
DEVICE_LOST = "device_lost"
SIGNAL = "signal"
TIMEOUT = "timeout"
HARNESS_FAILURE = "harness_failure"

CLASSIFICATIONS = frozenset(
    (
        PASS, WRONG_RESULT, UNSUPPORTED, INIT_FAILURE, LAUNCH_FAILURE,
        GPU_FAULT, DEVICE_LOST, SIGNAL, TIMEOUT, HARNESS_FAILURE,
    )
)

# Text markers used to disambiguate LAUNCH_FAILURE / GPU_FAULT / UNSUPPORTED
# / DEVICE_LOST / INIT_FAILURE / TIMEOUT from a bare nonzero exit code,
# checked against combined stdout+stderr. Kept narrow and literal (no regex
# surprises) -- extend only from real observed RCCL/RCCL-Tests output, never
# speculatively.
_DEVICE_LOST_MARKERS = ("GPU is lost", "amdgpu: GPU reset", "device lost")
_GPU_FAULT_MARKERS = (
    "unhandled cuda error", "HIP failure", "an illegal memory access",
    "the operation cannot be performed in the present state",
)
_UNSUPPORTED_MARKERS = (
    "not supported", "unsupported", "Unsupported"
)
# Real hardware evidence (2026-09-02, xtx_xtx homogeneous control, RCCL
# 2.30.4): "Symmetric memory is not supported. cuMemEnable 0, ..." is a
# routine NCCL_DEBUG=INFO capability-negotiation trace line printed on
# EVERY run on this hardware, successful or not -- it previously
# misclassified a clean PASS (0 wrong, correct algo/proto reported) as
# UNSUPPORTED. Same class of over-broad-substring bug this file's own
# INIT_FAILURE narrowing below already fixed once; exclude this specific
# known-benign line rather than removing the (still useful) general markers.
_UNSUPPORTED_BENIGN_MARKERS = (
    "Symmetric memory is not supported",
)
# Real hardware evidence (RQ08, xtx_r9700 Tree/LL): "ncclCommInitAll" and
# "commInit" as bare substrings are too loose -- they appear in routine
# trace lines on every run, successful or not (e.g. "ncclCommInitAll_impl
# comm ... Init COMPLETE"), and previously misclassified a real internal
# test timeout as INIT_FAILURE. Require an explicit failure phrase instead.
_INIT_FAILURE_MARKERS = (
    "ncclCommInitAll failed", "commInit failed", "initialization failed",
)
# RCCL Tests' own internal "-T <seconds>" test-level timeout (distinct from
# our outer harness's process-group-kill timeout -- this one exits the
# child process on its own with a real, non-negative returncode).
_RCCL_TEST_TIMEOUT_MARKERS = ("Test timeout",)


@dataclass(frozen=True)
class RcclTopology:
    """Persistent semantic topology identity -- ordered device
    architectures only. No HIP ordinal, PCI BDF, UUID, or hostname."""

    topology_id: str
    device_arches: tuple[str, ...]

    @property
    def rank_count(self) -> int:
        return len(self.device_arches)


@dataclass(frozen=True)
class RcclCase:
    """One qualification case: one topology, one algorithm/protocol,
    one reduction operation, run in its own process."""

    topology: RcclTopology
    element_count: int
    dtype: str = "float"
    algorithm: str = "Ring"
    protocol: str = "Simple"
    channels: int | None = None
    # Production-signature provenance (HI18's real recorded shape,
    # e.g. slice_shape=[4096,2,1,1]) -- metadata only, not part of identity.
    slice_shape: tuple[int, ...] | None = None

    @property
    def case_id(self) -> str:
        chan = f"__ch{self.channels}" if self.channels is not None else ""
        return (
            f"{self.topology.topology_id}__{self.element_count}__{self.dtype}"
            f"__{self.algorithm.lower()}__{self.protocol.lower()}{chan}"
        )

    @property
    def byte_count(self) -> int:
        itemsize = {"float": 4, "float16": 2, "float64": 8, "int32": 4}.get(
            self.dtype
        )
        if itemsize is None:
            raise ValueError(f"unknown dtype for byte-size derivation: {self.dtype!r}")
        return self.element_count * itemsize


@dataclass(frozen=True)
class RcclCaseResult:
    schema_version: int
    case_id: str
    topology_id: str
    device_arches: tuple[str, ...]
    diagnostic_visible_devices: tuple[int, ...]

    element_count: int
    dtype: str
    byte_count: int
    algorithm: str
    protocol: str
    requested_channels: int | None

    observed_algorithm: str | None
    observed_protocol: str | None
    observed_channels: int | None

    returncode: int | None
    terminating_signal: int | None
    elapsed_seconds: float
    classification: str
    correct: bool | None
    detail: str

    rccl_output_path: str
    stdout_path: str
    stderr_path: str

    # GP07: added at the end with a default so every existing positional/
    # keyword construction of this dataclass (tests included) stays valid
    # unchanged. None means "not recorded" -- callers that care about
    # cross-revision portability (GP01/GP06) must supply it, but this
    # module does not itself invent a value.
    compatibility_revision_id: str | None = None
    attempt: int | None = None
    plan_verification: str | None = None
    qualification_key: str | None = None

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification: {self.classification!r}")

    def to_json_row(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "topology_id": self.topology_id,
            "device_arches": list(self.device_arches),
            "diagnostic_visible_devices": list(self.diagnostic_visible_devices),
            "element_count": self.element_count,
            "dtype": self.dtype,
            "byte_count": self.byte_count,
            "algorithm": self.algorithm,
            "protocol": self.protocol,
            "requested_channels": self.requested_channels,
            "observed_algorithm": self.observed_algorithm,
            "observed_protocol": self.observed_protocol,
            "observed_channels": self.observed_channels,
            "returncode": self.returncode,
            "terminating_signal": self.terminating_signal,
            "elapsed_seconds": self.elapsed_seconds,
            "classification": self.classification,
            "correct": self.correct,
            "detail": self.detail,
            "rccl_output_path": self.rccl_output_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "compatibility_revision_id": self.compatibility_revision_id,
            "attempt": self.attempt,
            "plan_verification": self.plan_verification,
            "qualification_key": self.qualification_key,
        }


def build_command(
    case: RcclCase, *, binary: str, visible_devices: tuple[int, ...],
    rccl_output_path: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Construct the RCCL Tests invocation (RQ02) for one case.

    One tuple per process -- the caller must never reuse a persistent RCCL
    process across cases (this is the whole point of crash isolation).
    """
    size = str(case.byte_count)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = ",".join(str(d) for d in visible_devices)
    # Deliberately NOT setting HIP_ENABLE_DEFERRED_LOADING=0 here (RQ02's
    # original draft included it, matching the runbook's rationale of
    # exposing missing device code early). Real hardware evidence from this
    # investigation proved that flag forces eager resolution of a dead
    # fp8 reduction-kernel symbol that RDNA hardware never compiles and
    # normal lazy loading never calls -- a false-positive crash unrelated
    # to any real code-object gap, independent of RCCL build/config. See
    # HI138/docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md.
    env["NCCL_DEBUG"] = "INFO"
    env["RCCL_OVERRIDE_ALGO"] = case.algorithm
    env["RCCL_OVERRIDE_PROTO"] = case.protocol

    command = (
        binary,
        "-b", size,
        "-e", size,
        "-g", str(case.topology.rank_count),
        "-n", "5",
        "-w", "1",
        "-c", "1",
        "-T", "20",
        "-M", "1",
        "-Z", "json",
        "-x", rccl_output_path,
    )
    return command, env


# GP07 plan_verification values (gpt-dev-agent review, req_acf7c8da985f4f17):
# a separate stable field from `detail`'s free text, so machine consumers
# (the campaign driver, GP08's qualification lookup) don't have to parse
# prose to tell "RCCL explicitly declined" from "RCCL silently substituted
# a different plan" from "we never actually confirmed what ran."
PLAN_VERIFIED = "verified"
PLAN_SUBSTITUTED = "plan_substituted"
PLAN_UNVERIFIED = "unverified"
PLAN_EXPLICIT_DECLINE = "explicit_declined"


def _classify(
    *, returncode: int | None, term_signal: int | None, timed_out: bool,
    combined_output: str, rccl_json: list | None,
    requested_algorithm: str | None = None, requested_protocol: str | None = None,
    observed_algorithm: str | None = None, observed_protocol: str | None = None,
) -> tuple[str, bool | None, str, str | None]:
    """Return (classification, correct, detail, plan_verification). Order
    matters: check the most specific/actionable signal before falling back
    to a bare nonzero-exit guess."""
    if timed_out:
        return TIMEOUT, None, "outer harness timeout expired", None

    if any(marker in combined_output for marker in _RCCL_TEST_TIMEOUT_MARKERS):
        return TIMEOUT, None, "RCCL Tests' own internal -T test timeout expired", None

    # DEVICE_LOST takes precedence over SIGNAL/GPU_FAULT -- it is what the
    # campaign-level safety rule keys off (recheck GPU health, possibly
    # stop the whole matrix), not merely "this one case crashed."
    if any(marker in combined_output for marker in _DEVICE_LOST_MARKERS):
        return DEVICE_LOST, None, "device-loss marker found in output", None

    if term_signal is not None:
        return SIGNAL, None, f"terminated by signal {term_signal}", None

    if any(marker in combined_output for marker in _GPU_FAULT_MARKERS):
        return GPU_FAULT, None, "GPU/HIP fault marker found in output", None

    unsupported_scan_text = "\n".join(
        line for line in combined_output.splitlines()
        if not any(benign in line for benign in _UNSUPPORTED_BENIGN_MARKERS)
    )
    if any(marker in unsupported_scan_text for marker in _UNSUPPORTED_MARKERS):
        return (
            UNSUPPORTED, None, "RCCL declined the requested plan as unsupported",
            PLAN_EXPLICIT_DECLINE,
        )

    if any(marker in combined_output for marker in _INIT_FAILURE_MARKERS) and returncode:
        return INIT_FAILURE, None, "communicator initialization failed", None

    if returncode is None:
        return HARNESS_FAILURE, None, "process produced no return code", None

    if returncode != 0:
        return LAUNCH_FAILURE, None, f"nonzero exit code {returncode}, no specific marker matched", None

    if rccl_json is None:
        return HARNESS_FAILURE, None, "process exited 0 but produced no parseable RCCL JSON output", None

    # Real -Z json output (verified on real hardware, RCCL Tests
    # develop_deprecated:40b1b17) is a JSON ARRAY of per-pass records
    # (typically in-place/out-of-place), each carrying a "wrong" field as a
    # STRING count (e.g. "0"), not a top-level {"errors": N} object as the
    # runsheet's illustrative example assumed. No algorithm/protocol/
    # channels fields appear in the JSON at all (those are -M 1's
    # human-readable stdout table only).
    if not isinstance(rccl_json, list) or not rccl_json:
        return HARNESS_FAILURE, None, "process exited 0 but RCCL JSON output was not a non-empty array", None

    total_wrong = 0
    for record in rccl_json:
        if not isinstance(record, dict) or "wrong" not in record:
            return HARNESS_FAILURE, None, f"malformed RCCL JSON record: {record!r}", None
        try:
            total_wrong += int(record["wrong"])
        except (TypeError, ValueError):
            return HARNESS_FAILURE, None, f"non-integer 'wrong' field: {record['wrong']!r}", None

    if total_wrong != 0:
        return (
            WRONG_RESULT, False,
            f"RCCL reported {total_wrong} correctness error(s) across {len(rccl_json)} record(s)",
            None,
        )

    # GP07 (gpt-dev-agent findings, 2026-09-02, requests req_ee04ccbecc814cdb
    # then req_acf7c8da985f4f17): a clean exit with zero correctness errors
    # is NOT sufficient for PASS on its own -- RCCL_OVERRIDE_ALGO/PROTO
    # constrain selection, they don't guarantee it, so RCCL can silently
    # execute a different plan than requested and still exit 0/correct. A
    # qualification result must attest to the PLAN actually qualified, not
    # merely "some plan worked."
    if requested_algorithm is not None or requested_protocol is not None:
        # Second review round: failing to even OBSERVE the plan (no -M 1
        # table line matched at all) must NOT default to PASS either --
        # that silently treats "we never confirmed what ran" the same as
        # "we confirmed the right thing ran." Fail closed to HARNESS_FAILURE
        # instead, distinguishable via plan_verification=PLAN_UNVERIFIED.
        if observed_algorithm is None or observed_protocol is None:
            return (
                HARNESS_FAILURE, None,
                "clean/correct exit but the actually-executed algorithm/"
                "protocol could not be verified from output (no -M 1 table "
                "line matched) -- requested plan was never confirmed",
                PLAN_UNVERIFIED,
            )
        if (
            observed_algorithm.upper() != requested_algorithm.upper()
            or observed_protocol.upper() != requested_protocol.upper()
        ):
            return (
                UNSUPPORTED, None,
                f"requested {requested_algorithm}/{requested_protocol} but RCCL "
                f"selected {observed_algorithm}/{observed_protocol}",
                PLAN_SUBSTITUTED,
            )
        return PASS, True, "clean exit, correctness check passed, plan verified", PLAN_VERIFIED

    return PASS, True, "clean exit, correctness check passed", None


# -M 1's human-readable table ends each size row with "... algo proto
# nchannels" (verified against real RCCL Tests stdout, e.g.
# "    RING    SIMPLE           2"). Not present in -Z json output at all
# -- this is the only place RCCL Tests reports what was ACTUALLY selected,
# which per RQ02 must be checked against what was requested (RCCL_OVERRIDE_
# ALGO/PROTO constrain selection, they don't guarantee it).
_OBSERVED_PLAN_RE = re.compile(
    r"^\s*(RING|TREE|COLLNET_DIRECT|COLLNET_CHAIN|NVLS|NVLS_TREE|PAT)\s+"
    r"(LL|LL128|SIMPLE)\s+(\d+)\s*$", re.MULTILINE,
)


def _parse_observed_plan(combined_output: str) -> tuple[str | None, str | None, int | None]:
    match = _OBSERVED_PLAN_RE.search(combined_output)
    if match is None:
        return None, None, None
    algo, proto, channels = match.groups()
    return algo, proto, int(channels)


class CompatibilityManifestMismatch(RuntimeError):
    """Raised when an output directory already holds evidence from a
    DIFFERENT RCCL compatibility revision than the one now being run.

    Enforced, not merely encouraged (gpt-dev-agent review,
    req_acf7c8da985f4f17): the same case_id+attempt under a second RCCL
    build could otherwise silently overwrite the first build's raw
    artifacts. Namespacing output_dir by revision_id (below) prevents the
    overwrite in the common case; this check catches the harder case of
    the SAME namespaced directory being reused for two builds that happen
    to share a revision_id string but differ in some other recorded field
    (e.g. a corrected build_config) -- fail loudly rather than silently
    trust a stale manifest.
    """


_COMPATIBILITY_MANIFEST_NAME = "compatibility.json"


def _resolve_case_output_dir(
    output_dir: Path, compatibility: RcclCompatibilityRevision | None,
) -> Path:
    """When `compatibility` is supplied, artifacts are written under
    output_dir/<revision_id>/ automatically -- enforced namespacing, not
    left to the caller to remember. Writes (or verifies against) a
    `compatibility.json` manifest in that namespaced directory so a second
    run under a DIFFERENT compatibility that happens to resolve to the
    same directory is caught rather than silently overwriting evidence.
    """
    if compatibility is None:
        return output_dir

    namespaced = output_dir / compatibility.revision_id
    namespaced.mkdir(parents=True, exist_ok=True)
    manifest_path = namespaced / _COMPATIBILITY_MANIFEST_NAME
    this_manifest = compatibility.to_json()
    if manifest_path.exists():
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompatibilityManifestMismatch(
                f"{manifest_path} exists but could not be parsed as the "
                f"expected compatibility manifest: {exc}"
            ) from exc
        if stored_manifest != this_manifest:
            raise CompatibilityManifestMismatch(
                f"{namespaced} already holds evidence for a different "
                f"compatibility revision (stored={stored_manifest!r}, "
                f"requested={this_manifest!r}) despite sharing "
                f"revision_id={compatibility.revision_id!r} -- refusing to "
                "overwrite; use a distinct revision_id or a different "
                "output_dir."
            )
    else:
        manifest_path.write_text(json.dumps(this_manifest, indent=2), encoding="utf-8")

    return namespaced


def run_case(
    case: RcclCase, *, binary: str, visible_devices: tuple[int, ...],
    output_dir: Path, outer_timeout: float = 30.0,
    attempt: int | None = None,
    compatibility: RcclCompatibilityRevision | None = None,
) -> RcclCaseResult:
    """Run exactly one case in its own subprocess/process-group.

    Never retries automatically (RQ02). The caller decides whether/when to
    re-run after inspecting the classification.

    ``attempt`` and ``compatibility`` are both optional and default to the
    prior (pre-GP07) behaviour when omitted: ``attempt=None`` produces the
    exact same filenames as before (no suffix), so a caller that doesn't
    care about repetitions sees no change. A caller running a repeated
    qualification (e.g. the runbook's 20-fresh-process P1.12 gate) MUST
    pass a distinct ``attempt`` per rep -- otherwise every rep's
    stdout/stderr/rccl.json overwrites the previous one while cases.jsonl
    keeps accumulating rows that no longer match any file on disk (GP07's
    real found bug).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Raises InsufficientCompatibilityIdentity / CompatibilityManifestMismatch
    # -- deliberately propagated before any subprocess work starts.
    case_output_dir = _resolve_case_output_dir(output_dir, compatibility)
    case_id = case.case_id
    suffix = "" if attempt is None else f"__attempt{attempt}"
    stdout_path = case_output_dir / f"{case_id}{suffix}.stdout.log"
    stderr_path = case_output_dir / f"{case_id}{suffix}.stderr.log"
    rccl_output_path = case_output_dir / f"{case_id}{suffix}.rccl.json"

    command, env = build_command(
        case, binary=binary, visible_devices=visible_devices,
        rccl_output_path=str(rccl_output_path),
    )

    timed_out = False
    term_signal: int | None = None
    returncode: int | None = None
    start = time.monotonic()

    is_windows = sys.platform.startswith("win")
    popen_kwargs: dict = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if is_windows else {"start_new_session": True}
    )

    try:
        with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
            proc = subprocess.Popen(
                command, env=env, stdout=out_f, stderr=err_f, **popen_kwargs,
            )
            try:
                returncode = proc.wait(timeout=outer_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if is_windows:
                    # No process-group SIGKILL on Windows -- terminate the
                    # whole tree via taskkill (production RCCL qualification
                    # runs on Linux/Brutus; this branch exists so the harness
                    # is testable on a Windows dev machine).
                    subprocess.run(
                        ("taskkill", "/F", "/T", "/PID", str(proc.pid)),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                else:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait()
                returncode = proc.returncode
    except (OSError, FileNotFoundError) as exc:
        elapsed = time.monotonic() - start
        return RcclCaseResult(
            schema_version=SCHEMA_VERSION, case_id=case_id,
            topology_id=case.topology.topology_id,
            device_arches=case.topology.device_arches,
            diagnostic_visible_devices=visible_devices,
            element_count=case.element_count, dtype=case.dtype,
            byte_count=case.byte_count, algorithm=case.algorithm,
            protocol=case.protocol, requested_channels=case.channels,
            observed_algorithm=None, observed_protocol=None,
            observed_channels=None, returncode=None, terminating_signal=None,
            elapsed_seconds=elapsed, classification=HARNESS_FAILURE,
            correct=None, detail=f"failed to launch subprocess: {exc}",
            rccl_output_path=str(rccl_output_path), stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            compatibility_revision_id=compatibility.revision_id if compatibility else None,
            attempt=attempt,
        )

    elapsed = time.monotonic() - start
    if returncode is not None and returncode < 0:
        term_signal = -returncode

    combined_output = ""
    for path in (stdout_path, stderr_path):
        try:
            combined_output += path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass

    rccl_json = None
    if rccl_output_path.exists():
        try:
            rccl_json = json.loads(rccl_output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rccl_json = None

    observed_algorithm, observed_protocol, observed_channels = _parse_observed_plan(combined_output)

    classification, correct, detail, plan_verification = _classify(
        returncode=returncode, term_signal=term_signal, timed_out=timed_out,
        combined_output=combined_output, rccl_json=rccl_json,
        requested_algorithm=case.algorithm, requested_protocol=case.protocol,
        observed_algorithm=observed_algorithm, observed_protocol=observed_protocol,
    )

    # Raises InsufficientCompatibilityIdentity if `compatibility` was
    # supplied but lacks a durable identity component (library_build_id or
    # rccl_source_revision) -- deliberately propagated, not swallowed: a
    # qualification result recorded under insufficient identity is worse
    # than one that fails loudly at record time (GP07/gpt-dev-agent review).
    revision_id = compatibility.revision_id if compatibility else None

    return RcclCaseResult(
        schema_version=SCHEMA_VERSION, case_id=case_id,
        topology_id=case.topology.topology_id,
        device_arches=case.topology.device_arches,
        diagnostic_visible_devices=visible_devices,
        element_count=case.element_count, dtype=case.dtype,
        byte_count=case.byte_count, algorithm=case.algorithm,
        protocol=case.protocol, requested_channels=case.channels,
        observed_algorithm=observed_algorithm, observed_protocol=observed_protocol,
        observed_channels=observed_channels, returncode=returncode,
        terminating_signal=term_signal, elapsed_seconds=elapsed,
        classification=classification, correct=correct, detail=detail,
        plan_verification=plan_verification,
        compatibility_revision_id=revision_id,
        qualification_key=(
            qualification_key(compatibility, case_id) if compatibility else None
        ),
        attempt=attempt,
        rccl_output_path=str(rccl_output_path), stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def append_result(result: RcclCaseResult, cases_jsonl_path: Path) -> None:
    """Append-only write to the campaign's cases.jsonl (RQ15)."""
    cases_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_json_row()) + "\n")
